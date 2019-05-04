""" Different model components to use in building the overall model.

The main component of interest is SentenceEncoder, which all the models use. """
import json

import h5py
import numpy
import torch
import torch.nn as nn
import torch.utils.data
import torch.utils.data.distributed
from allennlp.common.checks import ConfigurationError
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper
from allennlp.models.model import Model
from allennlp.modules import Highway, Seq2SeqEncoder, SimilarityFunction, TimeDistributed
from allennlp.modules.elmo_lstm import ElmoLstm

# StackedSelfAttentionEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features, add_sentence_boundary_token_ids

from ..bert.utils import BertEmbedderModule
from ..tasks.tasks import PairClassificationTask, PairRegressionTask
from ..utils import utils
from ..utils.utils import MaskedMultiHeadSelfAttention
from .onlstm.ON_LSTM import ONLSTMStack
from .prpn.PRPN import PRPN


class NullPhraseLayer(nn.Module):
    """ Dummy phrase layer that does nothing. Exists solely for API compatibility. """

    def __init__(self, input_dim: int):
        super(NullPhraseLayer, self).__init__()
        self.input_dim = input_dim

    def get_input_dim(self):
        return self.input_dim

    def get_output_dim(self):
        return 0

    def forward(self, embs, mask):
        return None


class SentenceEncoder(Model):
    """ Given a sequence of tokens, embed each token and pass through a sequence encoder. """

    # NOTE: Do not apply dropout to the input of this module. Will be applied
    # internally.

    def __init__(
        self,
        vocab,
        text_field_embedder,
        num_highway_layers,
        phrase_layer,
        skip_embs=True,
        cove_layer=None,
        dropout=0.2,
        mask_lstms=True,
        sep_embs_for_skip=False,
        initializer=InitializerApplicator(),
    ):
        super(SentenceEncoder, self).__init__(vocab)

        if text_field_embedder is None:
            self._text_field_embedder = lambda x: x
            d_emb = 0
            self._highway_layer = lambda x: x
        else:
            self._text_field_embedder = text_field_embedder
            d_emb = text_field_embedder.get_output_dim()
            self._highway_layer = TimeDistributed(Highway(d_emb, num_highway_layers))

        self._phrase_layer = phrase_layer
        self._cove_layer = cove_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.skip_embs = skip_embs
        self.sep_embs_for_skip = sep_embs_for_skip
        d_inp_phrase = self._phrase_layer.get_input_dim()
        self.output_dim = phrase_layer.get_output_dim() + (skip_embs * d_inp_phrase)

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, sent, task, reset=True):
        # pylint: disable=arguments-differ
        """
        Args:
            - sent (Dict[str, torch.LongTensor]): From a ``TextField``.
            - task (Task): Used by the _text_field_embedder to pick the correct output
                           ELMo representation.
            - reset (Bool): if True, manually reset the states of the ELMo LSTMs present
                (if using BiLM or ELMo embeddings). Set False, if want to preserve statefulness.
        Returns:
            - sent_enc (torch.FloatTensor): (b_size, seq_len, d_emb)
                the padded values in sent_enc are set to 0
            - sent_mask (torch.FloatTensor): (b_size, seq_len, d_emb); all 0/1s
        """
        if reset:
            self.reset_states()

        # Embeddings
        # Note: These highway modules are actually identity functions by
        # default.
        is_pair_task = isinstance(task, (PairClassificationTask, PairRegressionTask))

        # General sentence embeddings (for sentence encoder).
        # Skip this for probing runs that don't need it.
        if not isinstance(self._phrase_layer, NullPhraseLayer):
            if isinstance(self._text_field_embedder, BertEmbedderModule):
                word_embs_in_context = self._text_field_embedder(sent, is_pair_task=is_pair_task)

            else:
                word_embs_in_context = self._text_field_embedder(sent)
            word_embs_in_context = self._highway_layer(word_embs_in_context)
        else:
            word_embs_in_context = None

        # Task-specific sentence embeddings (e.g. custom ELMo weights).
        # Skip computing this if it won't be used.
        if self.sep_embs_for_skip:
            if isinstance(self._text_field_embedder, BertEmbedderModule):
                task_word_embs_in_context = self._text_field_embedder(
                    sent, task._classifier_name, is_pair_task=is_pair_task
                )

            else:
                task_word_embs_in_context = self._text_field_embedder(sent, task._classifier_name)
            task_word_embs_in_context = self._highway_layer(task_word_embs_in_context)
        else:
            task_word_embs_in_context = None

        # Make sure we're embedding /something/
        assert (word_embs_in_context is not None) or (task_word_embs_in_context is not None)

        if self._cove_layer is not None:
            # Slightly wasteful as this repeats the GloVe lookup internally,
            # but this allows CoVe to be used alongside other embedding models
            # if we want to.
            sent_lens = torch.ne(sent["words"], self.pad_idx).long().sum(dim=-1).data
            # CoVe doesn't use <SOS> or <EOS>, so strip these before running.
            # Note that we need to also drop the last column so that CoVe returns
            # the right shape. If all inputs have <EOS> then this will be the
            # only thing clipped.
            sent_cove_embs_raw = self._cove_layer(sent["words"][:, 1:-1], sent_lens - 2)
            pad_col = torch.zeros(
                sent_cove_embs_raw.size()[0],
                1,
                sent_cove_embs_raw.size()[2],
                dtype=sent_cove_embs_raw.dtype,
                device=sent_cove_embs_raw.device,
            )
            sent_cove_embs = torch.cat([pad_col, sent_cove_embs_raw, pad_col], dim=1)
            if word_embs_in_context is not None:
                word_embs_in_context = torch.cat([word_embs_in_context, sent_cove_embs], dim=-1)
            if task_word_embs_in_context is not None:
                task_word_embs_in_context = torch.cat(
                    [task_word_embs_in_context, sent_cove_embs], dim=-1
                )

        if word_embs_in_context is not None:
            word_embs_in_context = self._dropout(word_embs_in_context)
        if task_word_embs_in_context is not None:
            task_word_embs_in_context = self._dropout(task_word_embs_in_context)

        # The rest of the model
        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None
        if word_embs_in_context is not None:
            if isinstance(self._phrase_layer, ONLSTMStack) or isinstance(self._phrase_layer, PRPN):
                # The ONLSTMStack or PRPN takes the raw words as input and computes
                # embeddings separately.
                sent_enc, _ = self._phrase_layer(
                    torch.transpose(sent["words"], 0, 1), sent_lstm_mask
                )
                sent_enc = torch.transpose(sent_enc, 0, 1)
            else:
                sent_enc = self._phrase_layer(word_embs_in_context, sent_lstm_mask)
        else:
            sent_enc = None

        # ELMoLSTM returns all layers, we just want to use the top layer
        sent_enc = sent_enc[-1] if isinstance(self._phrase_layer, BiLMEncoder) else sent_enc
        sent_enc = self._dropout(sent_enc) if sent_enc is not None else sent_enc
        if self.skip_embs:
            # Use skip connection with original sentence embs or task sentence
            # embs
            skip_vec = task_word_embs_in_context if self.sep_embs_for_skip else word_embs_in_context
            utils.assert_for_log(
                skip_vec is not None,
                "skip_vec is none - perhaps embeddings are not configured " "properly?",
            )
            if isinstance(self._phrase_layer, NullPhraseLayer):
                sent_enc = skip_vec
            else:
                sent_enc = torch.cat([sent_enc, skip_vec], dim=-1)

        sent_mask = sent_mask.unsqueeze(dim=-1)
        pad_mask = sent_mask == 0

        assert sent_enc is not None
        sent_enc = sent_enc.masked_fill(pad_mask, 0)
        return sent_enc, sent_mask

    def reset_states(self):
        """ Reset ELMo if present; reset BiLM (ELMoLSTM) states if present """
        if "token_embedder_elmo" in [
            name for name, _ in self._text_field_embedder.named_children()
        ] and "_elmo" in [
            name for name, _ in self._text_field_embedder.token_embedder_elmo.named_children()
        ]:
            self._text_field_embedder.token_embedder_elmo._elmo._elmo_lstm._elmo_lstm.reset_states()  # noqa # eek.
        if isinstance(self._phrase_layer, BiLMEncoder):
            self._phrase_layer.reset_states()


class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface to comply with SentEncoder
    See base class: ElmoLstm
    """

    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2


class BoWSentEncoder(Model):
    """ Bag-of-words sentence encoder """

    def __init__(self, vocab, text_field_embedder, initializer=InitializerApplicator()):
        super(BoWSentEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self.output_dim = text_field_embedder.get_output_dim()
        initializer(self)

    def forward(self, sent, task):
        # pylint: disable=arguments-differ
        """
        Args:
            - sent (Dict[str, torch.LongTensor]): From a ``TextField``.
            - task: Ignored.

        Returns
            - word_embs (torch.FloatTensor): (b_size, seq_len, d_emb)
                TODO: check what the padded values in word_embs are (0 or -inf or something else?)
            - word_mask (torch.FloatTensor): (b_size, seq_len, d_emb); all 0/1s
        """
        word_embs = self._text_field_embedder(sent)
        word_mask = util.get_text_field_mask(sent).float()
        return word_embs, word_mask  # need to get # nonzero elts


class PRPNPhraseLayer(Model):
    """
    Implementation of PRPN (Shen et al., 2018) as a phrase layer for sentence encoder.
    PRPN has a parser component that learns the latent constituency trees jointly with a
    downstream task.
    """

    def __init__(
        self,
        vocab,
        d_word,
        d_hid,
        n_layers_enc,
        n_slots,
        n_lookback,
        resolution,
        dropout,
        idropout,
        rdropout,
        res,
        embedder,
        batch_size,
        initializer=InitializerApplicator(),
    ):
        super(PRPNPhraseLayer, self).__init__(vocab)

        self.prpnlayer = PRPN(
            ninp=d_word,
            nhid=d_hid,
            nlayers=n_layers_enc,
            nslots=n_slots,
            nlookback=n_lookback,
            resolution=resolution,
            dropout=dropout,
            idropout=idropout,
            rdropout=rdropout,
            res=res,
            batch_size=batch_size,
            embedder=embedder,
            phrase_layer=None,
        )
        initializer(self)

    def get_input_dim(self):
        return self.prpnlayer.ninp

    def get_output_dim(self):
        return self.prpnlayer.ninp


class ONLSTMPhraseLayer(Model):
    """
    Implementation of ON-LSTM (Shen et al., 2019) as a phrase layer for sentence encoder.
    ON-LSTM is designed to add syntactic inductive bias to LSTM,
    and learns the latent constituency trees jointly with a downstream task.
    """

    def __init__(
        self,
        vocab,
        d_word,
        d_hid,
        n_layers_enc,
        chunk_size,
        onlstm_dropconnect,
        onlstm_dropouti,
        dropout,
        onlstm_dropouth,
        embedder,
        batch_size,
        initializer=InitializerApplicator(),
    ):
        super(ONLSTMPhraseLayer, self).__init__(vocab)
        self.onlayer = ONLSTMStack(
            [d_word] + [d_hid] * (n_layers_enc - 1) + [d_word],
            chunk_size=chunk_size,
            dropconnect=onlstm_dropconnect,
            dropouti=onlstm_dropouti,
            dropout=dropout,
            dropouth=onlstm_dropouth,
            embedder=embedder,
            phrase_layer=None,
            batch_size=batch_size,
        )
        initializer(self)

    def get_input_dim(self):
        return self.onlayer.layer_sizes[0]

    def get_output_dim(self):
        return self.onlayer.layer_sizes[-1]


class Pooler(nn.Module):
    """ Do pooling, possibly with a projection beforehand """

    def __init__(self, project=True, d_inp=512, d_proj=512, pool_type="max"):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

    def forward(self, sequence, mask):
        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)
        pad_mask = mask == 0
        proj_seq = self.project(sequence)  # linear project each hid state
        if self.pool_type == "max":
            proj_seq = proj_seq.masked_fill(pad_mask, -float("inf"))
            seq_emb = proj_seq.max(dim=1)[0]
        elif self.pool_type == "mean":
            proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1)
        elif self.pool_type == "final":
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs)
        elif self.pool_type == "first":
            seq_emb = proj_seq[:, 0]
        return seq_emb


class Classifier(nn.Module):
    """ Logistic regression or MLP classifier """

    # NOTE: Expects dropout to have already been applied to its input.

    def __init__(self, d_inp, n_classes, cls_type="mlp", dropout=0.2, d_hid=512):
        super(Classifier, self).__init__()
        if cls_type == "log_reg":
            classifier = nn.Linear(d_inp, n_classes)
        elif cls_type == "mlp":
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, n_classes),
            )
        elif cls_type == "fancy_mlp":  # What they did in Infersent.
            classifier = nn.Sequential(
                nn.Linear(d_inp, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(dropout),
                nn.Linear(d_hid, d_hid),
                nn.Tanh(),
                nn.LayerNorm(d_hid),
                nn.Dropout(p=dropout),
                nn.Linear(d_hid, n_classes),
            )
        else:
            raise ValueError("Classifier type %s not found" % type)
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits

    @classmethod
    def from_params(cls, d_inp, n_classes, params):
        return cls(
            d_inp,
            n_classes,
            cls_type=params["cls_type"],
            dropout=params["dropout"],
            d_hid=params["d_hid"],
        )


class SingleClassifier(nn.Module):
    """ Thin wrapper around a set of modules. For single-sentence classification. """

    def __init__(self, pooler, classifier):
        super(SingleClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, sent, mask, idxs=[]):
        """ Assumes batch_size x seq_len x d_emb """
        emb = self.pooler(sent, mask)

        # append any specific token representations, e.g. for WiC task
        ctx_embs = []
        for idx in [i.long() for i in idxs]:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1).expand([-1, -1, sent.size(-1)])
            ctx_emb = sent.gather(dim=1, index=idx)
            ctx_embs.append(ctx_emb.squeeze(dim=1))
        final_emb = torch.cat([emb] + ctx_embs, dim=-1)

        logits = self.classifier(final_emb)
        return logits


class PairClassifier(nn.Module):
    """ Thin wrapper around a set of modules.
    For sentence pair classification.
    Pooler specifies how to aggregate inputted sequence of vectors.
    Also allows for use of specific token representations to be addded to the overall
    representation
    """

    def __init__(self, pooler, classifier, attn=None):
        super(PairClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier
        self.attn = attn

    def forward(self, s1, s2, mask1, mask2, idx1=[], idx2=[]):
        """ s1, s2: sequences of hidden states corresponding to sentence 1,2
            mask1, mask2: binary mask corresponding to non-pad elements
            idx{1,2}: indexes of particular tokens to extract in sentence {1, 2}
                and append to the representation, e.g. for WiC
        """
        mask1 = mask1.squeeze(-1) if len(mask1.size()) > 2 else mask1
        mask2 = mask2.squeeze(-1) if len(mask2.size()) > 2 else mask2
        if self.attn is not None:
            s1, s2 = self.attn(s1, s2, mask1, mask2)
        emb1 = self.pooler(s1, mask1)
        emb2 = self.pooler(s2, mask2)

        s1_ctx_embs = []
        for idx in [i.long() for i in idx1]:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1).expand([-1, -1, s1.size(-1)])
            s1_ctx_emb = s1.gather(dim=1, index=idx)
            s1_ctx_embs.append(s1_ctx_emb.squeeze(dim=1))
        emb1 = torch.cat([emb1] + s1_ctx_embs, dim=-1)

        s2_ctx_embs = []
        for idx in [i.long() for i in idx2]:
            if len(idx.shape) == 1:
                idx = idx.unsqueeze(-1)
            if len(idx.shape) == 2:
                idx = idx.unsqueeze(-1).expand([-1, -1, s2.size(-1)])
            s2_ctx_emb = s2.gather(dim=1, index=idx)
            s2_ctx_embs.append(s2_ctx_emb.squeeze(dim=1))
        emb2 = torch.cat([emb2] + s2_ctx_embs, dim=-1)

        pair_emb = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)
        logits = self.classifier(pair_emb)
        return logits


class AttnPairEncoder(Model):
    """
    Simplified version of BiDAF.

    Parameters
    ----------
    vocab : ``Vocabulary``
    modeling_layer : ``Seq2SeqEncoder``
        The encoder (with its own internal stacking) that we will use in after the bidirectional
        attention.
    dropout : ``float``, optional (default=0.2)
        If greater than 0, we will apply dropout with this probability after all encoders (pytorch
        LSTMs do not apply dropout to their last layer).
    mask_lstms : ``bool``, optional (default=True)
        If ``False``, we will skip passing the mask to the LSTM layers.  This gives a ~2x speedup,
        with only a slight performance decrease, if any.  We haven't experimented much with this
        yet, but have confirmed that we still get very similar performance with much faster
        training times.  We still use the mask for all softmaxes, but avoid the shuffling that's
        required when using masking with pytorch LSTMs.
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """

    def __init__(
        self,
        vocab,
        modeling_layer,
        dropout=0.2,
        mask_lstms=True,
        initializer=InitializerApplicator(),
    ):
        super(AttnPairEncoder, self).__init__(vocab)

        self._matrix_attention = DotProductMatrixAttention()
        self._modeling_layer = modeling_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)

        d_out_model = modeling_layer.get_output_dim()
        self.output_dim = d_out_model

        self._dropout = torch.nn.Dropout(p=dropout) if dropout > 0 else lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2, s1_mask, s2_mask):  # pylint: disable=arguments-differ
        """ """
        # Similarity matrix
        # Shape: (batch_size, s2_length, s1_length)
        similarity_mat = self._matrix_attention(s2, s1)

        # s2 representation
        # Shape: (batch_size, s2_length, s1_length)
        s2_s1_attn = util.masked_softmax(similarity_mat, s1_mask)
        # Shape: (batch_size, s2_length, encoding_dim)
        s2_s1_vectors = util.weighted_sum(s1, s2_s1_attn)
        # batch_size, seq_len, 4*enc_dim
        s2_w_context = torch.cat([s2, s2_s1_vectors], 2)

        # s1 representation, using same attn method as for the s2
        # representation
        s1_s2_attn = util.masked_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = util.weighted_sum(s2, s1_s2_attn)
        s1_w_context = torch.cat([s1, s1_s2_vectors], 2)

        modeled_s1 = self._dropout(self._modeling_layer(s1_w_context, s1_mask))
        modeled_s2 = self._dropout(self._modeling_layer(s2_w_context, s2_mask))
        return modeled_s1, modeled_s2

    @classmethod
    def from_params(cls, vocab, params):
        """ Initialize from a Params object """
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        dropout = params.pop("dropout", 0.2)
        initializer = InitializerApplicator.from_params(params.pop("initializer", []))

        mask_lstms = params.pop("mask_lstms", True)
        params.assert_empty(cls.__name__)
        return cls(
            vocab=vocab,
            modeling_layer=modeling_layer,
            dropout=dropout,
            mask_lstms=mask_lstms,
            initializer=initializer,
        )


class MaskedStackedSelfAttentionEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .  # noqa

    This encoder combines 3 layers in a 'block':

    1. A 2 layer FeedForward network.
    2. Multi-headed self attention, which uses 2 learnt linear projections
       to perform a dot-product similarity between every pair of elements
       scaled by the square root of the sequence length.
    3. Layer Normalisation.

    These are then stacked into ``num_layers`` layers.

    Parameters
    ----------
    input_dim : ``int``, required.
        The input dimension of the encoder.
    hidden_dim : ``int``, required.
        The hidden dimension used for the _input_ to self attention layers
        and the _output_ from the feedforward layers.
    projection_dim : ``int``, required.
        The dimension of the linear projections for the self-attention layers.
    feedforward_hidden_dim : ``int``, required.
        The middle dimension of the FeedForward network. The input and output
        dimensions are fixed to ensure sizes match up for the self attention layers.
    num_layers : ``int``, required.
        The number of stacked self attention -> feedfoward -> layer normalisation blocks.
    num_attention_heads : ``int``, required.
        The number of attention heads to use per layer.
    use_positional_encoding: ``bool``, optional, (default = True)
        Whether to add sinusoidal frequencies to the input tensor. This is strongly recommended,
        as without this feature, the self attention layers have no idea of absolute or relative
        position (as they are just computing pairwise similarity between vectors of elements),
        which can be important features for many tasks.
    dropout_prob : ``float``, optional, (default = 0.2)
        The dropout probability for the feedforward network.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        projection_dim,
        feedforward_hidden_dim,
        num_layers,
        num_attention_heads,
        use_positional_encoding=True,
        dropout_prob=0.2,
    ):
        super(MaskedStackedSelfAttentionEncoder, self).__init__()

        self._use_positional_encoding = use_positional_encoding
        self._attention_layers = []
        self._feedfoward_layers = []
        self._layer_norm_layers = []
        self._feed_forward_layer_norm_layers = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            feedfoward = FeedForward(
                feedfoward_input_dim,
                activations=[Activation.by_name("relu")(), Activation.by_name("linear")()],
                hidden_dims=[feedforward_hidden_dim, hidden_dim],
                num_layers=2,
                dropout=dropout_prob,
            )

            self.add_module("feedforward_{i}".format(feedfoward))
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_input_dim())
            self.add_module("feedforward_layer_norm_{i}".format(feedforward_layer_norm))
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            self_attention = MaskedMultiHeadSelfAttention(
                num_heads=num_attention_heads,
                input_dim=hidden_dim,
                attention_dim=projection_dim,
                values_dim=projection_dim,
            )
            self.add_module("self_attention_{i}".format(self_attention))
            self._attention_layers.append(self_attention)

            layer_norm = LayerNorm(self_attention.get_input_dim())
            self.add_module("layer_norm_{i}".format(layer_norm))
            self._layer_norm_layers.append(layer_norm)

            feedfoward_input_dim = hidden_dim

        self.dropout = torch.nn.Dropout(dropout_prob)
        self._input_dim = input_dim
        self._output_dim = self._attention_layers[-1].get_output_dim()
        self._output_layer_norm = LayerNorm(self._output_dim)

    def get_input_dim(self):
        return self._input_dim

    def get_output_dim(self):
        return self._output_dim

    def is_bidirectional(self):
        return 0

    def forward(self, inputs, mask):  # pylint: disable=arguments-differ
        if self._use_positional_encoding:
            output = add_positional_features(inputs)
        else:
            output = inputs
        for (attention, feedforward, feedforward_layer_norm, layer_norm) in zip(
            self._attention_layers,
            self._feedfoward_layers,
            self._feed_forward_layer_norm_layers,
            self._layer_norm_layers,
        ):
            cached_input = output
            # Project output of attention encoder through a feedforward
            # network and back to the input size for the next layer.
            # shape (batch_size, timesteps, input_size)
            feedforward_output = feedforward(feedforward_layer_norm(output))
            feedforward_output = self.dropout(feedforward_output)
            if feedforward_output.size() == cached_input.size():
                # First layer might have the wrong size for highway
                # layers, so we exclude it here.
                feedforward_output += cached_input
            # shape (batch_size, sequence_length, hidden_dim)
            attention_output = attention(layer_norm(feedforward_output), mask)
            output = self.dropout(attention_output) + feedforward_output
        return self._output_layer_norm(output)

    @classmethod
    def from_params(cls, params):
        input_dim = params.pop_int("input_dim")
        hidden_dim = params.pop_int("hidden_dim")
        projection_dim = params.pop_int("projection_dim", None)
        feedforward_hidden_dim = params.pop_int("feedforward_hidden_dim")
        num_layers = params.pop_int("num_layers", 2)
        num_attention_heads = params.pop_int("num_attention_heads", 3)
        use_positional_encoding = params.pop_bool("use_positional_encoding", True)
        dropout_prob = params.pop_float("dropout_prob", 0.2)
        params.assert_empty(cls.__name__)

        return cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            feedforward_hidden_dim=feedforward_hidden_dim,
            projection_dim=projection_dim,
            num_layers=num_layers,
            num_attention_heads=num_attention_heads,
            use_positional_encoding=use_positional_encoding,
            dropout_prob=dropout_prob,
        )


class ElmoCharacterEncoder(torch.nn.Module):
    """Just the ELMo character encoder that we ripped so we could use alone.

    Compute context sensitive token representation using pretrained biLM.

    This embedder has input character ids of size (batch_size, sequence_length, 50)
    and returns (batch_size, sequence_length + 2, embedding_dim), where embedding_dim
    is specified in the options file (typically 512).

    We add special entries at the beginning and end of each sequence corresponding
    to <S> and </S>, the beginning and end of sentence tokens.

    Note: this is a lower level class useful for advanced usage.  Most users should
    use ``ElmoTokenEmbedder`` or ``allennlp.modules.Elmo`` instead.

    Parameters
    ----------
    options_file : ``str``
        ELMo JSON options file
    weight_file : ``str``
        ELMo hdf5 weight file
    requires_grad: ``bool``, optional
        If True, compute gradient of ELMo parameters for fine tuning.

    The relevant section of the options file is something like:
    .. example-code::

        .. code-block:: python

            {'char_cnn': {
                'activation': 'relu',
                'embedding': {'dim': 4},
                'filters': [[1, 4], [2, 8], [3, 16], [4, 32], [5, 64]],
                'max_characters_per_token': 50,
                'n_characters': 262,
                'n_highway': 2
                }
            }
    """

    def __init__(self, options_file, weight_file, requires_grad=False):
        super(ElmoCharacterEncoder, self).__init__()

        with open(cached_path(options_file), "r") as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options["lstm"]["projection_dim"]
        self.requires_grad = requires_grad

        self._load_weights()

        # Cache the arrays for use in forward -- +1 due to masking.
        self._beginning_of_sentence_characters = torch.from_numpy(
            numpy.array(ELMoCharacterMapper.beginning_of_sentence_characters) + 1
        )
        self._end_of_sentence_characters = torch.from_numpy(
            numpy.array(ELMoCharacterMapper.end_of_sentence_characters) + 1
        )

    def get_output_dim(self):
        return self.output_dim

    def forward(self, inputs):  # pylint: disable=arguments-differ
        """
        Compute context insensitive token embeddings for ELMo representations.

        Parameters
        ----------
        inputs: ``torch.Tensor``
            Shape ``(batch_size, sequence_length, 50)`` of character ids representing the
            current batch.

        Returns
        -------
        Dict with keys:
        ``'token_embedding'``: ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2, embedding_dim)`` tensor with context
            insensitive token representations.
        ``'mask'``:  ``torch.Tensor``
            Shape ``(batch_size, sequence_length + 2)`` long tensor with sequence mask.
        """
        # Add BOS/EOS
        mask = ((inputs > 0).long().sum(dim=-1) > 0).long()
        character_ids_with_bos_eos, mask_with_bos_eos = add_sentence_boundary_token_ids(
            inputs, mask, self._beginning_of_sentence_characters, self._end_of_sentence_characters
        )

        # the character id embedding
        max_chars_per_token = self._options["char_cnn"]["max_characters_per_token"]
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token), self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options["char_cnn"]
        if cnn_options["activation"] == "tanh":
            activation = torch.nn.functional.tanh
        elif cnn_options["activation"] == "relu":
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, "char_conv_{}".format(i))
            convolved = conv(character_embedding)
            # (batch_size * sequence_length, n_filters for this width)
            convolved, _ = torch.max(convolved, dim=-1)
            convolved = activation(convolved)
            convs.append(convolved)

        # (batch_size * sequence_length, n_filters)
        token_embedding = torch.cat(convs, dim=-1)

        # apply the highway layers (batch_size * sequence_length, n_filters)
        token_embedding = self._highways(token_embedding)

        # final projection  (batch_size * sequence_length, embedding_dim)
        token_embedding = self._projection(token_embedding)

        # reshape to (batch_size, sequence_length, embedding_dim)
        batch_size, sequence_length, _ = character_ids_with_bos_eos.size()

        return token_embedding.view(batch_size, sequence_length, -1)[:, 1:-1, :]

    def _load_weights(self):
        self._load_char_embedding()
        self._load_cnn_weights()
        self._load_highway()
        self._load_projection()

    def _load_char_embedding(self):
        with h5py.File(cached_path(self._weight_file), "r") as fin:
            char_embed_weights = fin["char_embed"][...]

        weights = numpy.zeros(
            (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]), dtype="float32"
        )
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _load_cnn_weights(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        char_embed_dim = cnn_options["embedding"]["dim"]

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim, out_channels=num, kernel_size=width, bias=True
            )
            # load the weights
            with h5py.File(cached_path(self._weight_file), "r") as fin:
                weight = fin["CNN"]["W_cnn_{}".format(i)][...]
                bias = fin["CNN"]["b_cnn_{}".format(i)][...]

            w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module("char_conv_{}".format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn
        # filters
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options["n_highway"]

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            with h5py.File(cached_path(self._weight_file), "r") as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                w_transform = numpy.transpose(fin["CNN_high_{}".format(k)]["W_transform"][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * numpy.transpose(fin["CNN_high_{}".format(k)]["W_carry"][...])
                weight = numpy.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad

                b_transform = fin["CNN_high_{}".format(k)]["b_transform"][...]
                b_carry = -1.0 * fin["CNN_high_{}".format(k)]["b_carry"][...]
                bias = numpy.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self):
        cnn_options = self._options["char_cnn"]
        filters = cnn_options["filters"]
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(cached_path(self._weight_file), "r") as fin:
            weight = fin["CNN_proj"]["W_proj"][...]
            bias = fin["CNN_proj"]["b_proj"][...]
            self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad
