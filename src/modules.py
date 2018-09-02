''' Different model components to use in building the overall model.

The main component of interest is SentenceEncoder, which all the models use. '''
import os
import sys
import json
import logging as log
import h5py

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model
from allennlp.modules import Highway
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.elmo_lstm import ElmoLstm
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.similarity_functions import LinearSimilarity, DotProductSimilarity
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
# StackedSelfAttentionEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features
from .utils import MaskedMultiHeadSelfAttention

from .cnns.alexnet import alexnet
from .cnns.resnet import resnet101
from .cnns.inception import inception_v3

class NullPhraseLayer(nn.Module):
    ''' Dummy phrase layer that does nothing. Exists solely for API compatibility. '''
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
    ''' Given a sequence of tokens, embed each token and pass thru an LSTM. '''
    # NOTE: Do not apply dropout to the input of this module. Will be applied internally.

    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 skip_embs=True, cove_layer=None, dropout=0.2, mask_lstms=True,
                 sep_embs_for_skip=False, initializer=InitializerApplicator()):
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

    def forward(self, sent, task):
        # pylint: disable=arguments-differ
        """
        Args:
            - sent (Dict[str, torch.LongTensor]): From a ``TextField``.
            - task (Task): Used by the _text_field_embedder to pick the correct output
                           ELMo representation.
        Returns:
            - sent_enc (torch.FloatTensor): (b_size, seq_len, d_emb)
                TODO: check what the padded values in sent_enc are (0 or -inf or something else?)
            - sent_mask (torch.FloatTensor): (b_size, seq_len, d_emb); all 0/1s
        """
        # Embeddings
        # Note: These highway modules are actually identity functions by default.
        sent_embs = self._highway_layer(self._text_field_embedder(sent))
        # task_sent_embs only used if sep_embs_for_skip
        task_sent_embs = self._highway_layer(self._text_field_embedder(sent, task._classifier_name))

        if self._cove_layer is not None:
            # Slightly wasteful as this repeats the GloVe lookup internally,
            # but this allows CoVe to be used alongside other embedding models
            # if we want to.
            sent_lens = torch.ne(sent['words'], self.pad_idx).long().sum(dim=-1).data
            sent_cove_embs = self._cove_layer(sent['words'], sent_lens)
            sent_embs = torch.cat([sent_embs, sent_cove_embs], dim=-1)
            task_sent_embs = torch.cat([task_sent_embs, sent_cove_embs], dim=-1)

        sent_embs = self._dropout(sent_embs)
        task_sent_embs = self._dropout(task_sent_embs)

        # The rest of the model
        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None
        sent_enc = self._phrase_layer(sent_embs, sent_lstm_mask)

        # ELMoLSTM returns all layers, we just want to use the top layer
        if isinstance(self._phrase_layer, BiLMEncoder):
            sent_enc = sent_enc[-1]
        if sent_enc is not None:
            sent_enc = self._dropout(sent_enc)
        if self.skip_embs:
            # Use skip connection with original sentence embs or task sentence embs
            skip_vec = task_sent_embs if self.sep_embs_for_skip else sent_embs
            if isinstance(self._phrase_layer, NullPhraseLayer):
                sent_enc = skip_vec
            else:
                sent_enc = torch.cat([sent_enc, skip_vec], dim=-1)

        sent_mask = sent_mask.unsqueeze(dim=-1)
        return sent_enc, sent_mask

class BiLMEncoder(ElmoLstm):
    """Wrapper around BiLM to give it an interface to comply with SentEncoder
    See base class: ElmoLstm
    """
    def get_input_dim(self):
        return self.input_size

    def get_output_dim(self):
        return self.hidden_size * 2

class BoWSentEncoder(Model):
    ''' Bag-of-words sentence encoder '''

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


class Pooler(nn.Module):
    ''' Do pooling, possibly with a projection beforehand '''

    def __init__(self, d_inp, project=True, d_proj=512, pool_type='max'):
        super(Pooler, self).__init__()
        self.project = nn.Linear(d_inp, d_proj) if project else lambda x: x
        self.pool_type = pool_type

    def forward(self, sequence, mask):
        if len(mask.size()) < 3:
            mask = mask.unsqueeze(dim=-1)
        pad_mask = 1 - mask.byte().data
        if sequence.min().item() != float('-inf'):  # this will f up the loss
            #log.warn('Negative infinity detected')
            sequence.masked_fill(pad_mask, 0)
        proj_seq = self.project(sequence)

        if self.pool_type == 'max':
            proj_seq = proj_seq.masked_fill(pad_mask, -float('inf'))
            seq_emb = proj_seq.max(dim=1)[0]
        elif self.pool_type == 'mean':
            #proj_seq = proj_seq.masked_fill(pad_mask, 0)
            seq_emb = proj_seq.sum(dim=1) / mask.sum(dim=1)
        elif self.pool_type == 'final':
            idxs = mask.expand_as(proj_seq).sum(dim=1, keepdim=True).long() - 1
            seq_emb = proj_seq.gather(dim=1, index=idxs)
        return seq_emb

    @classmethod
    def from_params(cls, d_inp, d_proj, project=True):
        return cls(d_inp, d_proj=d_proj, project=project)


class Classifier(nn.Module):
    ''' Logistic regression or MLP classifier '''
    # NOTE: Expects dropout to have already been applied to its input.

    def __init__(self, d_inp, n_classes, cls_type='mlp', dropout=.2, d_hid=512):
        super(Classifier, self).__init__()
        if cls_type == 'log_reg':
            classifier = nn.Linear(d_inp, n_classes)
        elif cls_type == 'mlp':
            classifier = nn.Sequential(nn.Linear(d_inp, d_hid),
                                       nn.Tanh(), nn.LayerNorm(d_hid),
                                       nn.Dropout(dropout), nn.Linear(d_hid, n_classes))
        elif cls_type == 'fancy_mlp':  # What they did in Infersent.
            classifier = nn.Sequential(nn.Linear(d_inp, d_hid),
                                       nn.Tanh(), nn.LayerNorm(d_hid), nn.Dropout(dropout),
                                       nn.Linear(d_hid, d_hid), nn.Tanh(),
                                       nn.LayerNorm(d_hid), nn.Dropout(p=dropout),
                                       nn.Linear(d_hid, n_classes))
        else:
            raise ValueError("Classifier type %s not found" % type)
        self.classifier = classifier

    def forward(self, seq_emb):
        logits = self.classifier(seq_emb)
        return logits

    @classmethod
    def from_params(cls, d_inp, n_classes, params):
        return cls(d_inp, n_classes, cls_type=params["cls_type"],
                   dropout=params["dropout"], d_hid=params["d_hid"])


class SingleClassifier(nn.Module):
    ''' Thin wrapper around a set of modules. For single-sentence classification. '''

    def __init__(self, pooler, classifier):
        super(SingleClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier

    def forward(self, sent, mask):
        emb = self.pooler(sent, mask)
        logits = self.classifier(emb)
        return logits


class PairClassifier(nn.Module):
    ''' Thin wrapper around a set of modules. For sentence pair classification. '''

    def __init__(self, pooler, classifier, attn=None):
        super(PairClassifier, self).__init__()
        self.pooler = pooler
        self.classifier = classifier
        self.attn = attn

    def forward(self, s1, s2, mask1, mask2):
        mask1 = mask1.squeeze(-1) if len(mask1.size()) > 2 else mask1
        mask2 = mask2.squeeze(-1) if len(mask2.size()) > 2 else mask2
        if self.attn is not None:
            s1, s2 = self.attn(s1, s2, mask1, mask2)
        emb1 = self.pooler(s1, mask1)
        emb2 = self.pooler(s2, mask2)
        pair_emb = torch.cat([emb1, emb2, torch.abs(emb1 - emb2), emb1 * emb2], 1)
        logits = self.classifier(pair_emb)
        return logits


class AttnPairEncoder(Model):
    """
    Simplified version of BiDAF.

    Parameters
    ----------
    vocab : ``Vocabulary``
    attention_similarity_function : ``SimilarityFunction``
        The similarity function that we will use when comparing encoded passage and question
        representations.
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

    def __init__(self, vocab, modeling_layer, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator()):
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
        s2_s1_attn = util.last_dim_softmax(similarity_mat, s1_mask)
        # Shape: (batch_size, s2_length, encoding_dim)
        s2_s1_vectors = util.weighted_sum(s1, s2_s1_attn)
        # batch_size, seq_len, 4*enc_dim
        s2_w_context = torch.cat([s2, s2_s1_vectors], 2)

        # s1 representation, using same attn method as for the s2 representation
        s1_s2_attn = util.last_dim_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = util.weighted_sum(s2, s1_s2_attn)
        s1_w_context = torch.cat([s1, s1_s2_vectors], 2)

        modeled_s1 = self._dropout(self._modeling_layer(s1_w_context, s1_mask))
        modeled_s2 = self._dropout(self._modeling_layer(s2_w_context, s2_mask))
        return modeled_s1, modeled_s2

    @classmethod
    def from_params(cls, vocab, params):
        ''' Initialize from a Params object '''
        similarity_function = SimilarityFunction.from_params(params.pop("similarity_function"))
        modeling_layer = Seq2SeqEncoder.from_params(params.pop("modeling_layer"))
        dropout = params.pop('dropout', 0.2)
        initializer = InitializerApplicator.from_params(params.pop('initializer', []))

        mask_lstms = params.pop('mask_lstms', True)
        params.assert_empty(cls.__name__)
        return cls(vocab=vocab, attention_similarity_function=similarity_function,
                   modeling_layer=modeling_layer, dropout=dropout,
                   mask_lstms=mask_lstms, initializer=initializer)

# This class is identical to the one in allennlp.modules.seq2seq_encoders


class MaskedStackedSelfAttentionEncoder(Seq2SeqEncoder):
    # pylint: disable=line-too-long
    """
    Implements a stacked self-attention encoder similar to the Transformer
    architecture in `Attention is all you Need
    <https://www.semanticscholar.org/paper/Attention-Is-All-You-Need-Vaswani-Shazeer/0737da0767d77606169cbf4187b83e1ab62f6077>`_ .

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

    def __init__(self,
                 input_dim,
                 hidden_dim,
                 projection_dim,
                 feedforward_hidden_dim,
                 num_layers,
                 num_attention_heads,
                 use_positional_encoding=True,
                 dropout_prob=0.2):
        super(MaskedStackedSelfAttentionEncoder, self).__init__()

        self._use_positional_encoding = use_positional_encoding
        self._attention_layers = []
        self._feedfoward_layers = []
        self._layer_norm_layers = []
        self._feed_forward_layer_norm_layers = []

        feedfoward_input_dim = input_dim
        for i in range(num_layers):
            feedfoward = FeedForward(feedfoward_input_dim,
                                     activations=[Activation.by_name('relu')(),
                                                  Activation.by_name('linear')()],
                                     hidden_dims=[feedforward_hidden_dim, hidden_dim],
                                     num_layers=2,
                                     dropout=dropout_prob)

            self.add_module("feedforward_{i}".format(feedfoward))
            self._feedfoward_layers.append(feedfoward)

            feedforward_layer_norm = LayerNorm(feedfoward.get_input_dim())
            self.add_module("feedforward_layer_norm_{i}".format(feedforward_layer_norm))
            self._feed_forward_layer_norm_layers.append(feedforward_layer_norm)

            self_attention = MaskedMultiHeadSelfAttention(num_heads=num_attention_heads,
                                                          input_dim=hidden_dim,
                                                          attention_dim=projection_dim,
                                                          values_dim=projection_dim)
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
        for (attention,
             feedforward,
             feedforward_layer_norm,
             layer_norm) in zip(self._attention_layers,
                                self._feedfoward_layers,
                                self._feed_forward_layer_norm_layers,
                                self._layer_norm_layers):
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
        input_dim = params.pop_int('input_dim')
        hidden_dim = params.pop_int('hidden_dim')
        projection_dim = params.pop_int('projection_dim', None)
        feedforward_hidden_dim = params.pop_int("feedforward_hidden_dim")
        num_layers = params.pop_int("num_layers", 2)
        num_attention_heads = params.pop_int('num_attention_heads', 3)
        use_positional_encoding = params.pop_bool('use_positional_encoding', True)
        dropout_prob = params.pop_float("dropout_prob", 0.2)
        params.assert_empty(cls.__name__)

        return cls(input_dim=input_dim,
                   hidden_dim=hidden_dim,
                   feedforward_hidden_dim=feedforward_hidden_dim,
                   projection_dim=projection_dim,
                   num_layers=num_layers,
                   num_attention_heads=num_attention_heads,
                   use_positional_encoding=use_positional_encoding,
                   dropout_prob=dropout_prob)


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

    def __init__(self,
                 options_file,
                 weight_file,
                 requires_grad=False):
        super(ElmoCharacterEncoder, self).__init__()

        with open(cached_path(options_file), 'r') as fin:
            self._options = json.load(fin)
        self._weight_file = weight_file

        self.output_dim = self._options['lstm']['projection_dim']
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
            inputs,
            mask,
            self._beginning_of_sentence_characters,
            self._end_of_sentence_characters
        )

        # the character id embedding
        max_chars_per_token = self._options['char_cnn']['max_characters_per_token']
        # (batch_size * sequence_length, max_chars_per_token, embed_dim)
        character_embedding = torch.nn.functional.embedding(
            character_ids_with_bos_eos.view(-1, max_chars_per_token),
            self._char_embedding_weights
        )

        # run convolutions
        cnn_options = self._options['char_cnn']
        if cnn_options['activation'] == 'tanh':
            activation = torch.nn.functional.tanh
        elif cnn_options['activation'] == 'relu':
            activation = torch.nn.functional.relu
        else:
            raise ConfigurationError("Unknown activation")

        # (batch_size * sequence_length, embed_dim, max_chars_per_token)
        character_embedding = torch.transpose(character_embedding, 1, 2)
        convs = []
        for i in range(len(self._convolutions)):
            conv = getattr(self, 'char_conv_{}'.format(i))
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
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            char_embed_weights = fin['char_embed'][...]

        weights = numpy.zeros(
            (char_embed_weights.shape[0] + 1, char_embed_weights.shape[1]),
            dtype='float32'
        )
        weights[1:, :] = char_embed_weights

        self._char_embedding_weights = torch.nn.Parameter(
            torch.FloatTensor(weights), requires_grad=self.requires_grad
        )

    def _load_cnn_weights(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        char_embed_dim = cnn_options['embedding']['dim']

        convolutions = []
        for i, (width, num) in enumerate(filters):
            conv = torch.nn.Conv1d(
                in_channels=char_embed_dim,
                out_channels=num,
                kernel_size=width,
                bias=True
            )
            # load the weights
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                weight = fin['CNN']['W_cnn_{}'.format(i)][...]
                bias = fin['CNN']['b_cnn_{}'.format(i)][...]

            w_reshaped = numpy.transpose(weight.squeeze(axis=0), axes=(2, 1, 0))
            if w_reshaped.shape != tuple(conv.weight.data.shape):
                raise ValueError("Invalid weight file")
            conv.weight.data.copy_(torch.FloatTensor(w_reshaped))
            conv.bias.data.copy_(torch.FloatTensor(bias))

            conv.weight.requires_grad = self.requires_grad
            conv.bias.requires_grad = self.requires_grad

            convolutions.append(conv)
            self.add_module('char_conv_{}'.format(i), conv)

        self._convolutions = convolutions

    def _load_highway(self):
        # pylint: disable=protected-access
        # the highway layers have same dimensionality as the number of cnn filters
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)
        n_highway = cnn_options['n_highway']

        # create the layers, and load the weights
        self._highways = Highway(n_filters, n_highway, activation=torch.nn.functional.relu)
        for k in range(n_highway):
            # The AllenNLP highway is one matrix multplication with concatenation of
            # transform and carry weights.
            with h5py.File(cached_path(self._weight_file), 'r') as fin:
                # The weights are transposed due to multiplication order assumptions in tf
                # vs pytorch (tf.matmul(X, W) vs pytorch.matmul(W, X))
                w_transform = numpy.transpose(fin['CNN_high_{}'.format(k)]['W_transform'][...])
                # -1.0 since AllenNLP is g * x + (1 - g) * f(x) but tf is (1 - g) * x + g * f(x)
                w_carry = -1.0 * numpy.transpose(fin['CNN_high_{}'.format(k)]['W_carry'][...])
                weight = numpy.concatenate([w_transform, w_carry], axis=0)
                self._highways._layers[k].weight.data.copy_(torch.FloatTensor(weight))
                self._highways._layers[k].weight.requires_grad = self.requires_grad

                b_transform = fin['CNN_high_{}'.format(k)]['b_transform'][...]
                b_carry = -1.0 * fin['CNN_high_{}'.format(k)]['b_carry'][...]
                bias = numpy.concatenate([b_transform, b_carry], axis=0)
                self._highways._layers[k].bias.data.copy_(torch.FloatTensor(bias))
                self._highways._layers[k].bias.requires_grad = self.requires_grad

    def _load_projection(self):
        cnn_options = self._options['char_cnn']
        filters = cnn_options['filters']
        n_filters = sum(f[1] for f in filters)

        self._projection = torch.nn.Linear(n_filters, self.output_dim, bias=True)
        with h5py.File(cached_path(self._weight_file), 'r') as fin:
            weight = fin['CNN_proj']['W_proj'][...]
            bias = fin['CNN_proj']['b_proj'][...]
            self._projection.weight.data.copy_(torch.FloatTensor(numpy.transpose(weight)))
            self._projection.bias.data.copy_(torch.FloatTensor(bias))

            self._projection.weight.requires_grad = self.requires_grad
            self._projection.bias.requires_grad = self.requires_grad


class CNNEncoder(Model):
    ''' Given an image, get image features from last layer of specified CNN
        e.g., Resnet101, AlexNet, InceptionV3
        New! Preprocessed and indexed image features, so just load from json!'''

    def __init__(self, model_name, path, model=None):
        super(CNNEncoder, self).__init__(model_name)
        self.model_name = model_name
        self.model = self._load_model(model_name)
        self.feat_path = path + '/all_feats/'

    def _load_model(self, model_name):
        if model_name == 'alexnet':
            model = alexnet(pretrained=True)
        elif model_name == 'inception':
            model = inception_v3(pretrained=True)
        elif model_name == 'resnet':
            model = resnet101(pretrained=True)
        return model

    def _load_features(self, path, dataset):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_dataset = datasets.ImageFolder(
            path,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset)

        classes = [
            d for d in os.listdir(
                train_dataset.root) if os.path.isdir(
                os.path.join(
                    train_dataset.root,
                    d))]

        class_to_idx = {classes[i]: i for i in range(len(classes))}
        rev_class = {class_to_idx[key]: key for key in class_to_idx.keys()}

        feat_dict = {}
        for i, (input, target) in enumerate(train_loader):
            x = self.model.forward(input)
            feat_dict[rev_class[i]] = x.data
        return feat_dict

    def forward(self, img_id):
        '''
        Args: img_id that maps image -> sentence pairs in respective datasets.
        '''

        with open(self.feat_path + str(img_id) + '.json') as fd:
            feat_dict = json.load(fd)
        return feat_dict[list(feat_dict.keys())[0]] # has one key
