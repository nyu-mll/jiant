""" Different model components to use in building the overall model.

The main component of interest is SentenceEncoder, which all the models use. """

import torch
import torch.utils.data
import torch.utils.data.distributed
from allennlp.models.model import Model

# StackedSelfAttentionEncoder
from allennlp.nn import InitializerApplicator, util
from allennlp.modules import Highway, TimeDistributed

from jiant.tasks.tasks import PairClassificationTask, PairRegressionTask
from jiant.utils import utils
from jiant.modules.simple_modules import NullPhraseLayer
from jiant.modules.bilm_encoder import BiLMEncoder
from jiant.modules.onlstm.ON_LSTM import ONLSTMStack
from jiant.modules.prpn.PRPN import PRPN


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

        # General sentence embeddings (for sentence encoder).
        # Make sent_mask first, transformers text_field_embedder will change the token index
        sent_mask = util.get_text_field_mask(sent).float()
        # Skip this for probing runs that don't need it.
        if not isinstance(self._phrase_layer, NullPhraseLayer):
            word_embs_in_context = self._highway_layer(self._text_field_embedder(sent))
        else:
            word_embs_in_context = None

        # Task-specific sentence embeddings (e.g. custom ELMo weights).
        # Skip computing this if it won't be used.
        if self.sep_embs_for_skip:
            task_word_embs_in_context = self._highway_layer(
                self._text_field_embedder(sent, task._classifier_name)
            )
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
                "skip_vec is none - perhaps embeddings are not configured properly?",
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
