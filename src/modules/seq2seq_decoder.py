# This is a slightly modified version of the AllenNLP SimpleSeq2Seq class:
# https://github.com/allenai/allennlp/blob/master/allennlp/models/encoder_decoders/simple_seq2seq.py  # noqa

import logging as log
from typing import Dict

import numpy
import torch
import torch.nn.functional as F
from allennlp.common import Params
from allennlp.common.util import END_SYMBOL, START_SYMBOL
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.attention import BilinearAttention
from allennlp.modules.similarity_functions import SimilarityFunction
from allennlp.modules.token_embedders import Embedding
from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits, weighted_sum
from overrides import overrides
from torch.nn.modules.linear import Linear
from torch.nn.modules.rnn import LSTMCell

from .modules import Pooler


class Seq2SeqDecoder(Model):
    """
    This is a slightly modified version of AllenNLP SimpleSeq2Seq class
    """

    def __init__(
        self,
        vocab: Vocabulary,
        input_dim: int,
        decoder_hidden_size: int,
        max_decoding_steps: int,
        output_proj_input_dim: int,
        target_namespace: str = "targets",
        target_embedding_dim: int = None,
        attention: str = "none",
        dropout: float = 0.0,
        scheduled_sampling_ratio: float = 0.0,
    ) -> None:
        super(Seq2SeqDecoder, self).__init__(vocab)

        # deprecated module
        log.warning(
            "DeprecationWarning: modules.Seq2SeqDecoder is deprecated and is no longer maintained"
        )

        self._max_decoding_steps = max_decoding_steps
        self._target_namespace = target_namespace

        # We need the start symbol to provide as the input at the first timestep of decoding, and
        # end symbol as a way to indicate the end of the decoded sequence.
        self._start_index = self.vocab.get_token_index(START_SYMBOL, self._target_namespace)
        self._end_index = self.vocab.get_token_index(END_SYMBOL, self._target_namespace)
        self._unk_index = self.vocab.get_token_index("@@UNKNOWN@@", self._target_namespace)
        num_classes = self.vocab.get_vocab_size(self._target_namespace)

        # Decoder output dim needs to be the same as the encoder output dim since we initialize the
        # hidden state of the decoder with that of the final hidden states of the encoder. Also, if
        # we're using attention with ``DotProductSimilarity``, this is needed.
        self._encoder_output_dim = input_dim
        self._decoder_hidden_dim = decoder_hidden_size
        if self._encoder_output_dim != self._decoder_hidden_dim:
            self._projection_encoder_out = Linear(
                self._encoder_output_dim, self._decoder_hidden_dim
            )
        else:
            self._projection_encoder_out = lambda x: x
        self._decoder_output_dim = self._decoder_hidden_dim
        self._output_proj_input_dim = output_proj_input_dim
        self._target_embedding_dim = target_embedding_dim
        self._target_embedder = Embedding(num_classes, self._target_embedding_dim)

        # Used to get an initial hidden state from the encoder states
        self._sent_pooler = Pooler(project=True, d_inp=input_dim, d_proj=decoder_hidden_size)

        if attention == "bilinear":
            self._decoder_attention = BilinearAttention(decoder_hidden_size, input_dim)
            # The output of attention, a weighted average over encoder outputs, will be
            # concatenated to the input vector of the decoder at each time
            # step.
            self._decoder_input_dim = input_dim + target_embedding_dim
        elif attention == "none":
            self._decoder_attention = None
            self._decoder_input_dim = target_embedding_dim
        else:
            raise Exception("attention not implemented {}".format(attention))

        self._decoder_cell = LSTMCell(self._decoder_input_dim, self._decoder_hidden_dim)
        # Allow for a bottleneck layer between encoder outputs and distribution over vocab
        # The bottleneck layer consists of a linear transform and helps to reduce
        # number of parameters
        if self._output_proj_input_dim != self._decoder_output_dim:
            self._projection_bottleneck = Linear(
                self._decoder_output_dim, self._output_proj_input_dim
            )
        else:
            self._projection_bottleneck = lambda x: x
        self._output_projection_layer = Linear(self._output_proj_input_dim, num_classes)
        self._dropout = torch.nn.Dropout(p=dropout)

    def _initalize_hidden_context_states(self, encoder_outputs, encoder_outputs_mask):
        """
        Initialization of the decoder state, based on the encoder output.
        Parameters
        ----------
        encoder_outputs: torch.FloatTensor, [bs, T, h]
        encoder_outputs_mask: torch.LongTensor, [bs, T, 1]
        """

        if self._decoder_attention is not None:
            encoder_outputs = self._projection_encoder_out(encoder_outputs)
            encoder_outputs.data.masked_fill_(1 - encoder_outputs_mask.byte().data, -float("inf"))

            decoder_hidden = encoder_outputs.new_zeros(
                encoder_outputs_mask.size(0), self._decoder_hidden_dim
            )
            decoder_context = encoder_outputs.max(dim=1)[0]
        else:
            decoder_hidden = self._sent_pooler(encoder_outputs, encoder_outputs_mask)
            decoder_context = encoder_outputs.new_zeros(
                encoder_outputs_mask.size(0), self._decoder_hidden_dim
            )

        return decoder_hidden, decoder_context

    @overrides
    def forward(
        self,  # type: ignore
        encoder_outputs,  # type: ignore
        encoder_outputs_mask,  # type: ignore
        target_tokens: Dict[str, torch.LongTensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Decoder logic for producing the entire target sequence at train time.

        Parameters
        ----------
        encoder_outputs : torch.FloatTensor, [bs, T, h]
        encoder_outputs_mask : torch.LongTensor, [bs, T, 1]
        target_tokens : Dict[str, torch.LongTensor]
        """
        # TODO: target_tokens is not optional.
        batch_size, _, _ = encoder_outputs.size()

        if target_tokens is not None:
            targets = target_tokens["words"]
            target_sequence_length = targets.size()[1]
            num_decoding_steps = target_sequence_length - 1
        else:
            num_decoding_steps = self._max_decoding_steps

        decoder_hidden, decoder_context = self._initalize_hidden_context_states(
            encoder_outputs, encoder_outputs_mask
        )

        step_logits = []

        for timestep in range(num_decoding_steps):
            input_choices = targets[:, timestep]
            decoder_input = self._prepare_decode_step_input(
                input_choices, decoder_hidden, encoder_outputs, encoder_outputs_mask
            )
            decoder_hidden, decoder_context = self._decoder_cell(
                decoder_input, (decoder_hidden, decoder_context)
            )

            # output projection
            proj_input = self._projection_bottleneck(decoder_hidden)
            # (batch_size, num_classes)
            output_projections = self._output_projection_layer(proj_input)

            # list of (batch_size, 1, num_classes)
            step_logit = output_projections.unsqueeze(1)
            step_logits.append(step_logit)

        # (batch_size, num_decoding_steps, num_classes)
        logits = torch.cat(step_logits, 1)

        output_dict = {"logits": logits}

        if target_tokens:
            target_mask = get_text_field_mask(target_tokens)
            loss = self._get_loss(logits, targets, target_mask)
            output_dict["loss"] = loss

        return output_dict

    def _prepare_decode_step_input(
        self,
        input_indices: torch.LongTensor,
        decoder_hidden_state: torch.LongTensor = None,
        encoder_outputs: torch.LongTensor = None,
        encoder_outputs_mask: torch.LongTensor = None,
    ) -> torch.LongTensor:
        """
        Given the input indices for the current timestep of the decoder, and all the encoder
        outputs, compute the input at the current timestep.  Note: This method is agnostic to
        whether the indices are gold indices or the predictions made by the decoder at the last
        timestep.

        If we're not using attention, the output of this method is just an embedding of the input
        indices.  If we are, the output will be a concatentation of the embedding and an attended
        average of the encoder inputs.

        Parameters
        ----------
        input_indices : torch.LongTensor
            Indices of either the gold inputs to the decoder or the predicted labels from the
            previous timestep.
        decoder_hidden_state : torch.LongTensor, optional (not needed if no attention)
            Output of from the decoder at the last time step. Needed only if using attention.
        encoder_outputs : torch.LongTensor, optional (not needed if no attention)
            Encoder outputs from all time steps. Needed only if using attention.
        encoder_outputs_mask : torch.LongTensor, optional (not needed if no attention)
            Masks on encoder outputs. Needed only if using attention.
        """
        input_indices = input_indices.long()
        # input_indices : (batch_size,)  since we are processing these one timestep at a time.
        # (batch_size, target_embedding_dim)
        embedded_input = self._target_embedder(input_indices)

        if self._decoder_attention is not None:
            # encoder_outputs : (batch_size, input_sequence_length, encoder_output_dim)
            # Ensuring mask is also a FloatTensor. Or else the multiplication within attention will
            # complain.

            # important - need to use zero-masking instead of -inf for attention
            # I've checked that doing this doesn't significantly increase time
            # per batch, but should consider only doing once
            encoder_outputs.data.masked_fill_(1 - encoder_outputs_mask.byte().data, 0.0)

            encoder_outputs = 0.5 * encoder_outputs
            encoder_outputs_mask = encoder_outputs_mask.float()
            encoder_outputs_mask = encoder_outputs_mask[:, :, 0]
            # (batch_size, input_sequence_length)
            input_weights = self._decoder_attention(
                decoder_hidden_state, encoder_outputs, encoder_outputs_mask
            )
            # (batch_size, input_dim)
            attended_input = weighted_sum(encoder_outputs, input_weights)
            # (batch_size, input_dim + target_embedding_dim)
            return torch.cat((attended_input, embedded_input), -1)
        else:
            return embedded_input

    @staticmethod
    def _get_loss(
        logits: torch.LongTensor, targets: torch.LongTensor, target_mask: torch.LongTensor
    ) -> torch.LongTensor:
        """
        Takes logits (unnormalized outputs from the decoder) of size (batch_size,
        num_decoding_steps, num_classes), target indices of size (batch_size, num_decoding_steps+1)
        and corresponding masks of size (batch_size, num_decoding_steps+1) steps and computes cross
        entropy loss while taking the mask into account.

        The length of ``targets`` is expected to be greater than that of ``logits`` because the
        decoder does not need to compute the output corresponding to the last timestep of
        ``targets``. This method aligns the inputs appropriately to compute the loss.

        During training, we want the logit corresponding to timestep i to be similar to the target
        token from timestep i + 1. That is, the targets should be shifted by one timestep for
        appropriate comparison.  Consider a single example where the target has 3 words, and
        padding is to 7 tokens.
           The complete sequence would correspond to <S> w1  w2  w3  <E> <P> <P>
           and the mask would be                     1   1   1   1   1   0   0
           and let the logits be                     l1  l2  l3  l4  l5  l6
        We actually need to compare:
           the sequence           w1  w2  w3  <E> <P> <P>
           with masks             1   1   1   1   0   0
           against                l1  l2  l3  l4  l5  l6
           (where the input was)  <S> w1  w2  w3  <E> <P>
        """
        relevant_targets = targets[:, 1:].contiguous()  # (batch_size, num_decoding_steps)
        # (batch_size, num_decoding_steps)
        relevant_mask = target_mask[:, 1:].contiguous()
        loss = sequence_cross_entropy_with_logits(logits, relevant_targets, relevant_mask)
        return loss
