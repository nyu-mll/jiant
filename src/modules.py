''' Different model components to use in building the overall model '''
import os
import sys
import json
import logging as log
import ipdb as pdb  # pylint: disable=unused-import
import h5py

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.common.checks import ConfigurationError
from allennlp.models.model import Model
from allennlp.modules import Highway, MatrixAttention
from allennlp.modules import Seq2SeqEncoder, SimilarityFunction, TimeDistributed
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import remove_sentence_boundaries, add_sentence_boundary_token_ids, get_device_of
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.token_indexers.elmo_indexer import ELMoCharacterMapper, ELMoTokenCharactersIndexer
from allennlp.modules.similarity_functions import LinearSimilarity, DotProductSimilarity
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder, CnnEncoder
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder as s2s_e
# StackedSelfAttentionEncoder
from allennlp.modules.feedforward import FeedForward
from allennlp.modules.layer_norm import LayerNorm
from utils import MaskedMultiHeadSelfAttention
from allennlp.nn.activations import Activation
from allennlp.nn.util import add_positional_features

from utils import combine_hidden_states


class SentenceEncoder(Model):
    ''' Given a sequence of tokens, embed each token and pass thru an LSTM '''

    def __init__(self, vocab, text_field_embedder, num_highway_layers, phrase_layer,
                 cove_layer=None, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator()):
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
        d_inp_phrase = phrase_layer.get_input_dim()
        self._cove = cove_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.output_dim = phrase_layer.get_output_dim()

        # if d_emb != d_inp_phrase:
        if (cove_layer is None and d_emb != d_inp_phrase) \
                or (cove_layer is not None and d_emb + 600 != d_inp_phrase):
            raise ConfigurationError("The output dimension of the text_field_embedder "
                                     "must match the input dimension of "
                                     "the phrase_encoder. Found {} and {} respectively."
                                     .format(d_emb, d_inp_phrase))
        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, sent):
        # pylint: disable=arguments-differ
        """
        Args:
            - sent (Dict[str, torch.LongTensor]): From a ``TextField``.

        Returns:
            - sent_enc (torch.FloatTensor): (b_size, seq_len, d_emb)
        """
        sent_embs = self._highway_layer(self._text_field_embedder(sent))
        if self._cove is not None:
            sent_lens = torch.ne(sent['words'], self.pad_idx).long().sum(dim=-1).data
            sent_cove_embs = self._cove(sent['words'], sent_lens)
            sent_embs = torch.cat([sent_embs, sent_cove_embs], dim=-1)
        sent_embs = self._dropout(sent_embs)

        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None

        sent_enc = self._phrase_layer(sent_embs, sent_lstm_mask)
        sent_enc = self._dropout(sent_enc)

        sent_mask = sent_mask.unsqueeze(dim=-1)
        sent_enc.data.masked_fill_(1 - sent_mask.byte().data, -float('inf'))
        return sent_enc, sent_mask


class BiLMEncoder(SentenceEncoder):
    ''' Given a sequence of tokens, embed each token and pass thru an LSTM
    A simple wrap up for bidirectional LM training
    '''

    def __init__(self, vocab, text_field_embedder, num_highway_layers,
                 phrase_layer, bwd_phrase_layer,
                 cove_layer=None, dropout=0.2, mask_lstms=True,
                 initializer=InitializerApplicator()):
        super(
            BiLMEncoder,
            self).__init__(
            vocab,
            text_field_embedder,
            num_highway_layers,
            phrase_layer,
            cove_layer,
            dropout,
            mask_lstms,
            initializer)
        self._bwd_phrase_layer = bwd_phrase_layer
        self.output_dim += self._bwd_phrase_layer.get_output_dim()
        initializer(self)

    def _uni_directional_forward(self, sent, go_forward=True):
        sent_embs = self._highway_layer(self._text_field_embedder(sent))
        if self._cove is not None:
            sent_lens = torch.ne(sent['words'], self.pad_idx).long().sum(dim=-1).data
            sent_cove_embs = self._cove(sent['words'], sent_lens)
            sent_embs = torch.cat([sent_embs, sent_cove_embs], dim=-1)
        sent_embs = self._dropout(sent_embs)

        sent_mask = util.get_text_field_mask(sent).float()
        sent_lstm_mask = sent_mask if self._mask_lstms else None

        if go_forward:
            sent_enc = self._phrase_layer(sent_embs, sent_lstm_mask)
        else:
            sent_enc = self._bwd_phrase_layer(sent_embs, sent_lstm_mask)

        sent_mask = sent_mask.unsqueeze(dim=-1)

        return sent_enc, sent_mask

    def forward(self, sent, bwd_sent=None):
        # pylint: disable=arguments-differ
        """
        Args:
            - sent (Dict[str, torch.LongTensor]): From a ``TextField``.

        Returns:
            - sent_enc (torch.FloatTensor): (b_size, seq_len, d_emb)
        """
        fwd_sent_enc, fwd_sent_mask = self._uni_directional_forward(sent)
        if bwd_sent is None:
            bwd_sent_enc, bwd_sent_mask = self._uni_directional_forward(sent, False)
        else:
            bwd_sent_enc, bwd_sent_mask = self._uni_directional_forward(bwd_sent, False)
        sent_enc = torch.cat([fwd_sent_enc, bwd_sent_enc], dim=-1)
        sent_enc = self._dropout(sent_enc)
        return sent_enc, fwd_sent_mask


class BoWSentEncoder(Model):
    def __init__(self, vocab, text_field_embedder, initializer=InitializerApplicator()):
        super(BoWSentEncoder, self).__init__(vocab)

        self._text_field_embedder = text_field_embedder
        self.output_dim = text_field_embedder.get_output_dim()
        initializer(self)

    def forward(self, sent):
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        question : Dict[str, torch.LongTensor]
            From a ``TextField``.
        passage : Dict[str, torch.LongTensor]
            From a ``TextField``.  The model assumes that this passage contains the answer to the
            question, and predicts the beginning and ending positions of the answer within the
            passage.

        Returns
        -------
        pair_rep : torch.FloatTensor?
            Tensor representing the final output of the BiDAF model
            to be plugged into the next module

        """
        word_embs = self._text_field_embedder(sent)
        word_mask = util.get_text_field_mask(sent).float()
        return word_embs, word_mask  # need to get # nonzero elts


class SimplePairEncoder(Model):
    ''' Given two sentence vectors u and v, model the pair as [u; v; |u-v|; u * v] '''

    def __init__(self, vocab, combine_method='max'):
        super(SimplePairEncoder, self).__init__(vocab)
        self.combine_method = combine_method

    def forward(self, s1, s2, s1_mask, s2_mask):
        """ See above """
        sent_emb1 = combine_hidden_states(s1, s1_mask, self.combine_method)
        sent_emb2 = combine_hidden_states(s2, s2_mask, self.combine_method)
        return torch.cat([sent_emb1, sent_emb2, torch.abs(sent_emb1 - sent_emb2),
                          sent_emb1 * sent_emb2], 1)


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

    def __init__(self, vocab, attention_similarity_function, modeling_layer,
                 combine_method='max',
                 dropout=0.2, mask_lstms=True, initializer=InitializerApplicator()):
        super(AttnPairEncoder, self).__init__(vocab)

        self._matrix_attention = MatrixAttention(attention_similarity_function)
        self._modeling_layer = modeling_layer
        self.pad_idx = vocab.get_token_index(vocab._padding_token)
        self.combine_method = combine_method

        d_out_model = modeling_layer.get_output_dim()
        self.output_dim = d_out_model

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._mask_lstms = mask_lstms

        initializer(self)

    def forward(self, s1, s2, s1_mask, s2_mask):  # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        s1 : Dict[str, torch.LongTensor]
            From a ``TextField``.
        s2 : Dict[str, torch.LongTensor]
            From a ``TextField``.

        Returns
        -------
        pair_rep : torch.FloatTensor?
            Tensor representing the final output of the BiDAF model
            to be plugged into the next module

        """
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
        s2_w_context = self._dropout(s2_w_context)

        # s1 representation, using same attn method as for the s2 representation
        s1_s2_attn = util.last_dim_softmax(similarity_mat.transpose(1, 2).contiguous(), s2_mask)
        # Shape: (batch_size, s1_length, encoding_dim)
        s1_s2_vectors = util.weighted_sum(s2, s1_s2_attn)
        s1_w_context = torch.cat([s1, s1_s2_vectors], 2)
        s1_w_context = self._dropout(s1_w_context)

        modeled_s1 = self._dropout(self._modeling_layer(s1_w_context, s1_mask))
        modeled_s2 = self._dropout(self._modeling_layer(s2_w_context, s2_mask))
        modeled_s1.data.masked_fill_(1 - s1_mask.unsqueeze(dim=-1).byte().data, -float('inf'))
        modeled_s2.data.masked_fill_(1 - s2_mask.unsqueeze(dim=-1).byte().data, -float('inf'))
        #s1_attn = modeled_s1.max(dim=1)[0]
        #s2_attn = modeled_s2.max(dim=1)[0]
        s1_attn = combine_hidden_states(modeled_s1, s1_mask, self.combine_method)
        s2_attn = combine_hidden_states(modeled_s2, s2_mask, self.combine_method)

        return torch.cat([s1_attn, s2_attn, torch.abs(s1_attn - s2_attn),
                          s1_attn * s2_attn], 1)

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
    """
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
