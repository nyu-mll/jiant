# StackedSelfAttentionEncoder
import torch
from allennlp.modules import Seq2SeqEncoder
from allennlp.modules.matrix_attention import DotProductMatrixAttention
from allennlp.nn import InitializerApplicator, util
from allennlp.models.model import Model


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
