from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, util


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
