from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from .onlstm.ON_LSTM import ONLSTMStack


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
