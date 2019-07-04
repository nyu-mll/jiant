from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator
from .prpn.PRPN import PRPN


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
