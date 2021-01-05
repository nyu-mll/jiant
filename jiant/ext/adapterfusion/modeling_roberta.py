from transformers.modeling_roberta import RobertaEmbeddings
from .modeling_bert import BertModel
from .configuration_roberta import RobertaConfig


class RobertaModel(BertModel):
    """
    This class overrides :class:`~transformers.BertModel`. Please check the
    superclass for the appropriate documentation alongside usage examples.
    """

    config_class = RobertaConfig
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
