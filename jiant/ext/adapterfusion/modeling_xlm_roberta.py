from .configuration_xlm_roberta import XLMRobertaConfig
from .modeling_roberta import RobertaModel


class XLMRobertaModel(RobertaModel):
    config_class = XLMRobertaConfig
