from dataclasses import dataclass
from enum import Enum
from jiant.utils.python.datastructures import BiDict

import transformers


class ModelArchitectures(Enum):
    BERT = "bert"
    XLM = "xlm"
    ROBERTA = "roberta"
    ALBERT = "albert"
    XLM_ROBERTA = "xlm-roberta"
    BART = "bart"
    MBART = "mbart"
    ELECTRA = "electra"
    DEBERTAV2 = "deberta-v2"
    DISTILBERT = "distilbert"

    @classmethod
    def from_model_type(cls, model_type: str):
        return cls(model_type)

    def get_encoder_prefix(self):
        if self.value == "xlm-roberta":
            return "roberta"
        else:
            return self.value


TOKENIZER_CLASS_DICT = BiDict(
    {
        ModelArchitectures.BERT: transformers.BertTokenizer,
        ModelArchitectures.XLM: transformers.XLMTokenizer,
        ModelArchitectures.ROBERTA: transformers.RobertaTokenizer,
        ModelArchitectures.XLM_ROBERTA: transformers.XLMRobertaTokenizer,
        ModelArchitectures.ALBERT: transformers.AlbertTokenizer,
        ModelArchitectures.BART: transformers.BartTokenizer,
        ModelArchitectures.MBART: transformers.MBartTokenizer,
        ModelArchitectures.ELECTRA: transformers.ElectraTokenizer,
        ModelArchitectures.DEBERTAV2: transformers.DebertaV2Tokenizer,
        ModelArchitectures.DISTILBERT: transformers.DistilBertTokenizer,
    }
)


@dataclass
class ModelClassSpec:
    config_class: type
    tokenizer_class: type
    model_class: type


def resolve_tokenizer_class(model_type):
    """Get tokenizer class for a given model architecture.

    Args:
        model_type (str): model shortcut name.

    Returns:
        Tokenizer associated with the given model.

    """
    return TOKENIZER_CLASS_DICT[ModelArchitectures(model_type)]


def resolve_model_arch_tokenizer(tokenizer):
    """Get the model architecture for a given tokenizer.

    Args:
        tokenizer

    Returns:
        ModelArchitecture

    """
    assert len(TOKENIZER_CLASS_DICT.inverse[tokenizer.__class__]) == 1
    return TOKENIZER_CLASS_DICT.inverse[tokenizer.__class__][0]


def resolve_is_lower_case(tokenizer):
    if isinstance(tokenizer, transformers.BertTokenizer):
        return tokenizer.basic_tokenizer.do_lower_case
    if isinstance(tokenizer, transformers.AlbertTokenizer):
        return tokenizer.do_lower_case
    else:
        return False


def bart_or_mbart_model_heuristic(model_config: transformers.BartConfig) -> ModelArchitectures:
    if model_config.is_valid_mbart():
        return ModelArchitectures.MBART
    else:
        return ModelArchitectures.BART
