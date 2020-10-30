from dataclasses import dataclass
from enum import Enum

import transformers

from jiant.tasks.core import FeaturizationSpec


class ModelArchitectures(Enum):
    BERT = 1
    XLM = 2
    ROBERTA = 3
    ALBERT = 4
    XLM_ROBERTA = 5
    BART = 6
    MBART = 7
    ELECTRA = 8

    @classmethod
    def from_model_type(cls, model_type: str):
        """Get the model architecture for the provided shortcut name.

        Args:
            model_type (str): model shortcut name.

        Returns:
            Model architecture associated with the provided shortcut name.

        """
        if model_type.startswith("bert-"):
            return cls.BERT
        elif model_type.startswith("xlm-") and not model_type.startswith("xlm-roberta"):
            return cls.XLM
        elif model_type.startswith("roberta-"):
            return cls.ROBERTA
        elif model_type.startswith("albert-"):
            return cls.ALBERT
        elif model_type == "glove_lstm":
            return cls.GLOVE_LSTM
        elif model_type.startswith("xlm-roberta-"):
            return cls.XLM_ROBERTA
        elif model_type.startswith("bart-"):
            return cls.BART
        elif model_type.startswith("mbart-"):
            return cls.MBART
        elif model_type.startswith("electra-"):
            return cls.ELECTRA
        else:
            raise KeyError(model_type)

    @classmethod
    def from_transformers_model(cls, transformers_model):
        if isinstance(
            transformers_model, transformers.BertPreTrainedModel
        ) and transformers_model.__class__.__name__.startswith("Bert"):
            return cls.BERT
        elif isinstance(transformers_model, transformers.XLMPreTrainedModel):
            return cls.XLM
        elif isinstance(
            transformers_model, transformers.BertPreTrainedModel
        ) and transformers_model.__class__.__name__.startswith("Robert"):
            return cls.ROBERTA
        elif isinstance(
            transformers_model, transformers.BertPreTrainedModel
        ) and transformers_model.__class__.__name__.startswith("XLMRoberta"):
            return cls.XLM_ROBERTA
        elif isinstance(transformers_model, transformers.modeling_albert.AlbertPreTrainedModel):
            return cls.ALBERT
        elif isinstance(transformers_model, transformers.modeling_bart.PretrainedBartModel):
            return bart_or_mbart_model_heuristic(model_config=transformers_model.config)
        elif isinstance(transformers_model, transformers.modeling_electra.ElectraPreTrainedModel):
            return cls.ELECTRA
        else:
            raise KeyError(str(transformers_model))

    @classmethod
    def from_tokenizer_class(cls, tokenizer_class):
        if isinstance(tokenizer_class, transformers.BertTokenizer):
            return cls.BERT
        elif isinstance(tokenizer_class, transformers.XLMTokenizer):
            return cls.XLM
        elif isinstance(tokenizer_class, transformers.RobertaTokenizer):
            return cls.ROBERTA
        elif isinstance(tokenizer_class, transformers.XLMRobertaTokenizer):
            return cls.XLM_ROBERTA
        elif isinstance(tokenizer_class, transformers.AlbertTokenizer):
            return cls.ALBERT
        elif isinstance(tokenizer_class, transformers.BartTokenizer):
            return cls.BART
        elif isinstance(tokenizer_class, transformers.MBartTokenizer):
            return cls.MBART
        elif isinstance(tokenizer_class, transformers.ElectraTokenizer):
            return cls.ELECTRA
        else:
            raise KeyError(str(tokenizer_class))

    @classmethod
    def is_transformers_model_arch(cls, model_arch):
        return model_arch in [
            cls.BERT,
            cls.XLM,
            cls.ROBERTA,
            cls.ALBERT,
            cls.XLM_ROBERTA,
            cls.BART,
            cls.MBART,
            cls.ELECTRA,
        ]

    @classmethod
    def from_encoder(cls, encoder):
        if (
            isinstance(encoder, transformers.BertModel)
            and encoder.__class__.__name__ == "BertModel"
        ):
            return cls.BERT
        elif (
            isinstance(encoder, transformers.XLMModel) and encoder.__class__.__name__ == "XLMModel"
        ):
            return cls.XLM
        elif (
            isinstance(encoder, transformers.RobertaModel)
            and encoder.__class__.__name__ == "RobertaModel"
        ):
            return cls.ROBERTA
        elif (
            isinstance(encoder, transformers.AlbertModel)
            and encoder.__class__.__name__ == "AlbertModel"
        ):
            return cls.ALBERT
        elif (
            isinstance(encoder, transformers.XLMRobertaModel)
            and encoder.__class__.__name__ == "XlmRobertaModel"
        ):
            return cls.XLM_ROBERTA
        elif (
            isinstance(encoder, transformers.BartModel)
            and encoder.__class__.__name__ == "BartModel"
        ):
            return bart_or_mbart_model_heuristic(model_config=encoder.config)
        elif (
            isinstance(encoder, transformers.ElectraModel)
            and encoder.__class__.__name__ == "ElectraModel"
        ):
            return cls.ELECTRA
        else:
            raise KeyError(type(encoder))


@dataclass
class ModelClassSpec:
    config_class: type
    tokenizer_class: type
    model_class: type


def build_featurization_spec(model_type, max_seq_length):
    model_arch = ModelArchitectures.from_model_type(model_type)
    if model_arch == ModelArchitectures.BERT:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
            sep_token_extra=False,
        )
    elif model_arch == ModelArchitectures.XLM:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # RoBERTa has no token_type_ids
            sep_token_extra=False,
        )
    elif model_arch == ModelArchitectures.ROBERTA:
        # RoBERTa is weird
        # token 0 = '<s>' which is the cls_token
        # token 1 = '</s>' which is the sep_token
        # Also two '</s>'s are used between sentences. Yes, not '</s><s>'.
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=1,  # Roberta uses pad_token_id = 1
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # RoBERTa has no token_type_ids
            sep_token_extra=True,
        )
    elif model_arch == ModelArchitectures.ALBERT:
        #
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,  # ?
            pad_on_left=False,  # ok
            cls_token_segment_id=0,  # ok
            pad_token_segment_id=0,  # ok
            pad_token_id=0,  # I think?
            pad_token_mask_id=0,  # I think?
            sequence_a_segment_id=0,  # I think?
            sequence_b_segment_id=1,  # I think?
            sep_token_extra=False,
        )
    elif model_arch == ModelArchitectures.XLM_ROBERTA:
        # XLM-RoBERTa is weird
        # token 0 = '<s>' which is the cls_token
        # token 1 = '</s>' which is the sep_token
        # Also two '</s>'s are used between sentences. Yes, not '</s><s>'.
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=1,  # XLM-RoBERTa uses pad_token_id = 1
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # XLM-RoBERTa has no token_type_ids
            sep_token_extra=True,
        )
    elif model_arch == ModelArchitectures.BART:
        # BART is weird
        # token 0 = '<s>' which is the cls_token
        # token 1 = '</s>' which is the sep_token
        # Also two '</s>'s are used between sentences. Yes, not '</s><s>'.
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=1,  # BART uses pad_token_id = 1
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # BART has no token_type_ids
            sep_token_extra=True,
        )
    elif model_arch == ModelArchitectures.MBART:
        # mBART is weird
        # token 0 = '<s>' which is the cls_token
        # token 1 = '</s>' which is the sep_token
        # Also two '</s>'s are used between sentences. Yes, not '</s><s>'.
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=1,  # mBART uses pad_token_id = 1
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=0,  # mBART has no token_type_ids
            sep_token_extra=True,
        )
    elif model_arch == ModelArchitectures.ELECTRA:
        return FeaturizationSpec(
            max_seq_length=max_seq_length,
            cls_token_at_end=False,
            pad_on_left=False,
            cls_token_segment_id=0,
            pad_token_segment_id=0,
            pad_token_id=0,
            pad_token_mask_id=0,
            sequence_a_segment_id=0,
            sequence_b_segment_id=1,
            sep_token_extra=False,
        )
    else:
        raise KeyError(model_arch)


TOKENIZER_CLASS_DICT = {
    ModelArchitectures.BERT: transformers.BertTokenizer,
    ModelArchitectures.XLM: transformers.XLMTokenizer,
    ModelArchitectures.ROBERTA: transformers.RobertaTokenizer,
    ModelArchitectures.XLM_ROBERTA: transformers.XLMRobertaTokenizer,
    ModelArchitectures.ALBERT: transformers.AlbertTokenizer,
    ModelArchitectures.BART: transformers.BartTokenizer,
    ModelArchitectures.MBART: transformers.MBartTokenizer,
    ModelArchitectures.ELECTRA: transformers.ElectraTokenizer,
}


def resolve_tokenizer_class(model_type):
    """Get tokenizer class for a given model architecture.

    Args:
        model_type (str): model shortcut name.

    Returns:
        Tokenizer associated with the given model.

    """
    return TOKENIZER_CLASS_DICT[ModelArchitectures.from_model_type(model_type)]


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
