import abc

from dataclasses import dataclass

from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import torch
import torch.nn as nn

import jiant.utils.python.strings as strings
from jiant.tasks.core import BatchMixin
from jiant.tasks.core import FeaturizationSpec
from jiant.tasks.core import Task

from jiant.proj.main.components.outputs import construct_output_from_dict
from jiant.proj.main.modeling.taskmodels import Taskmodel
from jiant.shared.model_resolution import ModelArchitectures

from jiant.utils.tokenization_utils import bow_tag_tokens
from jiant.utils.tokenization_utils import eow_tag_tokens
from jiant.utils.tokenization_utils import process_bytebpe_tokens
from jiant.utils.tokenization_utils import process_wordpiece_tokens
from jiant.utils.tokenization_utils import process_sentencepiece_tokens


@dataclass
class JiantModelOutput:
    pooled: torch.Tensor
    unpooled: torch.Tensor
    other: Any = None


class JiantModel(nn.Module):
    def __init__(
        self,
        task_dict: Dict[str, Task],
        encoder: nn.Module,
        taskmodels_dict: Dict[str, Taskmodel],
        task_to_taskmodel_map: Dict[str, str],
        tokenizer,
    ):
        super().__init__()
        self.task_dict = task_dict
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
        self.task_to_taskmodel_map = task_to_taskmodel_map
        self.tokenizer = tokenizer

    def forward(self, batch: BatchMixin, task: Task, compute_loss: bool = False):
        """Calls to this forward method are delegated to the forward of the appropriate taskmodel.

        When JiantModel forward is called, the task name from the task argument is used as a key
        to select the appropriate submodule/taskmodel, and that taskmodel's forward is called.

        Args:
            batch (BatchMixin): model input.
            task (Task): task to which to delegate the forward call.
            compute_loss (bool): whether to calculate and return the loss.

        Returns:
            Dict containing the model output, optionally including the loss.

        """
        if isinstance(batch, dict):
            batch = task.Batch.from_dict(batch)
        if isinstance(task, str):
            task_name = task
            task = self.task_dict[task]
        else:
            task_name = task.name
            task = task
        taskmodel_key = self.task_to_taskmodel_map[task_name]
        taskmodel = self.taskmodels_dict[taskmodel_key]
        return taskmodel(
            batch=batch, tokenizer=self.tokenizer, compute_loss=compute_loss,
        ).to_dict()


def wrap_jiant_forward(
    jiant_model: Union[JiantModel, nn.DataParallel],
    batch: BatchMixin,
    task: Task,
    compute_loss: bool = False,
):
    """Wrapper to repackage model inputs using dictionaries for compatibility with DataParallel.

    Wrapper that converts batches (type BatchMixin) to dictionaries before delegating to
    JiantModel's forward method, and then converts the resulting model output dict into the
    appropriate model output dataclass.

    Args:
        jiant_model (Union[JiantModel, nn.DataParallel]):
        batch (BatchMixin): model input batch.
        task (Task): Task object passed for access in the taskmodel.
        compute_loss (bool): True if loss should be computed, False otherwise.

    Returns:
        Union[LogitsOutput, LogitsAndLossOutput, EmbeddingOutput]: model output dataclass.

    """
    assert isinstance(jiant_model, (JiantModel, nn.DataParallel))
    is_multi_gpu = isinstance(jiant_model, nn.DataParallel)
    model_output = construct_output_from_dict(
        jiant_model(
            batch=batch.to_dict() if is_multi_gpu else batch, task=task, compute_loss=compute_loss,
        )
    )
    if is_multi_gpu and compute_loss:
        model_output.loss = model_output.loss.mean()
    return model_output


class JiantTransformersModelFactory:
    """This factory is used to create JiantTransformersModels based on Huggingface's models.
    A wrapper class around Huggingface's Transformer models is used to abstract any inconsistencies
    in the classes.

    Attributes:
        registry (dict): Dynamic registry mapping ModelArchitectures to JiantTransformersModels
    """

    registry = {}

    @classmethod
    def get_registry(cls):
        return cls.registry

    @classmethod
    def build_featurization_spec(cls, model_type, max_seq_length):
        model_arch = ModelArchitectures.from_model_type(model_type)
        model_class = cls.get_registry()[model_arch]
        return model_class.get_feat_spec(model_type, max_seq_length)

    @classmethod
    def register(cls, model_arch: ModelArchitectures) -> Callable:
        """Register model_arch as a key mapping to a TaskModel

        Args:
            model_arch (ModelArchitectures): ModelArchitecture key mapping to a
                                             JiantTransformersModel

        Returns:
            Callable: inner_wrapper() wrapping TaskModel constructor
        """

        def inner_wrapper(wrapped_class: JiantTransformersModel) -> Callable:
            assert model_arch not in cls.registry
            cls.registry[model_arch] = wrapped_class
            return wrapped_class

        return inner_wrapper

    def __call__(cls, hf_model):
        """Returns the JiantTransformersModel wrapper class for the corresponding Hugging Face
        Transformer model.

        Args:
            hf_model (PreTrainedModel): Hugging Face model to convert to JiantTransformersModel

        Returns:
            JiantTransformersModel: Jiant wrapper class for Hugging Face model
        """
        encoder_class = cls.registry[ModelArchitectures(hf_model.config.model_type)]
        encoder = encoder_class(hf_model)
        return encoder


class JiantTransformersModel(metaclass=abc.ABCMeta):
    def __init__(self, baseObject):
        self.__class__ = type(
            baseObject.__class__.__name__, (self.__class__, baseObject.__class__), {}
        )
        self.__dict__ = baseObject.__dict__

    @classmethod
    @abc.abstractmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        pass

    @abc.abstractmethod
    def get_mlm_weights_dict(self, weights_dict):
        pass

    @abc.abstractmethod
    def get_feat_spec(self, weights_dict):
        pass

    def get_hidden_size(self):
        return self.config.hidden_size

    def get_hidden_dropout_prob(self):
        return self.config.hidden_dropout_prob

    def encode(self, input_ids, segment_ids, input_mask, output_hidden_states=True):
        output = self.forward(
            input_ids=input_ids,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            output_hidden_states=output_hidden_states,
        )
        return JiantModelOutput(
            pooled=output.pooler_output,
            unpooled=output.last_hidden_state,
            other=output.hidden_states,
        )


@JiantTransformersModelFactory.register(ModelArchitectures.BERT)
class JiantBertModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """See tokenization_normalization.py for details"""
        if tokenizer.init_kwargs.get("do_lower_case", False):
            space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = process_wordpiece_tokens(target_tokenization)

        return modifed_space_tokenization, modifed_target_tokenization

    def get_feat_spec(self, max_seq_length):
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

    def get_mlm_weights_dict(self, weights_dict):
        mlm_weights_map = {
            "bias": "cls.predictions.bias",
            "dense.weight": "cls.predictions.transform.dense.weight",
            "dense.bias": "cls.predictions.transform.dense.bias",
            "LayerNorm.weight": "cls.predictions.transform.LayerNorm.weight",
            "LayerNorm.bias": "cls.predictions.transform.LayerNorm.bias",
            "decoder.weight": "cls.predictions.decoder.weight",
            "decoder.bias": "cls.predictions.bias",  # <-- linked directly to bias
        }
        mlm_weights_dict = {new_k: weights_dict[old_k] for new_k, old_k in mlm_weights_map.items()}
        return mlm_weights_dict


@JiantTransformersModelFactory.register(ModelArchitectures.ROBERTA)
class JiantRobertaModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """See tokenization_normalization.py for details"""
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = ["Ġ" + target_tokenization[0]] + target_tokenization[1:]
        modifed_target_tokenization = process_bytebpe_tokens(modifed_target_tokenization)

        return modifed_space_tokenization, modifed_target_tokenization

    def get_mlm_weights_dict(self, weights_dict):
        mlm_weights_dict = {
            strings.remove_prefix(k, "lm_head."): v for k, v in weights_dict.items()
        }
        mlm_weights_dict["decoder.bias"] = mlm_weights_dict["bias"]
        return mlm_weights_dict

    def get_feat_spec(self, max_seq_length):
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


@JiantTransformersModelFactory.register(ModelArchitectures.XLM_ROBERTA)
class JiantXLMRobertaModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """See tokenization_normalization.py for details"""
        space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = process_sentencepiece_tokens(target_tokenization)

        return modifed_space_tokenization, modifed_target_tokenization

    def get_feat_spec(self, max_seq_length):
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

    def get_mlm_weights_dict(self, weights_dict):
        mlm_weights_dict = {
            strings.remove_prefix(k, "lm_head."): v for k, v in weights_dict.items()
        }
        mlm_weights_dict["decoder.bias"] = mlm_weights_dict["bias"]
        return mlm_weights_dict


@JiantTransformersModelFactory.register(ModelArchitectures.XLM)
class JiantXLMModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """See tokenization_normalization.py for details"""
        if tokenizer.init_kwargs.get("do_lowercase_and_remove_accent", False):
            space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = eow_tag_tokens(space_tokenization)
        modifed_target_tokenization = target_tokenization

        return modifed_space_tokenization, modifed_target_tokenization

    def get_feat_spec(self, max_seq_length):
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


@JiantTransformersModelFactory.register(ModelArchitectures.ALBERT)
class JiantAlbertModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        """See tokenization_normalization.py for details"""
        space_tokenization = [token.lower() for token in space_tokenization]
        modifed_space_tokenization = bow_tag_tokens(space_tokenization)
        modifed_target_tokenization = process_sentencepiece_tokens(target_tokenization)

        return modifed_space_tokenization, modifed_target_tokenization

    def get_mlm_weights_dict(self, weights_dict):
        mlm_weights_dict = {
            strings.remove_prefix(k, "predictions."): v for k, v in weights_dict.items()
        }
        return mlm_weights_dict

    def get_feat_spec(self, max_seq_length):
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


@JiantTransformersModelFactory.register(ModelArchitectures.ELECTRA)
class JiantElectraModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    def __call__(self, encoder, input_ids, segment_ids, input_mask):
        output = super().__call__(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask
        )
        unpooled = output.hidden_states
        pooled = unpooled[:, 0, :]
        return JiantModelOutput(pooled=pooled, unpooled=unpooled, other=output.hidden_states)

    def get_feat_spec(self, max_seq_length):
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

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        raise NotImplementedError()

    def get_mlm_weights_dict(self, weights_dict):
        raise NotImplementedError()


@JiantTransformersModelFactory.register(ModelArchitectures.BART)
class JiantBartModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    def get_hidden_size(self):
        return self.config.d_model

    def get_hidden_dropout_prob(self):
        return self.config.dropout

    def get_feat_spec(self, max_seq_length):
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

    def __call__(self, encoder, input_ids, input_mask):
        # BART and mBART and encoder-decoder architectures.
        # As described in the BART paper and implemented in Transformers,
        # for single input tasks, the encoder input is the sequence,
        # the decode input is 1-shifted sequence, and the resulting
        # sentence representation is the final decoder state.
        # That's what we use for `unpooled` here.
        dec_last, dec_all, enc_last, enc_all = super().__call__(
            input_ids=input_ids,
            attention_mask=input_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        unpooled = dec_last
        other = (enc_all + dec_all,)

        bsize, slen = input_ids.shape
        batch_idx = torch.arange(bsize).to(input_ids.device)
        # Get last non-pad index
        pooled = unpooled[batch_idx, slen - input_ids.eq(encoder.config.pad_token_id).sum(1) - 1]
        return JiantModelOutput(pooled=pooled, unpooled=unpooled, other=other)

    def get_mlm_weights_dict(self, weights_dict):
        raise NotImplementedError()


@JiantTransformersModelFactory.register(ModelArchitectures.MBART)
class JiantMBartModel(JiantBartModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)

    @classmethod
    def normalize_tokenizations(cls, tokenizer, space_tokenization, target_tokenization):
        raise NotImplementedError()

    def get_feat_spec(self, max_seq_length):
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

    def get_mlm_weights_dict(self, weights_dict):
        raise NotImplementedError()
