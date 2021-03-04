import abc

from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import torch
import torch.nn as nn
import transformers

import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.tasks as tasks

from jiant.proj.main.components.outputs import construct_output_from_dict
from jiant.shared.model_resolution import ModelArchitectures


@dataclass
class JiantModelOutput:
    pooled: torch.Tensor
    unpooled: torch.Tensor
    other: Any = None


class JiantModel(nn.Module):
    def __init__(
        self,
        task_dict: Dict[str, tasks.Task],
        encoder: nn.Module,
        taskmodels_dict: Dict[str, taskmodels.Taskmodel],
        task_to_taskmodel_map: Dict[str, str],
        tokenizer,
    ):
        super().__init__()
        self.task_dict = task_dict
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
        self.task_to_taskmodel_map = task_to_taskmodel_map
        self.tokenizer = tokenizer

    def forward(self, batch: tasks.BatchMixin, task: tasks.Task, compute_loss: bool = False):
        """Calls to this forward method are delegated to the forward of the appropriate taskmodel.

        When JiantModel forward is called, the task name from the task argument is used as a key
        to select the appropriate submodule/taskmodel, and that taskmodel's forward is called.

        Args:
            batch (tasks.BatchMixin): model input.
            task (tasks.Task): task to which to delegate the forward call.
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
    batch: tasks.BatchMixin,
    task: tasks.Task,
    compute_loss: bool = False,
):
    """Wrapper to repackage model inputs using dictionaries for compatibility with DataParallel.

    Wrapper that converts batches (type tasks.BatchMixin) to dictionaries before delegating to
    JiantModel's forward method, and then converts the resulting model output dict into the
    appropriate model output dataclass.

    Args:
        jiant_model (Union[JiantModel, nn.DataParallel]):
        batch (tasks.BatchMixin): model input batch.
        task (tasks.Task): Task object passed for access in the taskmodel.
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
        jiant_transformers_model_class = cls.registry[
            ModelArchitectures(hf_model.config.model_type)
        ]
        jiant_transformers_model = jiant_transformers_model_class(hf_model)
        return jiant_transformers_model


class JiantTransformersModel(metaclass=abc.ABCMeta):
    def __init__(self, baseObject):
        self.__class__ = type(
            baseObject.__class__.__name__, (self.__class__, baseObject.__class__), {}
        )
        self.__dict__ = baseObject.__dict__

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
        self.hf_pretrained_encoder_with_pretrained_head = transformers.BertForPreTraining


@JiantTransformersModelFactory.register(ModelArchitectures.ROBERTA)
class JiantRobertaModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
        self.hf_pretrained_encoder_with_pretrained_head = transformers.RobertaForMaskedLM


@JiantTransformersModelFactory.register(ModelArchitectures.ALBERT)
class JiantAlbertModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
        self.hf_pretrained_encoder_with_pretrained_head = transformers.AlbertForMaskedLM


@JiantTransformersModelFactory.register(ModelArchitectures.XLM_ROBERTA)
class JiantXLMRobertaModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
        self.hf_pretrained_encoder_with_pretrained_head = transformers.XLMRobertaForMaskedLM


@JiantTransformersModelFactory.register(ModelArchitectures.ELECTRA)
class JiantElectraModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
        self.hf_pretrained_encoder_with_pretrained_head = transformers.ElectraForPreTraining

    def __call__(self, encoder, input_ids, segment_ids, input_mask):
        output = super().__call__(
            input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask
        )
        unpooled = output.hidden_states
        pooled = unpooled[:, 0, :]
        return JiantModelOutput(pooled=pooled, unpooled=unpooled, other=output.hidden_states)


@JiantTransformersModelFactory.register(ModelArchitectures.BART)
class JiantBartModel(JiantTransformersModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
        self.hf_pretrained_encoder_with_pretrained_head = transformers.BartForConditionalGeneration

    def get_hidden_size(self):
        return self.config.d_model

    def get_hidden_dropout_prob(self):
        return self.config.dropout

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


@JiantTransformersModelFactory.register(ModelArchitectures.MBART)
class JiantMBartModel(JiantBartModel):
    def __init__(self, baseObject):
        super().__init__(baseObject)
