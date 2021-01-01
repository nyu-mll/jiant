from typing import Dict, Union
import copy
import torch
import torch.nn as nn

import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.tasks as tasks
import jiant.proj.main.modeling.heads as heads
from jiant.proj.main.components.outputs import construct_output_from_dict, LogitsOutput, LogitsAndLossOutput
import jiant.shared.task_aware_unit as tau
import jiant.proj.main.modeling.modules as jiantmodules


class JiantModel(nn.Module):
    def __init__(
        self,
        task_dict: Dict[str, tasks.Task],
        encoder: nn.Module,
        taskmodels_dict: Dict[str, taskmodels.Taskmodel],
        task_to_taskmodel_map: Dict[str, str],
        tokenizer,
        args,
    ):
        super().__init__()
        self.task_dict = task_dict
        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)
        self.task_to_taskmodel_map = task_to_taskmodel_map
        self.tokenizer = tokenizer
        self.weight_regularization_type = args.weight_regularization_type
        self.weight_regularization_coef = args.weight_regularization_coef
        if self.weight_regularization_type == "EWC":
            self.saved_weights = copy.deepcopy(list(self.encoder.encoder.parameters()))
        self.model_taus = tau.create_tau_dict(list(self.named_modules()))

    def forward(self,
                batch: tasks.BatchMixin,
                task: tasks.Task,
                compute_loss: bool = False,
                unreduced_loss: bool = False):
        """Calls to this forward method are delegated to the forward of the appropriate taskmodel.

        When JiantModel forward is called, the task name from the task argument is used as a key
        to select the appropriate submodule/taskmodel, and that taskmodel's forward is called.

        Args:
            batch (tasks.BatchMixin): model input.
            task (tasks.Task): task to which to delegate the forward call.
            compute_loss (bool): whether to calculate and return the loss.
            unreduced_loss (bool): whether to return instance-wise losses or reduce them.

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
        tau.set_tau_task(self.model_taus, task_name)
        outputs = taskmodel(
            batch=batch, task=task, tokenizer=self.tokenizer,
            compute_loss=compute_loss, unreduced_loss=unreduced_loss
        ).to_dict()
        if compute_loss and self.compute_weight_regularization() is not None:
            outputs["loss"] = outputs["loss"] + self.compute_weight_regularization()
        return outputs

    def compute_weight_regularization(self):
        if self.weight_regularization_type == "EWC":
            diff_norm = [
                (p - q).pow(2).sum()
                for p, q in zip(self.encoder.encoder.parameters(), self.saved_weights)
            ]
            return self.weight_regularization_coef * sum(diff_norm)
        else:
            return None


class JiantModelWithDDSModel(JiantModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        encoder = copy.deepcopy(kwargs["encoder"])
        regression_head = heads.RegressionHead(
            hidden_size=encoder.config.hidden_size,
            hidden_dropout_prob=encoder.config.hidden_dropout_prob,
        )
        # # NOTE: Do not change attribute dds_model to something else.
        # # If you do, make sure to change it in runscript.py as well (after delegate_load_from_path)
        self.dds_model = taskmodels.RegressionModel(
            encoder=encoder,
            regression_head=regression_head
        )

    def dds_weights_forward(
            self,
            batch: tasks.BatchMixin,
            rewards: torch.Tensor = None,
            compute_loss: bool = False
        ):
        dds_weight_logits = self.dds_model(
            batch=batch, task=None, tokenizer=None, compute_loss=False
        ).logits.view(-1)
        dds_weights = dds_weight_logits.softmax(dim=-1)

        if not compute_loss:
            return LogitsOutput(logits=dds_weights)

        dds_loss = -(rewards*dds_weights).sum()

        return LogitsAndLossOutput(logits=dds_weights, loss=dds_loss)


class JiantModelWithAdapterFusion(JiantModel):
    def __init__(self, attention_fusion, freeze_transformer=False, freeze_adapters=False, **kwargs):
        super().__init__(**kwargs)
        self.attention_fusion = attention_fusion
        for i, layer in enumerate(self.encoder.encoder.layer):
            if self.attention_fusion:
                self.encoder.encoder.layer[i].output = jiantmodules.BertOutputWithAdapterFusion(
                    layer.output,
                    self.task_dict.keys(),
                    self.encoder.config.hidden_size,
                    reduction_factor=16,
                    non_linearity="relu",
                )
            else:
                self.encoder.encoder.layer[i].output = jiantmodules.BertOutputWithAdapter(
                    layer.output,
                    self.task_dict.keys(),
                    self.encoder.config.hidden_size,
                    reduction_factor=16,
                    non_linearity="relu",
                )

        if freeze_transformer:
            for p in self.parameters():
                p.requires_grad = False
        if not freeze_adapters:
            for layer in self.encoder.encoder.layer:
                for p in layer.output.adapters.parameters():
                    p.requires_grad = True
        if self.attention_fusion:
            for layer in self.encoder.encoder.layer:
                for p in (
                    list(layer.output.key_layer.parameters())
                    + list(layer.output.value_layer.parameters())
                    + list(layer.output.query_layer.parameters())
                ):
                    p.requires_grad = True

        self.model_taus = tau.create_tau_dict(list(self.named_modules()))

    def forward(self, batch: tasks.BatchMixin, task: tasks.Task, compute_loss: bool = False):
        outputs = JiantModel.forward(batch, task, compute_loss)
        if compute_loss and self.attention_fusion:
            fusion_regularization = torch.sum(
                [
                    layer.output.value_layer.weight
                    - torch.ones_like(layer.output.value_layer.weight.diag()).diag()
                    for layer in self.encoder.encoder.layer
                ]
            )
            outputs["loss"] = outputs["loss"] + fusion_regularization
        return outputs


class JiantModelWithSluice(JiantModel):
    def __init__(
        self, task_a, task_b, sluice_num_subspaces, sluice_init_var, sluice_lr_multiplier, **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder.encoder = jiantmodules.SluiceEncoder(
            self.encoder.encoder,
            self.encoder.config,
            task_a,
            task_b,
            sluice_num_subspaces,
            sluice_init_var,
            sluice_lr_multiplier,
        )

        self.model_taus = tau.create_tau_dict(list(self.named_modules()))


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
    if is_multi_gpu:
        model_output.loss = model_output.loss.mean()
    return model_output
