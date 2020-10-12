from typing import Dict, Union

import torch.nn as nn

import jiant.proj.main.modeling.taskmodels as taskmodels
import jiant.tasks as tasks
from jiant.proj.main.components.outputs import construct_output_from_dict


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
            batch=batch, task=task, tokenizer=self.tokenizer, compute_loss=compute_loss,
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
    if is_multi_gpu:
        model_output.loss = model_output.loss.mean()
    return model_output
