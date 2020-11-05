import os
from typing import Union, Optional

import torch
import torch.nn as nn

import jiant.shared.caching as caching
import jiant.utils.python.io as py_io
import jiant.utils.torch_utils as torch_utils


def complex_backpropagate(
    loss, optimizer, model, fp16, n_gpu, gradient_accumulation_steps, max_grad_norm
):
    if n_gpu > 1:
        loss = loss.mean()  # mean() to average on multi-gpu.
    if gradient_accumulation_steps > 1:
        loss = loss / gradient_accumulation_steps
    if fp16:
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        from apex import amp

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), max_grad_norm)
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    return loss


def get_train_dataloader_from_cache(
    train_cache: caching.ChunkedFilesDataCache, task, train_batch_size: int
):
    # TODO: Expose buffer_size parameter  (issue #1183)
    dataset = train_cache.get_iterable_dataset(buffer_size=10000, shuffle=True)
    train_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset, batch_size=train_batch_size, collate_fn=task.collate_fn,
    )
    return train_dataloader


def get_eval_dataloader_from_cache(
    eval_cache: caching.ChunkedFilesDataCache,
    task,
    eval_batch_size: int,
    subset_num=None,
    explicit_subset=None,
):
    dataset = eval_cache.get_iterable_dataset(
        buffer_size=10000, shuffle=False, subset_num=subset_num, explicit_subset=explicit_subset,
    )
    eval_dataloader = torch_utils.DataLoaderWithLength(
        dataset=dataset, batch_size=eval_batch_size, collate_fn=task.collate_fn,
    )
    return eval_dataloader


def save_model_with_metadata(
    model_or_state_dict: Union[nn.Module, dict],
    output_dir: str,
    file_name="model",
    metadata: Optional[dict] = None,
):
    if isinstance(model_or_state_dict, dict):
        state_dict = model_or_state_dict
    else:
        state_dict = torch_utils.get_model_for_saving(model_or_state_dict).state_dict()

    torch.save(state_dict, os.path.join(output_dir, f"{file_name}.p"))
    if metadata is not None:
        py_io.write_json(metadata, os.path.join(output_dir, f"{file_name}.metadata.json"))


def compare_steps_max_steps(step, max_steps):
    return max_steps is not None and max_steps != -1 and step >= max_steps
