"""
Functions for parsing configs.
"""
import torch
import logging as log


def parse_task_list_arg(task_list):
    """Parse task list argument into a list of task names."""
    task_names = []
    for task_name in task_list.split(","):
        if task_name == "glue":
            task_names.extend(ALL_GLUE_TASKS)
        elif task_name == "superglue":
            task_names.extend(ALL_SUPERGLUE_TASKS)
        elif task_name == "none" or task_name == "":
            continue
        else:
            task_names.append(task_name)
    return task_names


def parse_cuda_list_arg(cuda_arg):
    """
    Parse cuda_list settings
    """
    result_cuda = []
    if cuda_arg == "auto":
        result_cuda = list(range(torch.cuda.device_count()))
        if len(result_cuda) == 1:
            result_cuda = result_cuda[0]
    elif isinstance(cuda_arg, int):
        result_cuda = cuda_arg
    elif "," in cuda_arg:
        result_cuda = [int(d) for d in cuda_arg.split(",")]
    else:
        raise ValueError(
            "Your cuda settings do not match any of the possibilities in defaults.conf"
        )
    if torch.cuda.device_count() == 0:
        # If there are no cuda devices available, default to -1.
        log.info("CUDA devices not found, defaulting to cuda=-1")
        result_cuda = -1
    return result_cuda
