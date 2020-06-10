import json
import numpy as np
import os
import random
import time
import torch
from dataclasses import dataclass
from typing import Any

import jiant.utils.python.io as py_io
import jiant.utils.zlog as zlog


@dataclass
class QuickInitContainer:
    device: Any
    n_gpu: int
    log_writer: Any


def quick_init(args, verbose=True) -> QuickInitContainer:
    """Sets up logging, initializes device(s) and random seed, prepares output dir, and saves args."

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        verbose (bool): whether to print the input run config and the run config as saved.

    Returns:
        QuickInitContainer specifying the run's device, GPU count, and logging configuration.

    """
    if verbose:
        print_args(args)
    init_server_logging(server_ip=args.server_ip, server_port=args.server_port, verbose=verbose)
    device, n_gpu = init_cuda_from_args(
        no_cuda=args.no_cuda, local_rank=args.local_rank, fp16=args.fp16, verbose=verbose,
    )
    args.seed = init_seed(given_seed=args.seed, n_gpu=n_gpu, verbose=verbose)
    init_output_dir(output_dir=args.output_dir, force_overwrite=args.force_overwrite)
    log_writer = init_log_writer(output_dir=args.output_dir)
    save_args(args=args, verbose=verbose)
    return QuickInitContainer(device=device, n_gpu=n_gpu, log_writer=log_writer)


def init_server_logging(server_ip, server_port, verbose=True):
    """Sets ups Python Tools for Visual Studio debug (ptvsd) server.

    Adapted from Hugging Face template: https://github.com/huggingface/transformers/blob/ac99217
    e92c43066af7ec96554054d75532565d7/templates/adding_a_new_example_script/run_xxx.py#L569-L576

    """
    if server_ip and server_port:
        # Distant debugging, see:
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        # noinspection PyUnresolvedReferences,PyPackageRequirements
        import ptvsd

        if verbose:
            print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(server_ip, server_port), redirect_output=True)
        ptvsd.wait_for_attach()


def init_cuda_from_args(no_cuda, local_rank, fp16, verbose=True):
    """Perform initial CUDA setup for DistributedDataParallel, DataParallel or w/o CUDA configs.

    Adapted from Hugging Face template: https://github.com/huggingface/transformers/blob/ac99217e92
    c43066af7ec96554054d75532565d7/templates/adding_a_new_example_script/run_xxx.py#L578-L586

    Args:
        no_cuda (bool): True to ignore CUDA devices (i.e., use CPU instead).
        local_rank (int): Which GPU the script should use in DistributedDataParallel mode.
        fp16 (bool): True for half-precision mode.
        verbose: True to print device, device count, and whether training is distributed or FP16.

    Notes:
        local_rank == -1 is used to indicate that DistributedDataParallel should be disabled.
        n_gpu > 1 is used to indicate that DataParallel should be used. Currently, local_rank == -1
        sets n_gpu = 1 even if torch.cuda.device_count() would show more than one GPU is available.

    Returns:
        (tuple): tuple containing:
            device (str): string handle for device.
            n_gpu (int): number of GPU devices.

    """
    # TODO break local_rank == -1 and no_cuda into separate cases to make the logic easier to read.
    if local_rank == -1 or no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        # noinspection PyUnresolvedReferences
        torch.distributed.init_process_group(backend="nccl")
    if verbose:
        print(
            "device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
                device, n_gpu, bool(local_rank != -1), fp16
            )
        )

    return device, n_gpu


def init_seed(given_seed, n_gpu, verbose=True):
    """Initializes random seeds for sources of randomness. If seed is -1, randomly select seed.

    Sets the random seed for sources of randomness (numpy, torch and python random). If seed is
    specified as -1, the seed will be randomly selected and used to initialize all random seeds.
    The value used to initialize the random seeds is returned.

    Args:
        given_seed (int): random seed.
        n_gpu (int): number of GPUs.
        verbose: whether to print random seed.

    Returns:
        int: value used to initialize random seeds.

    """
    used_seed = get_seed(given_seed)
    random.seed(used_seed)
    np.random.seed(used_seed)
    torch.manual_seed(used_seed)
    if verbose:
        print("Using seed: {}".format(used_seed))

    if n_gpu > 0:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed_all(used_seed)

    # MAKE SURE THIS IS SET
    return used_seed


def init_output_dir(output_dir, force_overwrite):
    """Create output directory (and all intermediate dirs on the path) if it doesn't exist.

    Args:
        output_dir (str): output directory path.
        force_overwrite (bool): If False and output dir is complete, raise RuntimeError.

    Raises:
        RuntimeError if overwrite option is not enabled and output dir contains "DONE" signal file.

    """
    if not force_overwrite and is_done(output_dir):
        raise RuntimeError(f"'{output_dir}' run is already done, and not forcing overwrite")
    os.makedirs(output_dir, exist_ok=True)


def init_log_writer(output_dir):
    return zlog.ZLogger(os.path.join(output_dir, str(int(time.time()))), overwrite=True)


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def save_args(args, verbose=True):
    """Dumps RunConfiguration to a json file.

    Args:
        args (RunConfiguration): configuration carrying command line args specifying run params.
        verbose (bool): If True, print the arg object that was written to file.

    """
    formatted_args = json.dumps(vars(args), indent=2)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        f.write(formatted_args)
    if verbose:
        print(formatted_args)


def get_seed(seed):
    """Get random seed if seed is specified as -1, otherwise return seed.

    Args:
        seed (int): random seed.

    Returns:
        int: Random seed if seed is specified as -1, otherwise returns the provided input seed.

    """
    if seed == -1:
        return int(np.random.randint(0, 2 ** 32 - 1))
    else:
        return seed


def write_done(output_dir):
    py_io.write_file("DONE", os.path.join(output_dir, "DONE"))


def is_done(output_dir):
    return os.path.exists(os.path.join(output_dir, "DONE"))
