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
    if not force_overwrite and is_done(output_dir):
        raise RuntimeError(f"'{output_dir}' run is already done, and not forcing overwrite")
    os.makedirs(output_dir, exist_ok=True)


def init_log_writer(output_dir):
    return zlog.ZLogger(os.path.join(output_dir, str(int(time.time()))), overwrite=True)


def print_args(args):
    for k, v in vars(args).items():
        print("  {}: {}".format(k, v))


def save_args(args, verbose=True):
    formatted_args = json.dumps(vars(args), indent=2)
    with open(os.path.join(args.output_dir, "args.json"), "w") as f:
        f.write(formatted_args)
    if verbose:
        print(formatted_args)


def get_seed(seed):
    if seed == -1:
        return int(np.random.randint(0, 2 ** 32 - 1))
    else:
        return seed


def write_done(output_dir):
    py_io.write_file("DONE", os.path.join(output_dir, "DONE"))


def is_done(output_dir):
    return os.path.exists(os.path.join(output_dir, "DONE"))
