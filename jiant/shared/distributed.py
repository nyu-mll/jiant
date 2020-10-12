from contextlib import contextmanager
import torch


@contextmanager
def only_first_process(local_rank):
    if local_rank not in [-1, 0]:
        # noinspection PyUnresolvedReferences
        torch.distributed.barrier()

    try:
        yield
    finally:
        if local_rank == 0:
            # noinspection PyUnresolvedReferences
            torch.distributed.barrier()
