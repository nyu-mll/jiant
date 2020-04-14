def batch_size_limit_to_gpus(batch_size_limit, jiant):
    if batch_size_limit <= 4:
        gpu_available, sbatch = 4, ("4jp40.sbatch" if jiant else "4p40.sbatch")
    elif batch_size_limit == 8:
        gpu_available, sbatch = 2, ("2jp40.sbatch" if jiant else "2p40.sbatch")
    else:
        gpu_available, sbatch = 1, ("jp40.sbatch" if jiant else "p40.sbatch")
    return gpu_available, sbatch


def batch_size_to_accumulation(batch_size_limit, batch_size, gpu_available):
    gpu_needed = batch_size // batch_size_limit
    if gpu_needed <= gpu_available:
        real_batch_size = batch_size
        accumulation_steps = 1
    else:
        assert gpu_needed % gpu_available == 0
        accumulation_steps = gpu_needed // gpu_available
        assert batch_size % accumulation_steps == 0
        real_batch_size = batch_size // accumulation_steps

    return real_batch_size, accumulation_steps
