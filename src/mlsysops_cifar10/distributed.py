# configure the intra-op threads

import torch

def configure_cpu_threads(world_size: int):
    physical_cores = 4
    threads_per_process = max(1, physical_cores // world_size)

    torch.set_num_threads(threads_per_process)