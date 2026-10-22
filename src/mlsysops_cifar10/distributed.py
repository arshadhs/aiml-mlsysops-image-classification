"""Utilities for Pytorch distributed training."""

import os

import torch
import torch.distributed as dist

# configure the intra-op threads
def configure_cpu_threads(world_size: int):
    physical_cores = 4
    threads_per_process = max(1, physical_cores // world_size)

    torch.set_num_threads(threads_per_process)

def setup_distributed():
    """Initialise the distributed process group using Gloo."""
    
    # torchrun provides these environment variables
    rank = init(os.environ["RANK"])                 # Unique identifier of a process
    world_size  = init(os.environ["WORLD_SIZE"])    # Total number of participating processes
    local_rank = init(os.environ["LOCAL_RANK"])     # Process identifier on the current machine

    # GLOO supports CPU-based distributed communication
    # Communication backend used for CPU DDP
    # Connects workers so they can communicate
    dist.initi_process_group(
        backend = "gloo",
        init_method="env://"
    )

    print(
        f"Rank: {rank} | "
        f"World size: {world_size} | "
        f"PID: {os.getpid()} | "
        f"Threads: {torch.get_num_threads()}",        
        flush=True
    )
    
    return rank, world_size

def cleanup_distributed():
    """Release distributed communication resources"""
    if dist.is_initialized():
        dist.destroy_process_group()