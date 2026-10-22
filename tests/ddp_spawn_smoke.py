
"""Test CPU distributed communication using multiprocessing."""

import os
import socket

import torch
import torch.distributed as dist
import torch.multiprocessing as mp


def worker(rank, world_size, port):
    # Each worker is an independent Python process.
    torch.set_num_threads(1)

    # Create the rendezvous store without libuv.
    store = dist.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        use_libuv=False,
    )

    # Connect the workers using the Gloo backend.
    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    try:
        value = torch.tensor([float(rank + 1)])

        dist.all_reduce(value, op=dist.ReduceOp.SUM)

        expected = world_size * (world_size + 1) / 2

        assert value.item() == expected

        print(
            f"Rank {rank} | PID {os.getpid()} | "
            f"All-Reduce: {value.item()} | "
            f"Expected: {expected}",
            flush=True,
        )

    finally:
        dist.destroy_process_group()


def main():
    world_size = 4

    # Ask Windows for an available local TCP port.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    mp.spawn(
        worker,
        args=(world_size, port),
        nprocs=world_size,
        join=True,
    )


if __name__ == "__main__":
    main()
