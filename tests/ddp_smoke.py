
"""Smoke test for distributed CPU communication."""

import torch
import torch.distributed as dist

from mlsysops_cifar10.distributed import (
    setup_distributed,
    cleanup_distributed,
)


def main():
    rank, world_size = setup_distributed()

    try:
        # Each process starts with a different value
        value = torch.tensor([float(rank + 1)])

        # Sum values across all processes
        dist.all_reduce(value, op=dist.ReduceOp.SUM)

        expected = world_size * (world_size + 1) / 2

        assert value.item() == expected

        print(
            f"Rank {rank}: "
            f"All-Reduce result = {value.item()} "
            f"(expected {expected})",
            flush=True
        )

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
