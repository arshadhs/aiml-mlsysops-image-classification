# Test distributed CIFAR-10 data sampling

from torch.utils.data import TensorDataset
import torch

from mlsysops_cifar10.data import create_distributed_train_loader

def test_four_rank_partition():
    """Four ranks should cover the dataset without overlap."""
    dataset = TensorDataset(torch.arange(50000))
    all_indices = []

    for rank in range(4):
        loader, sampler = create_distributed_train_loader(
            dataset,
            rank=rank,
            world_size=4,
            global_batch_size=128,
        )

        indices = list(iter(sampler))

        assert len(indices) == 12500
        assert loader.batch_size == 32

        all_indices.extend(indices)

    assert len(all_indices) == 50000
    assert len(set(all_indices)) == 50000
    assert set(all_indices) == set(range(50000))
    

def test_sampler_changes_between_epochs():
    """Shuffling should change across epochs."""
    dataset = TensorDataset(torch.arange(1000))

    loader, sampler = create_distributed_train_loader(
        dataset,
        rank=0,
        world_size=2,
        global_batch_size=128,
    )

    sampler.set_epoch(0)
    epoch_zero = list(iter(sampler))

    sampler.set_epoch(1)
    epoch_one = list(iter(sampler))

    assert epoch_zero != epoch_one


def test_global_batch_size():
    """Each configuration should preserve the global batch size."""
    dataset = TensorDataset(torch.arange(50000))

    for world_size in (1, 2, 4):
        loader, _ = create_distributed_train_loader(
            dataset,
            rank=0,
            world_size=world_size,
            global_batch_size=128,
        )

        assert loader.batch_size * world_size == 128
