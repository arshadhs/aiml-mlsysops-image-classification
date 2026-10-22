# Download CIFAR-10, transform images, build loaders and distributed samplers

from pathlib import Path
from torchvision import datasets, transforms

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def load_cifar10(data_dir="data"):
    """
    Download and load the CIFAR-10 dataset.

    Returns:
        train_dataset: 50,000 training images
        test_dataset: 10,000 testing images
    """

    data_dir = Path(data_dir)

    # Convert images into PyTorch tensors
    transform = transforms.ToTensor()

    # Load training dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    # Load testing dataset
    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset

def create_distributed_train_loader(
    train_dataset,
    rank: int,                      # Unique identifier of a process
    world_size: int,                # Total number of participating processes
    global_batch_size: int = 128,   # Batch size is the number of training images a neural network processes together before updating its weights.
):
    """Create a rank-specific CIFAR-10 training loader.

    Each rank receives a distinct portion of the dataset.
    The global batch size stays fixed across configurations.
    """
    if world_size < 1:
        raise ValueError("world_size must be at least 1")

    if not 0 <= rank < world_size:
        raise ValueError("rank must be between 0 and world_size - 1")

    if global_batch_size <= 0:
        raise ValueError("global_batch_size must be positive")

    if global_batch_size % world_size != 0:
        raise ValueError(
            "global_batch_size must be divisible by world_size"
        )

    local_batch_size = global_batch_size // world_size


    # sampler: Determines which image indices belong to the current process.
    sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=42,
        drop_last=False,
    )

    # loader: Supplies images to the CNN in batches.
    loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        sampler=sampler,
        num_workers=0,
        drop_last=False,
    )

    return loader, sampler