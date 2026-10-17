# Download CIFAR-10, transform images, build loaders and distributed samplers

from pathlib import Path
from torchvision import datasets, transforms

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