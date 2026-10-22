# Training loop, optimisation and checkpoint saving

"""Single-worker CPU baseline for CIFAR-10 training."""

import argparse
import csv
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from datetime import datetime

from mlsysops_cifar10.data import load_cifar10
from mlsysops_cifar10.model import CIFAR10CNN


def train_one_epoch(model, loader, optimizer, criterion, max_batches=None):
    """Train the CNN for one epoch and measure performance."""
    model.train()

    total_loss = 0.0
    total_images = 0

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    elapsed = time.perf_counter() - start

    return {
        "train_loss": total_loss / total_images,
        "epoch_seconds": elapsed,
        "throughput": total_images / elapsed,
        "images_processed": total_images,
    }


@torch.no_grad()
def evaluate(model, loader):
    """Calculate classification accuracy on the test set."""
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Reproducibility and CPU configuration
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    print("Device: CPU")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    train_dataset, test_dataset = load_cifar10()

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = CIFAR10CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    results = []

    for epoch in range(1, args.epochs + 1):
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            max_batches=args.max_batches,
        )

        accuracy = evaluate(model, test_loader)

        metrics["epoch"] = epoch
        metrics["test_accuracy"] = accuracy
        results.append(metrics)

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"Loss: {metrics['train_loss']:.4f} | "
            f"Time: {metrics['epoch_seconds']:.2f}s | "
            f"Throughput: {metrics['throughput']:.1f} images/s | "
            f"Accuracy: {accuracy:.2f}%"
        )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(f"results/benchmarks/cpu_baseline_{timestamp}.csv")   
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
