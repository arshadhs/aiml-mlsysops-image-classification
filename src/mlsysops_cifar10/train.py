# Training loop, optimisation and checkpoint saving

"""CPU training for CIFAR-10 using single-process or PyTorch DDP."""

import argparse
import csv
import time
from pathlib import Path

import os
import socket

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

from datetime import datetime

from mlsysops_cifar10.model import CIFAR10CNN
from mlsysops_cifar10.data import (
    load_cifar10,
    create_distributed_train_loader,
)


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    max_batches=None,
    distributed=False,
):
    """Train one epoch and calculate performance metrics.

    Args:
        model: CNN or DDP-wrapped CNN.
        loader: Training DataLoader for this process.
        optimizer: Updates model weights.
        criterion: Calculates classification loss.
        max_batches: Optional limit for quick tests.
        distributed: Whether gradients are synchronized using DDP.

    Returns:
        Dictionary containing training loss, elapsed time,
        throughput, and total images processed.
    """
    model.train()

    # Track the weighted loss and number of processed images.
    total_loss = 0.0
    total_images = 0

    # Synchronize epoch timing across DDP workers.
    if distributed:
        dist.barrier()

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        # DDP automatically synchronizes gradients during backward().
        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    # Ensure all workers finish before measuring epoch duration.
    if distributed:
        dist.barrier()

    elapsed = time.perf_counter() - start

    if distributed:
        # Combine image counts and weighted loss from every rank.
        stats = torch.tensor(
            [total_loss, float(total_images)],
            dtype=torch.float64,
        )

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # Use the slowest worker's time for distributed performance.
        elapsed_tensor = torch.tensor(
            [elapsed],
            dtype=torch.float64,
        )
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)

        total_loss = stats[0].item()
        total_images = int(stats[1].item())
        elapsed = elapsed_tensor.item()

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


def find_free_port():
    """Find an available TCP port for distributed communication."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def train_worker(rank, world_size, port, args):
    """Run distributed CNN training in one CPU process.

    Args:
        rank: Unique process identifier, starting from zero.
        world_size: Total number of distributed processes.
        port: TCP port used for Gloo process coordination.
        args: Command-line training configuration.
    """

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    # Create TCPStore without libuv (Windows compatibility).
    store = dist.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        use_libuv=False,
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )

    try:
        print(
            f"Rank {rank} | "
            f"PID {os.getpid()} | "
            f"World size {world_size} | "
            f"Threads {torch.get_num_threads()} | "
            f"Gloo initialized: {dist.is_initialized()}",
            flush=True,
        )

        # Verify synchronization across all workers.
        # is a synchronization point. Every process must reach the barrier before any of them can continue past it.
        dist.barrier()

        print(
            f"Rank {rank} | Passed Gloo barrier",
            flush=True,
        )

        # Load CIFAR-10 using the same preprocessing as the baseline.
        train_dataset, test_dataset = load_cifar10()

        # Only Rank 0 performs model evaluation.
        # Other ranks don't require a test DataLoader.
        test_loader = None

        if rank == 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Partition the 50,000 training images across all ranks.
        train_loader, train_sampler = create_distributed_train_loader(
            train_dataset=train_dataset,
            rank=rank,
            world_size=world_size,
            global_batch_size=args.batch_size,
        )

        # Each worker receives an equal share of the dataset.
        local_images = len(train_sampler)

        # The local batch size is global batch size / world size.
        local_batch_size = train_loader.batch_size

        print(
            f"Rank {rank} | "
            f"Images: {local_images} | "
            f"Local batch: {local_batch_size}",
            flush=True,
        )

        # Create an identical CNN architecture in each worker.
        # DDP synchronizes initial parameters from Rank 0.
        model = CIFAR10CNN()

        # Wrap the model with DistributedDataParallel.
        # During backward(), DDP synchronizes gradients via Gloo.
        ddp_model = DDP(model)

        # Number of trainable parameters in our CNN.
        parameter_count = sum(
            parameter.numel()
            for parameter in ddp_model.parameters()
            if parameter.requires_grad
        )

        # Check that workers start with identical model weights.
        # We use the first trainable parameter as a simple checksum.
        first_parameter = next(ddp_model.parameters())
        parameter_sum = first_parameter.detach().sum().item()

        print(
            f"Rank {rank} | "
            f"DDP initialized: {isinstance(ddp_model, DDP)} | "
            f"Parameters: {parameter_count:,} | "
            f"First parameter sum: {parameter_sum:.6f}",
            flush=True,
        )


        # Use the same loss function and optimizer as our baseline.
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            ddp_model.parameters(),
            lr=args.lr,
        )

        # Rank 0 collects the aggregated metrics for CSV reporting.
        results = []


        # Train the CNN for the configured number of epochs.
        for epoch in range(1, args.epochs + 1):
            train_sampler.set_epoch(epoch - 1)

            metrics = train_one_epoch(
                model=ddp_model,
                loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                max_batches=args.max_batches,
                distributed=True,
            )

            if rank == 0:
                # Only Rank 0 evaluates and collects results.
                accuracy = evaluate(ddp_model.module, test_loader)

                metrics["epoch"] = epoch
                metrics["test_accuracy"] = accuracy
                metrics["processes"] = world_size
                metrics["threads_per_process"] = args.threads
                metrics["global_batch_size"] = args.batch_size

                results.append(metrics)

                print(
                    f"Epoch {epoch}/{args.epochs} | "
                    f"Loss: {metrics['train_loss']:.4f} | "
                    f"Time: {metrics['epoch_seconds']:.2f}s | "
                    f"Throughput: {metrics['throughput']:.1f} images/s | "
                    f"Accuracy: {accuracy:.2f}%",
                    flush=True,
                )

            # Wait until Rank 0 finishes evaluation.
            # All ranks must enter the next epoch together.
            dist.barrier()

        # IMPORTANT: This block is OUTSIDE the epoch loop.
        # Save all epoch results into one CSV file per run.
        if rank == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output = Path(
                f"results/benchmarks/"
                f"cpu_ddp_{world_size}processes_{timestamp}.csv"
            )

            output.parent.mkdir(parents=True, exist_ok=True)

            with output.open("w", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=results[0].keys(),
                )
                writer.writeheader()
                writer.writerows(results)

            print(f"Results saved to: {output}", flush=True)

        # Ensure all workers have finished before cleanup.
        dist.barrier()

    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=1)
    
    args = parser.parse_args()

    if args.processes not in (1, 2, 4):
        parser.error("--processes must be 1, 2, or 4")

    if args.processes > 1:
        port = find_free_port()

        mp.spawn(
            train_worker,
            args=(args.processes, port, args),
            nprocs=args.processes,
            join=True,
        )
        
        return
        
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
