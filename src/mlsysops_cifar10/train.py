
"""Train CIFAR-10 using CPU, CPU DDP, or a single CUDA GPU.

Supported configurations:
    - Single-process CPU training
    - Multi-process CPU training with PyTorch DDP and Gloo
    - Single-GPU training with CUDA

Training metrics are saved as timestamped CSV files.
"""

import argparse
import csv
import os
import socket
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity

from mlsysops_cifar10.data import (
    load_cifar10,
    create_distributed_train_loader,
)
from mlsysops_cifar10.model import CIFAR10CNN


def train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    max_batches=None,
    distributed=False,
    device=None,
):
    """Train the CNN for one epoch and measure its performance.

    Args:
        model: CNN or DDP-wrapped CNN.
        loader: DataLoader providing training images and labels.
        optimizer: Optimizer responsible for updating model weights.
        criterion: Loss function used to compare predictions and labels.
        max_batches: Optional batch limit for smoke testing.
        distributed: Whether CPU DDP is enabled.
        device: CPU or CUDA device used for computation.

    Returns:
        Dictionary containing average loss, training time,
        throughput, and number of images processed.
    """
    model.train()

    # Determine the device from the model if not specified.
    if device is None:
        device = next(model.parameters()).device

    # Accumulate image-weighted loss and processed image count.
    total_loss = 0.0
    total_images = 0

    # Start distributed workers from a common synchronization point.
    if distributed:
        dist.barrier()

    # GPU operations are asynchronous.
    # Wait for pending CUDA work before starting the timer.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start = time.perf_counter()

    for batch_index, (images, labels) in enumerate(loader):

        # Allow short experiments without processing the full dataset.
        if max_batches is not None and batch_index >= max_batches:
            break

        # Move training data onto the same device as the CNN.
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous training step.
        optimizer.zero_grad()

        # Forward pass: calculate class predictions.
        outputs = model(images)

        # Calculate cross-entropy classification loss.
        loss = criterion(outputs, labels)

        # Backward pass: calculate gradients.
        # With DDP, gradients are synchronized automatically.
        loss.backward()

        # Update the CNN parameters using the calculated gradients.
        optimizer.step()

        # Record loss weighted by the current batch size.
        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_images += batch_size

    # Ensure all CUDA operations have completed.
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Wait until all distributed workers finish training.
    if distributed:
        dist.barrier()

    elapsed = time.perf_counter() - start

    if distributed:
        # Combine image counts and weighted loss across all ranks.
        stats = torch.tensor(
            [total_loss, float(total_images)],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        # Distributed epoch time is determined by the slowest rank.
        elapsed_tensor = torch.tensor(
            [elapsed],
            dtype=torch.float64,
            device=device,
        )

        dist.all_reduce(
            elapsed_tensor,
            op=dist.ReduceOp.MAX,
        )

        total_loss = stats[0].item()
        total_images = int(stats[1].item())
        elapsed = elapsed_tensor.item()

    # All configurations return the same metric structure.
    return {
        "train_loss": total_loss / total_images,
        "epoch_seconds": elapsed,
        "throughput": total_images / elapsed,
        "images_processed": total_images,
    }


@torch.no_grad()
def evaluate(model, loader, device=None):
    """Evaluate classification accuracy on the CIFAR-10 test set.

    Args:
        model: Trained CNN.
        loader: Test DataLoader containing images and labels.
        device: CPU or CUDA device used by the model.

    Returns:
        Classification accuracy as a percentage.
    """
    model.eval()

    # Use the model's device when none is specified.
    if device is None:
        device = next(model.parameters()).device

    correct = 0
    total = 0

    for images, labels in loader:

        # Move test data onto the model's computing device.
        images = images.to(device)
        labels = labels.to(device)

        # Predict the class with the highest output score.
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def find_free_port():
    """Find an available local TCP port for distributed workers."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def save_results(results, device_name, processes):
    """Save aggregated experiment metrics to a timestamped CSV.

    Args:
        results: List of per-epoch metric dictionaries.
        device_name: Device identifier, such as cpu or cuda.
        processes: Number of participating training processes.

    Returns:
        Path of the generated CSV file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Preserve existing CPU and DDP filename conventions.
    if processes > 1:
        filename = (
            f"{device_name}_ddp_{processes}processes_{timestamp}.csv"
        )
    else:
        filename = f"{device_name}_baseline_{timestamp}.csv"

    output = Path("results/benchmarks") / filename
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=results[0].keys(),
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to: {output}", flush=True)
    return output


def train_worker(rank, world_size, port, args):
    """Run one CPU distributed training process.

    Args:
        rank: Unique worker identifier, starting from zero.
        world_size: Total number of distributed processes.
        port: TCP port used to coordinate the workers.
        args: Command-line training configuration.

    Each process trains a complete CNN replica on a different
    subset of CIFAR-10 images.
    """
    torch.manual_seed(args.seed)

    # Limit CPU computation threads available to each process.
    torch.set_num_threads(args.threads)

    # Assign each distributed process to its own GPU when CUDA is selected.
    use_cuda = args.device == "cuda" or (
        args.device == "auto" and torch.cuda.is_available()
    )

    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    # Use TCPStore without libuv for Windows compatibility.
    store = dist.TCPStore(
        host_name="127.0.0.1",
        port=port,
        world_size=world_size,
        is_master=(rank == 0),
        use_libuv=False,
    )

    # Initialize the communication group.
    # Gloo on Windows CPU and uses NCCL for distributed CUDA training
    dist.init_process_group(
        backend=backend,
        store=store,
        rank=rank,
        world_size=world_size,
        device_id=device if device.type == "cuda" else None,
    )

    try:
        print(
            f"Rank {rank} | "
            f"PID {os.getpid()} | "
            f"World size {world_size} | "
            f"Threads {torch.get_num_threads()} | "
            f"Backend: {backend} | "
            f"Device: {device} | "
            f"DDP initialized: {dist.is_initialized()}",
            flush=True,
        )

        # Ensure every rank has joined the process group.
        dist.barrier()

        print(
            f"Rank {rank} | Passed {backend} barrier",
            flush=True,
        )

        # Load CIFAR-10 using the same preprocessing as CPU baseline.
        train_dataset, test_dataset = load_cifar10()

        # Rank 0 alone evaluates test accuracy.
        test_loader = None

        if rank == 0:
            test_loader = DataLoader(
                test_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Partition CIFAR-10 across distributed workers.
        train_loader, train_sampler = (
            create_distributed_train_loader(
                train_dataset=train_dataset,
                rank=rank,
                world_size=world_size,
                global_batch_size=args.batch_size,
            )
        )

        # Images assigned to this worker for each epoch.
        local_images = len(train_sampler)

        # Images processed by this worker per training step.
        local_batch_size = train_loader.batch_size

        print(
            f"Rank {rank} | "
            f"Images: {local_images} | "
            f"Local batch: {local_batch_size}",
            flush=True,
        )

        # Put the complete CNN replica on this worker's device.
        model = CIFAR10CNN().to(device)

        if device.type == "cuda":
            # Each CUDA process controls exactly one GPU.
            ddp_model = DDP(
                model,
                device_ids=[rank],
                output_device=rank,
            )
        else:
            # Preserve existing CPU DDP behavior.
            ddp_model = DDP(model)

        # Count all trainable weights and biases in the CNN.
        parameter_count = sum(
            parameter.numel()
            for parameter in ddp_model.parameters()
            if parameter.requires_grad
        )

        # Initial parameter checksum for a basic consistency check.
        first_parameter = next(ddp_model.parameters())
        parameter_sum = first_parameter.detach().sum().item()

        print(
            f"Rank {rank} | "
            f"DDP initialized: {isinstance(ddp_model, DDP)} | "
            f"Parameters: {parameter_count:,} | "
            f"First parameter sum: {parameter_sum:.6f}",
            flush=True,
        )

        # Match the baseline optimizer and learning rate.
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            ddp_model.parameters(),
            lr=args.lr,
        )

        # Rank 0 stores aggregated results for the CSV.
        results = []

        for epoch in range(1, args.epochs + 1):

            # Change the shuffle order between epochs while
            # coordinating non-overlapping samples across ranks.
            train_sampler.set_epoch(epoch - 1)


            # Profile a short distributed run without changing the model
            # or gradient synchronization behavior.
            if args.profile_communication and device.type == "cuda":

                with profile(
                    activities=[
                        ProfilerActivity.CPU,
                        ProfilerActivity.CUDA,
                    ],
                ) as prof:

                    metrics = train_one_epoch(
                        model=ddp_model,
                        loader=train_loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        max_batches=args.max_batches,
                        distributed=True,
                        device=device,
                    )

                # Store a Chrome/Perfetto-compatible execution trace
                # for each GPU worker independently.
                profile_dir = Path("results/profiles")
                profile_dir.mkdir(parents=True, exist_ok=True)

                trace_file = profile_dir / f"gpu_rank{rank}_trace.json"
                prof.export_chrome_trace(str(trace_file))

                # Rank 0 prints the operation-level timing summary.
                if rank == 0:
                    print("\nGPU and NCCL Profiling Results")
                    print(
                        prof.key_averages().table(
                            sort_by="self_cuda_time_total",
                            row_limit=30,
                        ),
                        flush=True,
                    )
                    print(f"Trace saved: {trace_file}")

            else:
                # Normal training without profiler instrumentation.
                metrics = train_one_epoch(
                    model=ddp_model,
                    loader=train_loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    max_batches=args.max_batches,
                    distributed=True,
                    device=device,
                )


            if rank == 0:
                # Evaluate the underlying CNN on Rank 0 only.
                accuracy = evaluate(
                    ddp_model.module,
                    test_loader,
                    device=device,
                )

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

            # Wait for Rank 0 to complete evaluation.
            dist.barrier()

        # Write all five epochs to one CSV, on Rank 0 only.
        if rank == 0:
            save_results(
                results,
                device_name=device.type,
                processes=world_size,
            )

        # Prevent other ranks from exiting before result saving.
        dist.barrier()

    finally:
        # Release Gloo communication resources.
        dist.destroy_process_group()


def main():
    """Parse arguments and select single-process or DDP training."""
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--processes", type=int, default=1)

    # Enable profiling of CPU operations and CUDA kernels.
    parser.add_argument(
        "--profile-communication",
        action="store_true",
        help="Profile GPU operations and NCCL gradient communication",
    )
    
    # Device choice: CPU, NVIDIA CUDA GPU, or automatic selection.
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "auto"),
        default="cpu",
    )

    args = parser.parse_args()

    # Validate experiment parameters before running.
    if args.processes not in (1, 2, 4):
        parser.error("--processes must be 1, 2, or 4")

    if args.epochs < 1:
        parser.error("--epochs must be at least 1")

    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    if args.threads < 1:
        parser.error("--threads must be positive")

    if args.max_batches is not None and args.max_batches < 1:
        parser.error("--max-batches must be positive")

    if args.batch_size % args.processes != 0:
        parser.error("--batch-size must be divisible by --processes")

    # Resolve the computing device.
    if args.device == "cuda":
        if not torch.cuda.is_available():
            parser.error(
                "CUDA requested, but no CUDA GPU is available"
            )
        device = torch.device("cuda")

    elif args.device == "auto":
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    else:
        device = torch.device("cpu")

    # DDP implementation uses CPU workers and Gloo.
    # Multi-GPU training require a CUDA/NCCL DDP path.
    
    # One distributed worker must be assigned to each GPU.
    if args.processes > 1 and device.type == "cuda":
        if not dist.is_nccl_available():
            parser.error("NCCL is not available in this PyTorch build")

        if args.processes > torch.cuda.device_count():
            parser.error(
                "Requested more CUDA processes than available GPUs"
            )

    # Launch multiple processes only for CPU DDP.
    if args.processes > 1:

        # All workers must connect through the same TCP port.
        port = find_free_port()

        mp.spawn(
            train_worker,
            args=(args.processes, port, args),
            nprocs=args.processes,
            join=True,
        )

        return

    # Single-process CPU or CUDA training.
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    print(f"Device: {device}")
    print(f"PyTorch threads: {torch.get_num_threads()}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    # Load the same CIFAR-10 dataset on either device.
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

    # Move model parameters to the selected device.
    model = CIFAR10CNN().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    results = []

    for epoch in range(1, args.epochs + 1):

        metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            max_batches=args.max_batches,
            distributed=False,
            device=device,
        )

        # Evaluate on the same device used for training.
        accuracy = evaluate(
            model,
            test_loader,
            device=device,
        )

        # Store common metrics for CPU/GPU visualizations.
        metrics["epoch"] = epoch
        metrics["test_accuracy"] = accuracy
        metrics["device"] = device.type
        metrics["processes"] = 1
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

    # Save results from the single-process experiment.
    save_results(
        results,
        device_name=device.type,
        processes=1,
    )


if __name__ == "__main__":
    main()
