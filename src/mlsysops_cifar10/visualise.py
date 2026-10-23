
"""Compare CIFAR-10 CPU and GPU training benchmarks.

Reads experiments.csv and generates reproducible result figures.
Run from the repository root:
    python -m mlsysops_cifar10.visualise
"""

import csv
from pathlib import Path

import matplotlib

# Save figures without needing an interactive display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# Paths are relative to the repository root.
BENCHMARK_DIR = Path("results/benchmarks")
FIGURE_DIR = Path("results/figures")
MANIFEST = BENCHMARK_DIR / "experiments.csv"


def read_benchmark(filename):
    """Read and validate one five-epoch training benchmark.

    Args:
        filename: CSV filename from the experiment manifest.

    Returns:
        List of dictionaries with numeric training metrics.
    """
    path = BENCHMARK_DIR / filename

    with path.open(newline="", encoding="utf-8-sig") as file:
        records = list(csv.DictReader(file))

    if len(records) != 5:
        raise ValueError(f"{path}: expected 5 epochs")

    rows = []

    for record in records:
        rows.append({
            "epoch": int(record["epoch"]),
            "train_loss": float(record["train_loss"]),
            "epoch_seconds": float(record["epoch_seconds"]),
            "throughput": float(record["throughput"]),
            "test_accuracy": float(record["test_accuracy"]),
            "images_processed": int(record["images_processed"]),
        })

    if [row["epoch"] for row in rows] != list(range(1, 6)):
        raise ValueError(f"{path}: invalid epoch sequence")

    if any(row["images_processed"] != 50000 for row in rows):
        raise ValueError(f"{path}: incomplete training epoch")

    return rows


def load_experiments():
    """Load experiment metadata and corresponding benchmark rows."""
    with MANIFEST.open(newline="", encoding="utf-8-sig") as file:
        manifest_rows = list(csv.DictReader(file))

    experiments = []

    for record in manifest_rows:
        # Each manifest row describes a single completed training run.
        experiments.append({
            "configuration": record["configuration"],
            "platform": record["platform"],
            "device": record["device"].upper(),
            "processes": int(record["processes"]),
            "threads": int(record["threads"] or 0),
            "file": record["file"],
            "epochs": read_benchmark(record["file"]),
        })

    if not experiments:
        raise ValueError("No experiments found")

    return experiments


def create_summary(experiments):
    """Calculate total training time, throughput and final accuracy."""
    summary = []

    # Identify the reference CPU experiment for speedup calculations.
    baselines = [
        exp for exp in experiments
        if exp["device"] == "CPU" and exp["processes"] == 1
    ]

    if len(baselines) != 1:
        raise ValueError("Exactly one single-process CPU baseline is required")

    baseline_time = sum(
        row["epoch_seconds"] for row in baselines[0]["epochs"]
    )

    for exp in experiments:
        # Sum training-only epoch timings across the complete run.
        total_seconds = sum(
            row["epoch_seconds"] for row in exp["epochs"]
        )

        total_images = sum(
            row["images_processed"] for row in exp["epochs"]
        )

        # Only CPU configurations receive CPU-relative speedup.
        speedup = (
            baseline_time / total_seconds
            if exp["device"] == "CPU"
            else None
        )

        summary.append({
            "configuration": exp["configuration"],
            "platform": exp["platform"],
            "device": exp["device"],
            "processes": exp["processes"],
            "threads_per_process": exp["threads"],
            "training_seconds": round(total_seconds, 2),
            "throughput": round(total_images / total_seconds, 2),
            "cpu_speedup": (
                round(speedup, 3) if speedup is not None else ""
            ),
            "final_accuracy": exp["epochs"][-1]["test_accuracy"],
        })

    return summary


def line_chart(experiments, field, ylabel, filename):
    """Plot an epoch-level metric for selected experiments."""
    plt.figure(figsize=(8, 5))

    for exp in experiments:
        epochs = exp["epochs"]

        plt.plot(
            [row["epoch"] for row in epochs],
            [row[field] for row in epochs],
            marker="o",
            label=exp["configuration"],
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close()


def bar_chart(summary, field, ylabel, filename):
    """Create a comparison bar chart from summary measurements."""
    labels = [row["configuration"] for row in summary]
    values = [float(row[field]) for row in summary]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)

    # Put the measured value above each bar.
    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.2f}",
            ha="center",
            va="bottom",
        )

    plt.ylabel(ylabel)
    plt.ylim(0, max(values) * 1.18)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close()


def save_summary(summary):
    """Write one summary CSV for the assignment and README."""
    output = BENCHMARK_DIR / "benchmark_summary.csv"

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    return output



def plot_gpu_scaling(summary, figures_dir):
    """Plot GPU speedup and parallel efficiency.

    Uses the measured single-GPU training time as the baseline.
    Only Kaggle CUDA experiments are included.
    """
    gpu_runs = [
        row for row in summary
        if row["device"].lower() == "cuda"
        and row["platform"].lower() == "kaggle"
    ]

    if not gpu_runs:
        return

    # Sort configurations by GPU/process count.
    gpu_runs.sort(key=lambda row: int(row["processes"]))

    # Find the single-GPU reference measurement.
    baseline = next(
        (
            row for row in gpu_runs
            if int(row["processes"]) == 1
        ),
        None,
    )

    if baseline is None:
        return

    baseline_time = float(baseline["training_seconds"])

    labels = []
    speedups = []
    efficiencies = []

    for row in gpu_runs:
        gpu_count = int(row["processes"])
        training_time = float(row["training_seconds"])

        # Strong-scaling speedup relative to one GPU.
        speedup = baseline_time / training_time

        # Parallel efficiency relative to ideal linear speedup.
        efficiency = 100.0 * speedup / gpu_count

        labels.append(f"{gpu_count} GPU")
        speedups.append(speedup)
        efficiencies.append(efficiency)

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # GPU scaling speedup.
    plt.figure(figsize=(7, 5))
    plt.bar(labels, speedups)
    # Ideal speedup equals the number of GPUs.
    plt.plot(
        labels,
        [int(row["processes"]) for row in gpu_runs],
        linestyle="--",
        marker="o",
        label="Ideal scaling",
    )
    plt.ylabel("Speedup (×)")
    plt.title("GPU Strong Scaling — Speedup")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "gpu_speedup.png", dpi=150)
    plt.close()

    # GPU parallel efficiency.
    plt.figure(figsize=(7, 5))
    plt.bar(labels, efficiencies)
    plt.axhline(100, linestyle="--", label="Ideal efficiency")
    plt.ylabel("Parallel efficiency (%)")
    plt.title("GPU Strong Scaling — Parallel Efficiency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "gpu_efficiency.png", dpi=150)
    plt.close()


def plot_scaling_comparison(summary):
    """Compare CPU and GPU scaling using separate reference baselines.

    CPU experiments maintain a fixed total of four PyTorch threads.
    GPU experiments scale from one to two physical Tesla T4 GPUs.

    The two series represent different scaling experiments.
    """
    cpu_runs = sorted(
        [
            row for row in summary
            if row["device"] == "CPU"
            and row["platform"] == "Windows"
        ],
        key=lambda row: int(row["processes"]),
    )

    gpu_runs = sorted(
        [
            row for row in summary
            if row["device"] == "CUDA"
            and row["platform"] == "Kaggle"
        ],
        key=lambda row: int(row["processes"]),
    )

    if not cpu_runs or not gpu_runs:
        return

    # Reference configurations for each independent experiment.
    cpu_baseline = next(
        row for row in cpu_runs if row["processes"] == 1
    )
    gpu_baseline = next(
        row for row in gpu_runs if row["processes"] == 1
    )

    cpu_time = float(cpu_baseline["training_seconds"])
    gpu_time = float(gpu_baseline["training_seconds"])

    cpu_processes = [row["processes"] for row in cpu_runs]
    gpu_processes = [row["processes"] for row in gpu_runs]

    cpu_speedups = [
        cpu_time / float(row["training_seconds"])
        for row in cpu_runs
    ]

    gpu_speedups = [
        gpu_time / float(row["training_seconds"])
        for row in gpu_runs
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        cpu_processes,
        cpu_speedups,
        marker="o",
        linewidth=2,
        label="CPU DDP (fixed 4-thread budget)",
    )

    plt.plot(
        gpu_processes,
        gpu_speedups,
        marker="s",
        linewidth=2,
        label="GPU DDP (Tesla T4)",
    )

    # Reference speedup of each single-process configuration.
    plt.axhline(
        1.0,
        color="gray",
        linestyle=":",
        label="Baseline (1.00×)",
    )

    plt.xlabel("Number of training processes / GPUs")
    plt.ylabel("Speedup relative to respective baseline (×)")
    plt.title("CPU and GPU Scaling Comparison")
    plt.xticks(sorted(set(cpu_processes + gpu_processes)))
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "scaling_comparison.png",
        dpi=180,
    )
    plt.close()


def main():
    """Generate CPU scaling and cross-platform performance figures."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    experiments = load_experiments()
    summary = create_summary(experiments)

    # Compare observed CPU and GPU scaling on one chart.
    plot_scaling_comparison(summary)

    # Compare epoch-level training times across all five configurations.
    line_chart(
        experiments,
        "epoch_seconds",
        "Training time (seconds)",
        "all_epoch_time.png",
    )
    
    # Restrict parallel-scaling comparisons to the same CPU machine.
    cpu_experiments = [
        exp for exp in experiments
        if exp["device"] == "CPU" and exp["platform"] == "Windows"
    ]

    cpu_summary = [
        row for row in summary
        if row["device"] == "CPU" and row["platform"] == "Windows"
    ]

    if cpu_experiments:
        line_chart(
            cpu_experiments, "epoch_seconds",
            "Training time (seconds)", "cpu_epoch_time.png",
        )
        line_chart(
            cpu_experiments, "throughput",
            "Images per second", "cpu_throughput.png",
        )
        bar_chart(
            cpu_summary, "cpu_speedup",
            "Observed speedup vs CPU 1x4", "cpu_speedup.png",
        )

    # Accuracy comparison can include both CPU and future GPU runs.
    line_chart(
        experiments, "test_accuracy",
        "Test accuracy (%)", "accuracy_comparison.png",
    )

    # Create hardware comparisons only when GPU results are available.
    if any(exp["device"] == "CUDA" for exp in experiments):
        bar_chart(
            summary, "training_seconds",
            "Training time (seconds)", "cpu_vs_gpu_time.png",
        )
        bar_chart(
            summary, "throughput",
            "Images per second", "cpu_vs_gpu_throughput.png",
        )

    output = save_summary(summary)

    print("\nCIFAR-10 Benchmark Summary")
    print("-" * 83)

    for row in summary:
        speedup = row["cpu_speedup"]
        speedup_text = f"{speedup:.2f}x" if speedup != "" else "N/A"

        print(
            f"{row['configuration']:12} | "
            f"{row['training_seconds']:7.2f}s | "
            f"{row['throughput']:8.1f} images/s | "
            f"Speedup {speedup_text:5} | "
            f"Accuracy {row['final_accuracy']:.2f}%"
        )

    # Generate GPU scaling charts when CUDA experiments are present.
    # Generate GPU speedup and parallel efficiency charts.
    plot_gpu_scaling(summary, FIGURE_DIR)
    
    print(f"\nSummary saved: {output}")
    print(f"Figures saved: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
