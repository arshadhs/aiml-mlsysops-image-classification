"""Generate benchmark comparisons for CIFAR-10 CPU training."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


# Benchmark directory containing the five-epoch CSV files.
BENCHMARK_DIR = Path("results/benchmarks")
FIGURE_DIR = Path("results/figures")

# Each configuration's recorded experiment, thread layout and label.
EXPERIMENTS = {
    "CPU 1x4": (
        "cpu_baseline_20261021_222856.csv",
        1,
        4,
    ),
    "DDP 2x2": (
        "cpu_ddp_2processes_20261022_224405.csv",
        2,
        2,
    ),
    "DDP 4x1": (
        "cpu_ddp_4processes_20261023_171311.csv",
        4,
        1,
    ),
}


def read_experiment(filename):
    """Read five epochs of measurements from one benchmark CSV.

    Args:
        filename: CSV filename within BENCHMARK_DIR.

    Returns:
        List of dictionaries containing numeric epoch metrics.
    """
    path = BENCHMARK_DIR / filename

    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5:
        raise ValueError(
            f"{path}: expected 5 epochs, found {len(rows)}"
        )

    # Convert stored CSV strings into numerical values.
    metrics = []
    for row in rows:
        metrics.append({
            "epoch": int(row["epoch"]),
            "seconds": float(row["epoch_seconds"]),
            "throughput": float(row["throughput"]),
            "accuracy": float(row["test_accuracy"]),
            "loss": float(row["train_loss"]),
            "images": int(row["images_processed"]),
        })

    if [r["epoch"] for r in metrics] != [1, 2, 3, 4, 5]:
        raise ValueError(f"{path}: invalid epoch sequence")

    if any(r["images"] != 50000 for r in metrics):
        raise ValueError(f"{path}: incomplete training epoch")

    return metrics


def save_line_chart(data, field, ylabel, filename):
    """Plot one metric against epoch for all configurations."""
    plt.figure(figsize=(8, 5))

    for name, rows in data.items():
        epochs = [row["epoch"] for row in rows]
        values = [row[field] for row in rows]
        plt.plot(epochs, values, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.xticks(range(1, 6))
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=180)
    plt.close()


def save_speedup_chart(summary):
    """Compare observed training-loop speedup against baseline."""
    plt.figure(figsize=(8, 5))

    labels = [row["configuration"] for row in summary]
    speedups = [row["speedup"] for row in summary]

    bars = plt.bar(labels, speedups)

    for bar, value in zip(bars, speedups):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{value:.2f}x",
            ha="center",
        )

    plt.ylabel("Observed speedup vs CPU 1x4")
    plt.ylim(0, max(speedups) * 1.20)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "speedup.png", dpi=180)
    plt.close()


def main():
    """Load experiment CSVs and create summary and figures."""
    parser = argparse.ArgumentParser()
    parser.parse_args()

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # Load the baseline and both DDP configurations.
    data = {
        name: read_experiment(filename)
        for name, (filename, _, _) in EXPERIMENTS.items()
    }

    # Total training time excludes evaluation and process startup.
    baseline_seconds = sum(
        row["seconds"] for row in data["CPU 1x4"]
    )

    summary = []

    for name, (_, processes, threads) in EXPERIMENTS.items():
        rows = data[name]
        seconds = sum(row["seconds"] for row in rows)
        images = sum(row["images"] for row in rows)

        summary.append({
            "configuration": name,
            "processes": processes,
            "threads_per_process": threads,
            "training_seconds": round(seconds, 2),
            "throughput": round(images / seconds, 2),
            "speedup": round(baseline_seconds / seconds, 3),
            "final_accuracy": rows[-1]["accuracy"],
        })

    # Save a machine-readable comparison for the assignment.
    output = BENCHMARK_DIR / "benchmark_summary.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary[0].keys(),
        )
        writer.writeheader()
        writer.writerows(summary)

    # Create the four figures used in the final report.
    save_line_chart(
        data, "seconds", "Training time (seconds)",
        "epoch_time.png",
    )
    save_line_chart(
        data, "throughput", "Images per second",
        "throughput.png",
    )
    save_line_chart(
        data, "accuracy", "Test accuracy (%)",
        "accuracy.png",
    )
    save_speedup_chart(summary)

    print("\nCIFAR-10 CPU benchmark comparison")
    print("-" * 75)

    for row in summary:
        print(
            f"{row['configuration']:10} | "
            f"{row['training_seconds']:7.2f}s | "
            f"{row['throughput']:8.1f} img/s | "
            f"{row['speedup']:.2f}x | "
            f"Accuracy {row['final_accuracy']:.2f}%"
        )

    print(f"\nSummary: {output}")
    print(f"Figures: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
