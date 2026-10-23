# Use a Linux image with Python 3.12.
# The container environment is independent of Windows Python.
FROM python:3.12-slim

# Disable Python bytecode and enable immediate console output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# All commands execute inside this directory.
WORKDIR /app

# Install CPU-only PyTorch to avoid unnecessary CUDA libraries.
# Pin these versions later for fully reproducible builds.
RUN pip install --no-cache-dir \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Copy project metadata and source code.
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install our package without replacing CPU PyTorch.
RUN pip install --no-cache-dir --no-deps .

# Provide directories for datasets and experiment results.
RUN mkdir -p data results/benchmarks results/figures

# Default command: run a quick CPU training smoke test.
CMD ["python", "-m", "mlsysops_cifar10.train", \
     "--epochs", "1", "--max-batches", "10", \
     "--processes", "1", "--threads", "2"]
