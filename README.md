# aiml-mlsysops-image-classification

## 1. Overview

Design, implement, and evaluate a parallelised or distributed machine learning system for training an image classification model on the CIFAR-10 dataset.

## 2. Objectives

Model architecture, parallelism strategy, execution platform, and development framework — and rigorously justify each decision.

## 3. Dataset — CIFAR-10

CIFAR-10 consists of 60,000 colour images (32×32 pixels) across 10 classes (50,000 training / 10,000 test). It is a well-understood benchmark, allowing to focus effort on the systems and optimisation aspects rather than dataset preparation.

## 4. CNN Architecture


flowchart TD
    A["CIFAR-10 Input Image<br/>3 × 32 × 32"]

    B["Convolution Layer 1<br/>3 → 32 channels, 3×3 kernel"]
    C["ReLU Activation"]
    D["Max Pooling 2×2<br/>32 × 16 × 16"]

    E["Convolution Layer 2<br/>32 → 64 channels, 3×3 kernel"]
    F["ReLU Activation"]
    G["Max Pooling 2×2<br/>64 × 8 × 8"]

    H["Flatten<br/>4096 features"]
    I["Fully Connected Layer<br/>4096 → 128"]
    J["ReLU Activation"]
    K["Output Layer<br/>128 → 10 class scores"]

    A --> B --> C --> D
    D --> E --> F --> G
    G --> H --> I --> J --> K

    style A fill:#90CAF9,stroke:#1E88E5,color:#000
    style B fill:#CE93D8,stroke:#8E24AA,color:#000
    style C fill:#E1BEE7,stroke:#8E24AA,color:#000
    style D fill:#C8E6C9,stroke:#2E7D32,color:#000
    style E fill:#CE93D8,stroke:#8E24AA,color:#000
    style F fill:#E1BEE7,stroke:#8E24AA,color:#000
    style G fill:#C8E6C9,stroke:#2E7D32,color:#000
    style H fill:#FFE0B2,stroke:#EF6C00,color:#000
    style I fill:#CE93D8,stroke:#8E24AA,color:#000
    style J fill:#E1BEE7,stroke:#8E24AA,color:#000
    style K fill:#90CAF9,stroke:#1E88E5,color:#000


## 5. Implementation


flowchart TD
    A["GitHub Repository<br/>Python + PyTorch"]
    B["CIFAR-10 Dataset<br/>50,000 Train + 10,000 Test"]
    C["Implement CNN<br/>Unit Tests + Validation"]

    D["Local CPU Baseline<br/>1 Training Process"]
    E["Distributed CPU Training<br/>2 and 4 DDP Processes"]

    F["Google Colab<br/>Single GPU Training"]
    G["Optional AWS Cloud<br/>1, 2 and 4 GPU DDP"]

    H["CPU Scalability Analysis<br/>1 vs 2 vs 4 Processes"]
    I["CPU vs GPU Comparison"]
    J["Optional Multi-GPU<br/>Scalability Analysis"]

    K["Performance Metrics<br/>Throughput, Speedup, Efficiency<br/>Epoch Time, Communication, Accuracy"]

    L["Visualisations and Analysis<br/>Graphs, Bottlenecks, Deviations"]
    M["GitHub Documentation<br/>Assignment Report"]

    A --> B --> C --> D
    D --> E
    E --> H

    E --> F
    F --> I
    F -.-> G
    G --> J

    H --> K
    I --> K
    J --> K

    K --> L --> M

    style A fill:#90CAF9,stroke:#1E88E5,color:#000
    style B fill:#C8E6C9,stroke:#2E7D32,color:#000
    style C fill:#CE93D8,stroke:#8E24AA,color:#000
    style D fill:#90CAF9,stroke:#1E88E5,color:#000
    style E fill:#90CAF9,stroke:#1E88E5,color:#000
    style F fill:#C8E6C9,stroke:#2E7D32,color:#000
    style G fill:#FFE0B2,stroke:#EF6C00,color:#000
    style H fill:#CE93D8,stroke:#8E24AA,color:#000
    style I fill:#CE93D8,stroke:#8E24AA,color:#000
    style J fill:#CE93D8,stroke:#8E24AA,color:#000
    style K fill:#FFE0B2,stroke:#EF6C00,color:#000
    style L fill:#C8E6C9,stroke:#2E7D32,color:#000
    style M fill:#90CAF9,stroke:#1E88E5,color:#000


```text

cd /d C:\gitHub\aiml-mlsysops-parallel-cifar10
python -m venv .venv
venv\Scripts\activate
python -m venv .venv
venv\Scripts\activate
.venv\Scripts\activate.bat
where python
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

(.venv) C:\gitHub\aiml-mlsysops-parallel-cifar10>python -c "import torch; import torchvision; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
PyTorch: 2.14.1+cpu
Torchvision: 0.29.1+cpu
CUDA: False

C:\gitHub\aiml-mlsysops-parallel-cifar10\
│
├── README.md
├── pyproject.toml
├── .gitignore
├── LICENSE
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── configs/
│   ├── cpu_baseline.yaml
│   ├── cpu_ddp.yaml
│   └── gpu.yaml
│
├── src/
│   └── mlsysops_cifar10/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── distributed.py
│       ├── metrics.py
│       └── utils.py
│
├── experiments/
│   ├── benchmark.py
│   ├── profile.py
│   └── plot_results.py
│
├── notebooks/
│   └── colab_gpu.ipynb
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   ├── test_training.py
│   └── test_distributed.py
│
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── experiments.md
│   └── findings.md
│
├── results/
│   ├── README.md
│   ├── benchmarks/
│   └── figures/
│
├── report/
│   └── assignment-report.docx
│
├── data/                  # Git ignored
├── checkpoints/           # Git ignored
└── runs/                  # Git ignored

```

## 6. Technology Stack

## 7. Experimental Configurations

## 8. Performance Metrics

## 9. Results and Analysis

## 10. Installation and Usage

## 11. References