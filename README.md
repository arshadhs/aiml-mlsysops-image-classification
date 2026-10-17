# aiml-mlsysops-image-classification

Design, implement, and evaluate a parallelised or distributed machine learning system for training an image classification model on the CIFAR-10 dataset.

Model architecture, parallelism strategy, execution platform, and development framework —
and rigorously justify each decision.

CIFAR-10 consists of 60,000 colour images (32×32 pixels) across 10 classes (50,000 training / 10,000
test). It is a well-understood benchmark, allowing to focus effort on the systems and
optimisation aspects rather than dataset preparation.

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