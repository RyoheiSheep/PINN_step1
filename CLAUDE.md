# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Structure

This repository contains two main components:

1. **SOAP Optimizer** (`SOAP/`): PyTorch implementation of the SOAP optimizer from "SOAP: Improving and Stabilizing Shampoo using Adam"
2. **JAX-PI** (`jaxpi_pirate_copy/`): JAX implementation of physics-informed neural networks (PINNs) with advanced network architectures

## Development Commands

### JAX-PI (Physics-Informed Neural Networks)

**Installation:**
```bash
cd jaxpi_pirate_copy
pip install .
```

**Training a model:**
```bash
cd jaxpi_pirate_copy/examples/{example_name}
python3 main.py --config=configs/default.py
```

**Evaluation:**
```bash
python3 main.py --config.mode=eval
```

**Multi-GPU training:**
```bash
CUDA_VISIBLE_DEVICES=0,1 python3 main.py
```

**Adjusting batch size for memory constraints:**
```bash
python3 main.py --config.batch_size_per_device=64
```

### SOAP Optimizer

**Basic usage:**
```python
from soap import SOAP
optim = SOAP(lr=3e-3, betas=(.95, .95), weight_decay=.01, precondition_frequency=10)
```

## High-Level Architecture

### JAX-PI Framework
- **Entry Point**: Each example has a `main.py` that handles both training and evaluation modes
- **Configuration**: ML Collections configs in `configs/` directories define hyperparameters, architectures, and training settings
- **Core Components**:
  - `jaxpi/models.py`: TrainState class and architecture creation functions
  - `jaxpi/archs.py`: Neural network architectures (MLP, ModifiedMlp, etc.)
  - `jaxpi/samplers.py`: Data sampling strategies
  - `jaxpi/evaluator.py`: Model evaluation utilities

- **Example Structure**: Each physics problem (burgers, allen_cahn, stokes_cylinder, etc.) has its own directory with:
  - `main.py`: Training/evaluation script
  - `train.py` & `eval.py`: Training and evaluation logic
  - `configs/`: Configuration variants (default, sota, plain, etc.)

### SOAP Optimizer
- Single-file PyTorch optimizer implementation (`SOAP/soap.py`)
- Combines Adam with Shampoo preconditioning
- Supports multi-dimensional layers with dimension merging capabilities

## Key Configuration Patterns

- **Architecture configs**: Define network type (`arch_name`), layers, activation functions, Fourier embeddings, and reparameterization
- **Optimizer configs**: Learning rates, gradient accumulation, optimizer type
- **Wandb integration**: Built-in Weights & Biases logging with project/name/tag configuration
- **Precision**: Uses JAX's "highest" precision by default for reproducibility

## Requirements

### JAX-PI
- Python ≥3.8
- JAX/JAXlib with GPU support
- Dependencies: flax, optax, ml_collections, wandb, numpy, scipy, matplotlib

### SOAP
- PyTorch
- Works best with large batch sizes due to second-order nature