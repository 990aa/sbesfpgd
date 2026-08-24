# Spectral Bound on Effective Sharpness for Fisher-Preconditioned Gradient Descent

A rigorous theoretical and empirical framework for analyzing the spectral stability and effective sharpness of Natural Gradient Descent and second-order optimizers at the Edge of Stability.

## Overview

During gradient descent training of deep neural networks, the maximum eigenvalue of the loss Hessian $\lambda_{\max}(H)$ often rises until it reaches the Edge of Stability (EoS) threshold $2/\eta$, where $\eta$ is the learning rate. For preconditioned gradient methods—such as Natural Gradient Descent (NGD), K-FAC, and second-order approximations—the relevant dynamical quantity governing stability is the **effective sharpness**:

$$S_{\text{eff}} = \lambda_{\max}\left((F + \gamma I)^{-1} H\right)$$

where $H$ denotes the full loss Hessian, $F$ denotes the Fisher Information Matrix (or the Gauss-Newton matrix $G$), and $\gamma > 0$ represents the Tikhonov damping factor.

This codebase provides exact numerical verification, scalable matrix-free estimators, and comprehensive experimental suites to evaluate theoretical bounds on effective sharpness, investigate model misspecification, and benchmark second-order optimizers across architectures.

## Theoretical Foundations

### Spectral Sharpness Bounds

1. **Exact Specification (Theorem IV.2):**  
   When the Fisher matrix matches the Gauss-Newton curvature ($G = F$), the effective sharpness is bounded by:
   $$S_{\text{eff}} \le 1 + \frac{\varepsilon}{\mu_{\min}(F + \gamma I)}$$
   where $\varepsilon = \|H - G\|_2$ denotes the residual (non-Gauss-Newton) curvature and $\mu_{\min}(F + \gamma I)$ is the minimum eigenvalue of the regularized Fisher matrix.

2. **General Misspecification (Corollary IV.4):**  
   Under empirical distribution shifts or model misspecification where $G \neq F$:
   $$S_{\text{eff}} \le 1 + \frac{\varepsilon + \delta}{\mu_{\min}(F + \gamma I)}$$
   where $\delta = \|G - F\|_2$ quantifies the misspecification gap between the true generalized Gauss-Newton matrix and the empirical Fisher matrix.

3. **Stochastic and Mini-Batch Extensions:**  
   When curvature is estimated over a mini-batch $B \subset \{1, \dots, N\}$ of size $b$:
   $$S_{\text{eff}, B} \le 1 + \frac{\varepsilon + \delta + \xi_H + \xi_F}{\mu_{\min}(F) - \xi_F}$$
   where $\xi_H = \|H_B - H\|_2$ and $\xi_F = \|F_B - F\|_2$ capture finite-sample sampling fluctuations.

## Installation

### Prerequisites

- Python $\ge$ 3.12 (Python 3.14 compatible)
- PyTorch $\ge$ 2.0.0
- NVIDIA GPU with CUDA support (optional, required only for GPU benchmark suites)

### Environment Setup

Install dependencies and set up the virtual environment using `uv`:

```bash
# Synchronize environment and install dependencies
uv sync
```

Alternatively, install requirements via standard `pip`:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install torch torchvision numpy scipy matplotlib tqdm asdl pymupdf
```

## Dataset Preparation

The benchmark pipelines use synthetic Deep Linear Networks (DLN), MNIST, and CIFAR-10. Download and cache the vision datasets before running large-scale benchmarks:

```bash
python sbesfpgd-verify/download_datasets.py
```

Standard PyTorch `torchvision.datasets` handles automatic download if this step is omitted.

## Guided Execution Workflows

### 1. Minimal Self-Contained Theorem Verification

To run a fast, standalone verification of Theorem IV.2 and Corollary IV.4 without external curvature libraries:

```bash
python sbesfpgd-verify/verify_theorem_iv2.py
```

- **Mechanism:** Trains a 110-parameter Deep Linear Network ($d=110, N=200$) using full-batch gradient descent.
- **Evaluation:** At every 5 iterations, evaluates the exact $110 \times 110$ Hessian $H$, Gauss-Newton $G$, and Fisher $F$ matrices via explicit outer products and eigendecomposition.
- **Assertions:** Validates that $S_{\text{eff}} \le 1 + (\varepsilon + \delta)/\mu_{\min}(F + \gamma I)$ holds strictly at all checkpoints.

### 2. Comprehensive Figure and Statistical Reproduction

To reproduce all publication figures (Figures 1 through 8) and comprehensive statistical validation tables:

```bash
uv run python scripts/reproduce_eos.py
```

This suite executes:
- DLN trajectory analysis comparing exact Fisher NGD, diagonal Fisher NGD, K-FAC, and standard gradient descent.
- EoS dynamics and stability threshold tracking across varying learning rates $\eta$.
- Exact empirical vs. theoretical bound tracking across training epochs.
- MNIST non-linear MLP evaluations with statistical hypothesis testing (Wilcoxon signed-rank tests and Cohen's $d$ effect sizes).
- High-resolution vector outputs saved directly to the `figures/` directory.

### 3. Scalable Matrix-Free Curvature Estimation

For neural architectures with parameter counts where explicit $d \times d$ matrix materialization is computationally intractable, matrix-free operators compute effective sharpness:

```bash
uv run python scripts/matrix_free_experiments.py
```

- **Hessian-Vector Products (HVP):** Computes exact directional derivatives via Pearlmutter's algorithmic differentiation trick:
  $$\text{HVP}(v) = \nabla_\theta \left( \nabla_\theta \mathcal{L}^\top v \right)$$
- **Fisher/GGN-Vector Products:** Computes exact forward-backward Jacobian-vector products without forming the $d \times d$ matrix.
- **Conjugate Gradient (CG) Solver:** Solves the linear system $(F + \gamma I) x = v$ iteratively.
- **Lanczos Eigensolvers:** Employs SciPy `LinearOperator` and `eigsh` to extract the leading eigenvalue $S_{\text{eff}} = \lambda_{\max}\left((F + \gamma I)^{-1} H\right)$ on models with $> 50,000$ parameters (e.g., MNIST Tanh MLPs).

### 4. Optimizer Baseline Comparisons

To compare Fisher-preconditioned descent against contemporary first-order, adaptive, and second-order optimizers:

```bash
# Benchmark against AdamW, SGD with cosine decay, and K-FAC
uv run python scripts/optimizer_baselines.py

# Benchmark against AdaHessian (Hutchinson trace estimation)
uv run python scripts/adahessian_baselines.py

# Benchmark against Sophia (diagonal Hessian clipping)
uv run python scripts/sophia_baselines.py
```

These experiments benchmark convergence rates, effective sharpness control, gradient noise tolerance, and compute overhead across DLN regression and MNIST classification.

### 5. Stochastic Extension and Finite-Batch Analysis

To analyze the impact of stochastic gradient noise and mini-batch sampling on curvature bounds:

```bash
uv run python scripts/stochastic_extension.py
```

- Subsamples mini-batches $b \in \{25, 50, 100, 250, 500\}$.
- Measures empirical operator norm deviations $\|H_B - H\|_2$ and $\|F_B - F\|_2$.
- Empirically validates the stochastic bound $S_{\text{eff}, B} \le 1 + (\varepsilon + \delta + \xi_H + \xi_F)/(\mu_{\min}(F) - \xi_F)$.

### 6. Architectural Scaling and CPU/GPU Benchmarks

```bash
# Width and depth scaling regressions on CPU
uv run python scripts/cpu_experiments.py

# High-dimensional misspecification scaling on GPU
uv run python scripts/misspec_scale_gpu.py

# Deep vision models (ResNet-18 on CIFAR-10) with K-FAC and adaptive damping
uv run python scripts/cifar_baselines_gpu.py
uv run python scripts/cifar_1cycle_adaptive_damping_gpu.py
uv run python scripts/cifar_adahessian_sophia_gpu.py
```

## Damping and Hyperparameter Guidelines

The Tikhonov regularization factor $\gamma$ plays a pivotal role in bounding effective sharpness:

1. **Theoretical Bound Condition:** To ensure $S_{\text{eff}} \le S_{\max}$, set the damping factor according to:
   $$\gamma \ge \frac{\varepsilon + \delta}{S_{\max} - 1} - \lambda_{\min}(F)$$
2. **Ill-Conditioned Curvature:** In deep or wide networks where $\lambda_{\min}(F) \to 0$, $\gamma$ prevents spectral explosion and bounds the effective condition number $\kappa\left((F + \gamma I)^{-1} H\right)$.
3. **Adaptive Damping:** When running non-stationary optimization (e.g., 1cycle schedules), dynamically adjusting $\gamma$ relative to the trace or top eigenvalue preserves second-order curvature alignment without sacrificing convergence speed.

## Numerical Verification Metrics

| Metric | Mathematical Definition | Role |
| :--- | :--- | :--- |
| **$H$** | $\nabla^2 \mathcal{L}(\theta)$ | Full loss Hessian matrix ($d \times d$) |
| **$G$** | $\frac{1}{N} J^\top \nabla^2_{\hat{y}} \ell J$ | Generalized Gauss-Newton curvature ($d \times d$) |
| **$F$** | $\frac{1}{N} \sum_{i=1}^N g_i g_i^\top$ | Empirical Fisher Information Matrix ($d \times d$) |
| **$\varepsilon$** | $\|H - G\|_2$ | Residual non-linear curvature |
| **$\delta$** | $\|G - F\|_2$ | Curvature misspecification gap |
| **$S_{\text{eff}}$** | $\lambda_{\max}\left((F + \gamma I)^{-1} H\right)$ | Effective sharpness determining dynamical stability |
| **$\mu_{\min}$** | $\lambda_{\min}(F + \gamma I)$ | Regularized minimum spectral eigenvalue |

## Output Files and Artifacts

- **Figures (`figures/`):** Generates publication-ready vector PDFs and 300 DPI PNGs (`eos_comparison.pdf`, `theorem_verification.pdf`, `spectrum_comparison.pdf`, `damping_validation.pdf`, `matrix_free_results.png`, etc.).
- **Structured Results (`*.json`):** Logs complete numerical metrics across iterations, including `cpu_experiment_results.json`, `matrix_free_results.json`, `adahessian_baselines_results.json`, `sophia_baselines_results.json`, and `stochastic_extension_results.json`.

## Citation and References

If you utilize these theoretical bounds or experimental implementations, please reference:

```bibtex
@article{ahad2026spectral,
  title={A Spectral Bound on Effective Sharpness for Fisher-Preconditioned Gradient Descent},
  author={Abdul Ahad and Napa Lakshmi},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2026},
  url={https://openreview.net/forum?id=EabuvggEbb}
}
```