# Numerical Verification — Theorem IV.2 and Corollary IV.4

Self-contained exact numerical verification of spectral bounds on effective sharpness for Fisher-preconditioned optimization.

## Overview

This module provides a standalone numerical verification of Theorem IV.2 and Corollary IV.4 from:

> **"A Spectral Bound on Effective Sharpness for Fisher-Preconditioned Gradient Descent"**

### Theoretical Statement

Theorem IV.2 establishes that under Fisher preconditioning $(F + \gamma I)^{-1}$ with residual non-Gauss-Newton curvature $\varepsilon = \|H - G\|_2$:

$$S_{\text{eff}} \le 1 + \frac{\varepsilon}{\mu_{\min}(F + \gamma I)}$$

Under distribution shift or model misspecification where $G \ne F$, Corollary IV.4 provides the general bound:

$$S_{\text{eff}} \le 1 + \frac{\varepsilon + \delta}{\mu_{\min}(F + \gamma I)}$$

where $\delta = \|G - F\|_2$ and $\mu_{\min}(F + \gamma I) = \lambda_{\min}(F) + \gamma$.

## Verification Procedure

The script `verify_theorem_iv2.py` trains a 110-parameter deep linear network (DLN, depth 2, width 10) on a synthetic regression dataset ($N = 200$) with standard gradient descent and computes exact curvature quantities at every 5th iteration:

- **$H$:** Full loss Hessian ($110 \times 110$) via exact autograd Hessian assembly.
- **$G$:** Generalized Gauss-Newton matrix ($110 \times 110$).
- **$F$:** Empirical Fisher information matrix ($110 \times 110$) computed via exact per-sample outer products.
- **$\varepsilon, \delta$:** Operator 2-norms $\|H - G\|_2$ and $\|G - F\|_2$ computed via exact SVD / eigendecomposition.
- **$S_{\text{eff}}$:** Exact leading eigenvalue $\lambda_{\max}\left((F + \gamma I)^{-1} H\right)$.

The assertion $S_{\text{eff}} \le \text{Bound}$ is evaluated at every checkpoint. The script exits with status code 0 upon successful validation across all steps.

## Requirements

- Python $\ge$ 3.12
- PyTorch $\ge$ 2.0 (CPU execution is fully supported)
- NumPy $\ge$ 1.24

```bash
pip install -r requirements.txt
```

## Running the Verification

```bash
python verify_theorem_iv2.py
```

Expected runtime: approximately 60–90 seconds on a standard multi-core CPU.

## Dataset Utility

To download the MNIST and CIFAR-10 datasets used in broader benchmark evaluations:

```bash
python download_datasets.py
```

## Reproducibility Notes

- All hyperparameters strictly match the experimental configuration in the paper.
- Random seeds are fixed (`torch.manual_seed(42)` and `numpy.random.seed(42)`) to ensure deterministic reproduction across platforms.
