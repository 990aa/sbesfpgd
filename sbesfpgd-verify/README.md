# Verification Code — Theorem IV.2 & Experimental Results

Minimal, self-contained numerical verification of **Theorem IV.2** (and
Corollary IV.4) from:

> **"A Spectral Bound on Effective Sharpness for Fisher-Preconditioned Gradient Descent"**  

## What this verifies

Theorem IV.2 states that the effective sharpness of a Fisher-preconditioned
gradient step satisfies

$$S_{\text{eff}} \;\leq\; 1 + \frac{\varepsilon}{\mu_{\min}(F + \gamma I)}$$

The script `verify_theorem_iv2.py` trains a **110-parameter deep linear
network** (DLN, depth 2, width 10) with SGD and measures all quantities
*exactly* (no approximation) at every 5th iteration across 100 training steps.

The bound `S_eff ≤ bound` must be satisfied at **every** measured iteration.
The script exits with code 1 and prints a violation message if it is not.

Note: `S_eff` can exceed 1 because the preconditioner is the damped Fisher
$(F + \gamma I)^{-1}$ rather than the exact inverse $F^{-1}$.  The bound
accounts for this; see the paper for details.

## Requirements

- Python 3.12+
- PyTorch ≥ 2.0 (CPU-only is fine)
- NumPy ≥ 1.24

```
pip install -r requirements.txt
```

## Running

```
python verify_theorem_iv2.py
```

Runtime: approximately 60–90 s on a modern CPU (the bottleneck is the
O(d²·N) exact Fisher computation at each checkpoint; d = 110, N = 200).

## Repository Contents

This contains the verification code and the experimental results
used to produce the paper figures and tables. It also includes a small helper script to download the datasets used in the experiments (MNIST and CIFAR-10).


## Notes

- All hyper-parameters match the paper exactly: see the constants block near
  the top of `verify_theorem_iv2.py`.
- The script uses `torch.manual_seed(42)` and `numpy.random.seed(42)` to
  ensure reproducibility.  Minor floating-point differences across PyTorch
  versions (≤ 1 %) are expected and do not affect the pass/fail outcome.
