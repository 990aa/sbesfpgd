"""
verify_theorem_iv2.py

Self-contained numerical verification of Theorem IV.2 from:

  "A Spectral Bound on Effective Sharpness for Fisher-Preconditioned
   Gradient Descent" (IEEE submission)

The script trains a 110-parameter deep linear network (DLN) with SGD and, at
every 5th iteration, computes the following quantities *exactly* (no
approximations):

  H       -- full Hessian  (d x d)
  F       -- full Fisher   (d x d, exact per-sample outer products)
  Q       -- residual curvature  Q = H - F  (equals H - G for a DLN where G≈F)
  ε       -- spectral-norm  ||Q||_2
  μ_min   -- smallest eigenvalue of  F + γI   (γ = 1e-3)
  S_eff   -- actual effective sharpness  λ_max(F_reg^{-1} H)
  bound   -- Theorem IV.2 bound  1 + ε / μ_min

EXPECTED OUTPUT (matching paper Section VII.D):
  iter   0: S_eff ≈  902,  ε ≈ 1.55,  bound ≈ 1552
  iter  50: S_eff ≈  258,  ε ≈ 1.79,  bound ≈ 1792
  iter 100: S_eff ≈ 1080,  ε ≈ 1.82,  bound ≈ 1823

The assertion  S_eff < bound  must hold at EVERY measured iteration.

Requirements: torch, numpy  (no scipy needed)
Usage:
  pip install torch numpy
  python verify_theorem_iv2.py
"""

import sys

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Hyper-parameters (must match paper Section VII.D exactly)
# ---------------------------------------------------------------------------
DEPTH = 2
WIDTH = 10
INPUT_DIM = 10
OUTPUT_DIM = 1
N = 200          # training samples
SEED = 42
LR = 0.1         # SGD learning rate
STEPS = 105      # run slightly past 100 so iter-100 is included
GAMMA = 1e-3     # damping  (γ in paper)
EVERY = 5        # measure every EVERY-th step
HIGHLIGHT = {0, 50, 100}   # iterations emphasised in paper Table


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DeepLinearNet(nn.Module):
    """Depth-D linear network with no bias (matches reproduce_eos.py)."""

    def __init__(self, depth=3, width=20, input_dim=20, output_dim=1):
        super().__init__()
        layers = []
        for i in range(depth):
            ind = input_dim if i == 0 else width
            outd = width if i < depth - 1 else output_dim
            layers.append(nn.Linear(ind, outd, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Exact Hessian (autograd, small models only)
# ---------------------------------------------------------------------------
def full_hessian(loss, params):
    """Return the full Hessian matrix as a numpy array."""
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.view(-1) for g in grads])
    n = flat.shape[0]
    H = torch.zeros((n, n))
    for i in range(n):
        row = torch.autograd.grad(flat[i], params, retain_graph=True)
        H[i] = torch.cat([r.detach().view(-1) for r in row])
    return H.numpy()


# ---------------------------------------------------------------------------
# Exact Fisher  F = (1/N) Σ g_i g_i^T  (per-sample outer products)
# ---------------------------------------------------------------------------
def full_fisher(model, X, Y, params, damping=0.0):
    """Compute full Fisher matrix for MSE loss."""
    N_local = X.shape[0]
    d = sum(p.numel() for p in params)
    F = np.zeros((d, d))
    criterion = nn.MSELoss(reduction="none")
    for i in range(N_local):
        xi, yi = X[i : i + 1], Y[i : i + 1]
        pred = model(xi)
        li = criterion(pred, yi).sum()
        gi = torch.autograd.grad(li, params, retain_graph=False)
        gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
        F += np.outer(gvec, gvec)
    F /= N_local
    if damping > 0:
        F += damping * np.eye(d)
    return F


# ---------------------------------------------------------------------------
# Main verification loop
# ---------------------------------------------------------------------------
def verify():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # --- Data: teacher-student ---
    X = torch.randn(N, INPUT_DIM)
    teacher = DeepLinearNet(depth=DEPTH, width=WIDTH, input_dim=INPUT_DIM,
                            output_dim=OUTPUT_DIM)
    with torch.no_grad():
        Y = teacher(X)

    # --- Student model ---
    model = DeepLinearNet(depth=DEPTH, width=WIDTH, input_dim=INPUT_DIM,
                          output_dim=OUTPUT_DIM)
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=LR)

    print(f"Model: DLN depth={DEPTH}, width={WIDTH}, input={INPUT_DIM} → {n_params} parameters")
    print(f"Data:  N={N}, seed={SEED}, LR={LR}, γ={GAMMA}")
    print()

    # --- Header ---
    col = "{:>5}  {:>9}  {:>9}  {:>9}  {:>10}  {:>5}"
    print(col.format("iter", "ε=‖Q‖₂", "μ_min(F+γI)", "S_eff", "bound", "ok?"))
    print("-" * 60)

    records = []
    violations = 0

    for step in range(STEPS):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)

        if step % EVERY == 0:
            H_np = full_hessian(loss, params)
            F_np = full_fisher(model, X, Y, params, damping=0.0)
            F_reg = F_np + GAMMA * np.eye(n_params)

            Q = H_np - F_np
            eps = float(np.linalg.norm(Q, ord=2))
            mu_min = float(np.min(np.linalg.eigvalsh(F_reg)))
            F_inv_H = np.linalg.solve(F_reg, H_np)
            seff = float(np.max(np.abs(np.linalg.eigvals(F_inv_H))).real)
            bound = 1.0 + eps / max(mu_min, 1e-12)
            ok = seff <= bound * (1 + 1e-6)   # tiny tolerance for floating-point

            if not ok:
                violations += 1

            records.append(dict(step=step, eps=eps, mu_min=mu_min,
                                seff=seff, bound=bound, ok=ok))

            marker = "✓" if ok else "✗ VIOLATION"
            highlight = " ◄" if step in HIGHLIGHT else ""
            print(col.format(step, f"{eps:.4f}", f"{mu_min:.6f}",
                             f"{seff:.1f}", f"{bound:.1f}", marker) + highlight)

        opt.zero_grad()
        loss.backward()
        opt.step()

    # --- Summary ---
    print("-" * 60)
    print(f"\nTotal measured steps : {len(records)}")
    print(f"Bound satisfied      : {len(records) - violations}/{len(records)}")

    if violations:
        print(f"\n*** {violations} VIOLATION(S) DETECTED — theorem NOT verified ***")
        sys.exit(1)
    else:
        print("\nAll checks passed — Theorem IV.2 bound S_eff ≤ 1 + ε/μ_min(F+γI)")
        print("is satisfied at every measured iteration.")

    # --- Paper-specific checkpoints (Section VII.D) ---
    print()
    print("Paper-highlight checkpoints (Section VII.D)")
    print("{:>8}  {:>9}  {:>9}  {:>10}".format("iter", "S_eff", "ε", "bound"))
    for r in records:
        if r["step"] in HIGHLIGHT:
            print("{:>8d}  {:>9.1f}  {:>9.4f}  {:>10.1f}".format(
                r["step"], r["seff"], r["eps"], r["bound"]))

    print()
    print("Expected (from paper, Section VII.D):")
    print("  iter   0:  S_eff ≈  902,  ε ≈ 1.55,  bound ≈ 1552")
    print("  iter  50:  S_eff ≈  258,  ε ≈ 1.79,  bound ≈ 1792")
    print("  iter 100:  S_eff ≈ 1080,  ε ≈ 1.82,  bound ≈ 1823")


if __name__ == "__main__":
    verify()
