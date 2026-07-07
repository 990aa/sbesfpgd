"""
Compute relative misspecification delta/||F||_2 for the 110-parameter DLN.
Uses the same setup as verify_theorem_iv2.py to compute delta, mu_max(F), and
the relative misspecification ratio at iterations 0, 50, 100.

Run via: uv run python scripts/compute_relative_misspec.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "relative_misspec_results.json")


class DeepLinearNet(nn.Module):
    def __init__(self, depth=2, width=10, input_dim=10, output_dim=1):
        super().__init__()
        layers = []
        for i in range(depth):
            ind = input_dim if i == 0 else width
            outd = width if i < depth - 1 else output_dim
            layers.append(nn.Linear(ind, outd, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def full_hessian(loss, params):
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat = torch.cat([g.view(-1) for g in grads])
    n = flat.shape[0]
    H = torch.zeros((n, n))
    for i in range(n):
        row = torch.autograd.grad(flat[i], params, retain_graph=True)
        H[i] = torch.cat([r.detach().view(-1) for r in row])
    return H.numpy()


def full_gauss_newton(model, X, params):
    d = sum(p.numel() for p in params)
    G = np.zeros((d, d))
    N_local = X.shape[0]
    for i in range(N_local):
        pred = model(X[i : i + 1])
        gi = torch.autograd.grad(pred, params, retain_graph=True)
        gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
        G += np.outer(gvec, gvec)
    G *= 2.0 / N_local
    return G


def full_fisher(model, X, Y, params):
    N = X.shape[0]
    d = sum(p.numel() for p in params)
    F = np.zeros((d, d))
    criterion = nn.MSELoss(reduction="none")
    for i in range(N):
        pred = model(X[i : i + 1])
        li = criterion(pred, Y[i : i + 1]).sum()
        gi = torch.autograd.grad(li, params, retain_graph=False)
        gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
        F += np.outer(gvec, gvec)
    F /= N
    return F


def main():
    # Setup: 110-parameter DLN (depth=2, width=10, input_dim=10)
    # Same setup as verify_theorem_iv2.py
    depth = 2
    width = 10
    input_dim = 10
    N = 100
    damping = 1e-3

    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randn(N, input_dim)
    teacher = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
    with torch.no_grad():
        Y = teacher(X)

    model = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    print(f"Model: {npar} parameters, depth={depth}, width={width}")

    opt = torch.optim.SGD(params, lr=0.1)
    results = []

    for it in [0, 50, 100]:
        # Train to iteration 'it'
        if it > 0:
            target_iters = it - (results[-1]["iteration"] if results else 0)
            for _ in range(target_iters):
                pred = model(X)
                loss = nn.MSELoss()(pred, Y)
                opt.zero_grad()
                loss.backward()
                opt.step()

        # Compute matrices
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        H_np = full_hessian(loss, params)
        G_np = full_gauss_newton(model, X, params)
        F_np = full_fisher(model, X, Y, params)
        F_reg = F_np + damping * np.eye(npar)

        Q = H_np - G_np
        eps = float(np.linalg.norm(Q, ord=2))
        delta = float(np.linalg.norm(G_np - F_np, ord=2))

        F_eigvals = np.linalg.eigvalsh(F_np)
        F_reg_eigvals = np.linalg.eigvalsh(F_reg)

        mu_min_F = float(np.min(F_reg_eigvals))
        mu_max_F = float(np.max(F_eigvals))  # ||F||_2 = mu_max(F)
        F_spectral_norm = mu_max_F

        # Relative misspecification
        delta_relative = delta / max(F_spectral_norm, 1e-12)

        # S_eff and bounds
        F_inv_H = np.linalg.solve(F_reg, H_np)
        seff = float(np.max(np.abs(np.linalg.eigvals(F_inv_H))).real)
        bound_iv2 = 1.0 + eps / max(mu_min_F, 1e-12)
        bound_iv4 = 1.0 + (eps + delta) / max(mu_min_F, 1e-12)

        entry = {
            "iteration": it,
            "loss": float(loss.item()),
            "n_params": npar,
            "eps": float(eps),
            "delta": float(delta),
            "mu_min_F_damped": mu_min_F,
            "mu_max_F": mu_max_F,
            "F_spectral_norm": F_spectral_norm,
            "delta_relative": float(delta_relative),
            "seff": seff,
            "bound_iv2": bound_iv2,
            "bound_iv4": bound_iv4,
            "iv2_holds": seff <= bound_iv2 + 0.01,
            "iv4_holds": seff <= bound_iv4 + 0.01,
        }
        results.append(entry)

        print(f"\n  Iteration {it}:")
        print(f"    Loss:                     {loss.item():.6f}")
        print(f"    eps = ||Q||_2:             {eps:.6f}")
        print(f"    delta = ||G - F||_2:       {delta:.6f}")
        print(f"    ||F||_2 = mu_max(F):       {F_spectral_norm:.6f}")
        print(f"    delta/||F||_2 (relative):  {delta_relative:.6f}")
        print(f"    mu_min(F + gamma*I):       {mu_min_F:.6f}")
        print(f"    S_eff:                     {seff:.2f}")
        print(f"    Thm IV.2 bound:            {bound_iv2:.2f} ({'OK' if entry['iv2_holds'] else 'FAIL'})")
        print(f"    Cor IV.4 bound:            {bound_iv4:.2f} ({'OK' if entry['iv4_holds'] else 'FAIL'})")

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
