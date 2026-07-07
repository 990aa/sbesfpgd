"""
Stochastic Extension Experiments for "Spectral Stability of NGD"

On the 110-parameter DLN (CPU, exact computation):
1. Subsample mini-batches of size b in {25, 50, 100, 250, 500=full batch}.
2. At a fixed checkpoint, compute H_B, F_B exactly for many draws of B at each b.
3. Empirically measure ||H_B - H||_2 and ||F_B - F||_2 as a function of b.
4. Compute S_eff_B for each draw and compare to full-batch S_eff.
5. Verify the proposed stochastic bound:
   S_eff_B <= 1 + (eps + delta + xi_H + xi_F) / (mu_min(F) - xi_F)

Run via: uv run python scripts/stochastic_extension.py
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
import time

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "stochastic_extension_results.json"
)
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


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


def full_empirical_fisher(model, X, Y, params):
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


def per_sample_hessian(model, X, Y, params, idx):
    """Compute the per-sample Hessian for sample idx."""
    xi, yi = X[idx:idx+1], Y[idx:idx+1]
    pred = model(xi)
    loss_i = nn.MSELoss()(pred, yi)
    return full_hessian(loss_i, params)


def per_sample_fisher_outer(model, X, Y, params, idx):
    """Compute the per-sample Fisher outer product g_i g_i^T."""
    d = sum(p.numel() for p in params)
    xi, yi = X[idx:idx+1], Y[idx:idx+1]
    pred = model(xi)
    li = nn.MSELoss(reduction="none")(pred, yi).sum()
    gi = torch.autograd.grad(li, params, retain_graph=False)
    gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
    return np.outer(gvec, gvec)


def main():
    print("=" * 70)
    print("Stochastic Extension Experiment")
    print("=" * 70)

    # Setup: 110-parameter DLN
    depth, width, input_dim = 2, 10, 10
    N_total = 500  # full dataset
    damping = 1e-3
    train_steps = 50  # fixed checkpoint

    torch.manual_seed(42)
    np.random.seed(42)

    X = torch.randn(N_total, input_dim)
    teacher = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
    with torch.no_grad():
        Y = teacher(X)

    model = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
    params = list(model.parameters())
    d = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=0.1)

    # Train to checkpoint
    for step in range(train_steps):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred = model(X)
    loss = nn.MSELoss()(pred, Y)
    print(f"  Model: {d} parameters, checkpoint at step {train_steps}")
    print(f"  Loss at checkpoint: {loss.item():.6f}")

    # ── Full-batch exact quantities ──
    print("\n  Computing full-batch quantities...")
    t0 = time.time()
    H_full = full_hessian(loss, params)
    G_full = full_gauss_newton(model, X, params)
    F_full = full_empirical_fisher(model, X, Y, params)
    F_reg = F_full + damping * np.eye(d)

    eps_full = float(np.linalg.norm(H_full - G_full, ord=2))
    delta_full = float(np.linalg.norm(G_full - F_full, ord=2))
    mu_min_full = float(np.min(np.linalg.eigvalsh(F_reg)))
    F_inv_H = np.linalg.solve(F_reg, H_full)
    seff_full = float(np.max(np.abs(np.linalg.eigvals(F_inv_H))).real)

    bound_iv4 = 1.0 + (eps_full + delta_full) / mu_min_full

    print(f"  Full-batch: eps={eps_full:.4f}, delta={delta_full:.4f}, mu_min={mu_min_full:.6f}")
    print(f"  Full-batch: S_eff={seff_full:.1f}, Cor IV.4 bound={bound_iv4:.1f}")
    print(f"  Time: {time.time()-t0:.1f}s")

    # Precompute per-sample Hessians and Fisher outer products
    print("\n  Precomputing per-sample Hessians and Fisher outer products...")
    t0 = time.time()
    per_sample_H = []
    per_sample_F = []
    for i in range(N_total):
        if i % 100 == 0:
            print(f"    Sample {i}/{N_total}...", flush=True)
        # Per-sample Hessian
        xi, yi = X[i:i+1], Y[i:i+1]
        pred_i = model(xi)
        loss_i = nn.MSELoss()(pred_i, yi)
        Hi = full_hessian(loss_i, params)
        per_sample_H.append(Hi)

        # Per-sample Fisher outer product
        Fi = per_sample_fisher_outer(model, X, Y, params, i)
        per_sample_F.append(Fi)

    per_sample_H = np.array(per_sample_H)  # (N, d, d)
    per_sample_F = np.array(per_sample_F)  # (N, d, d)
    print(f"  Precomputation done in {time.time()-t0:.1f}s")

    # ── Mini-batch experiments ──
    batch_sizes = [25, 50, 100, 250, 500]
    n_draws = 50  # number of random mini-batches per batch size
    results = {}

    for b in batch_sizes:
        print(f"\n  Batch size b={b} ({n_draws} draws):")
        
        H_norms = []
        F_norms = []
        seff_batches = []
        bound_stochastic_list = []
        bound_holds_list = []
        
        for draw in range(n_draws):
            np.random.seed(draw * 1000 + b)
            indices = np.random.choice(N_total, size=b, replace=False)

            # Mini-batch Hessian = mean of per-sample Hessians over the batch
            H_B = np.mean(per_sample_H[indices], axis=0)
            # Mini-batch empirical Fisher = mean of per-sample outer products over the batch
            F_B = np.mean(per_sample_F[indices], axis=0)

            # Concentration: ||H_B - H||_2 and ||F_B - F||_2
            xi_H = float(np.linalg.norm(H_B - H_full, ord=2))
            xi_F = float(np.linalg.norm(F_B - F_full, ord=2))
            H_norms.append(xi_H)
            F_norms.append(xi_F)

            # Mini-batch S_eff
            F_B_reg = F_B + damping * np.eye(d)
            try:
                F_B_inv_H_B = np.linalg.solve(F_B_reg, H_B)
                seff_B = float(np.max(np.abs(np.linalg.eigvals(F_B_inv_H_B))).real)
            except np.linalg.LinAlgError:
                seff_B = float('inf')
            seff_batches.append(seff_B)

            # Proposed stochastic bound:
            # S_eff_B <= 1 + (eps + delta + xi_H + xi_F) / (mu_min(F) - xi_F)
            # provided xi_F < mu_min(F)
            if xi_F < mu_min_full:
                stoch_bound = 1.0 + (eps_full + delta_full + xi_H + xi_F) / (mu_min_full - xi_F)
                bound_holds = seff_B <= stoch_bound * (1 + 1e-6)
            else:
                stoch_bound = float('inf')
                bound_holds = True  # vacuously true (bound is infinity)

            bound_stochastic_list.append(stoch_bound)
            bound_holds_list.append(bound_holds)

        H_norms = np.array(H_norms)
        F_norms = np.array(F_norms)
        seff_batches = np.array(seff_batches)
        
        # Check 1/sqrt(b) scaling
        mean_xi_H = float(np.mean(H_norms))
        mean_xi_F = float(np.mean(F_norms))
        mean_seff = float(np.mean(seff_batches))
        std_seff = float(np.std(seff_batches))
        
        n_holds = sum(bound_holds_list)
        
        print(f"    ||H_B - H||_2: mean={mean_xi_H:.4f}, std={np.std(H_norms):.4f}")
        print(f"    ||F_B - F||_2: mean={mean_xi_F:.4f}, std={np.std(F_norms):.4f}")
        print(f"    S_eff_B:       mean={mean_seff:.1f}, std={std_seff:.1f} (full-batch: {seff_full:.1f})")
        print(f"    Stochastic bound holds: {n_holds}/{n_draws}")

        results[str(b)] = {
            "batch_size": b,
            "n_draws": n_draws,
            "xi_H_mean": mean_xi_H,
            "xi_H_std": float(np.std(H_norms)),
            "xi_H_max": float(np.max(H_norms)),
            "xi_F_mean": mean_xi_F,
            "xi_F_std": float(np.std(F_norms)),
            "xi_F_max": float(np.max(F_norms)),
            "seff_B_mean": mean_seff,
            "seff_B_std": std_seff,
            "seff_B_min": float(np.min(seff_batches)),
            "seff_B_max": float(np.max(seff_batches)),
            "seff_full": seff_full,
            "bound_holds_count": n_holds,
            "bound_holds_fraction": n_holds / n_draws,
            "xi_F_lt_mu_min_count": sum(1 for xi in F_norms if xi < mu_min_full),
        }

    # Check 1/sqrt(b) scaling
    print("\n  Concentration scaling check (should be ~1/sqrt(b)):")
    print(f"  {'b':>5}  {'mean ||H_B-H||':>14}  {'predicted':>10}  {'mean ||F_B-F||':>14}  {'predicted':>10}")
    b_ref = batch_sizes[-1]
    h_ref = results[str(b_ref)]["xi_H_mean"]
    f_ref = results[str(b_ref)]["xi_F_mean"]
    for b in batch_sizes:
        h_pred = h_ref * np.sqrt(b_ref / b) if b < b_ref else h_ref
        f_pred = f_ref * np.sqrt(b_ref / b) if b < b_ref else f_ref
        r = results[str(b)]
        print(f"  {b:>5d}  {r['xi_H_mean']:>14.4f}  {h_pred:>10.4f}  {r['xi_F_mean']:>14.4f}  {f_pred:>10.4f}")

    # Save
    output = {
        "full_batch": {
            "N": N_total,
            "d": d,
            "eps": eps_full,
            "delta": delta_full,
            "mu_min": mu_min_full,
            "seff": seff_full,
            "bound_iv4": bound_iv4,
        },
        "mini_batch_results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # ── Generate figure ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        bs = [r["batch_size"] for r in results.values()]
        xi_h = [r["xi_H_mean"] for r in results.values()]
        xi_f = [r["xi_F_mean"] for r in results.values()]
        xi_h_std = [r["xi_H_std"] for r in results.values()]
        xi_f_std = [r["xi_F_std"] for r in results.values()]

        # Panel (a): Concentration vs batch size
        ax = axes[0]
        ax.errorbar(bs, xi_h, yerr=xi_h_std, marker='o', label=r'$\|\hat{H}_B - H\|_2$')
        ax.errorbar(bs, xi_f, yerr=xi_f_std, marker='s', label=r'$\|\hat{F}_B - F\|_2$')
        # 1/sqrt(b) reference
        b_arr = np.array(bs)
        scale_h = xi_h[0] * np.sqrt(bs[0])
        scale_f = xi_f[0] * np.sqrt(bs[0])
        ax.plot(b_arr, scale_h / np.sqrt(b_arr), '--', color='gray', alpha=0.5, label=r'$\propto 1/\sqrt{b}$')
        ax.set_xlabel('Batch size $b$')
        ax.set_ylabel('Spectral norm')
        ax.set_title('(a) Concentration of mini-batch estimates')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(fontsize=7)

        # Panel (b): S_eff distribution
        ax = axes[1]
        seff_means = [r["seff_B_mean"] for r in results.values()]
        seff_stds = [r["seff_B_std"] for r in results.values()]
        ax.errorbar(bs, seff_means, yerr=seff_stds, marker='o', label=r'$\hat{S}_{\mathrm{eff},B}$ (mean $\pm$ s.d.)')
        ax.axhline(y=seff_full, color='r', linestyle='--', label=f'Full-batch $S_{{\\mathrm{{eff}}}}$ = {seff_full:.0f}')
        ax.set_xlabel('Batch size $b$')
        ax.set_ylabel(r'$\hat{S}_{\mathrm{eff},B}$')
        ax.set_title('(b) Mini-batch effective sharpness')
        ax.set_xscale('log')
        ax.legend(fontsize=7)

        # Panel (c): Bound satisfaction rate
        ax = axes[2]
        holds_frac = [r["bound_holds_fraction"] for r in results.values()]
        ax.bar(range(len(bs)), holds_frac, tick_label=[str(b) for b in bs])
        ax.set_xlabel('Batch size $b$')
        ax.set_ylabel('Fraction bound satisfied')
        ax.set_title('(c) Stochastic bound verification')
        ax.set_ylim([0, 1.1])

        plt.tight_layout()
        for ext in ("png", "pdf"):
            plt.savefig(os.path.join(FIGDIR, f"stochastic_extension.{ext}"), bbox_inches="tight", dpi=200)
        plt.close()
        print("Figure saved: stochastic_extension.png/pdf")
    except Exception as e:
        print(f"Warning: could not generate figure: {e}")


if __name__ == "__main__":
    main()
