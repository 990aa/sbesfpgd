"""
CPU experiments for DLN-based validation (Issues 4, 6, 7).
  - Scaling analysis with 5 seeds per width + regression
  - Alignment-aware bound validation (Theorem IV.3)
  - Damping rule-of-thumb validation

Run via: uv run python scripts/cpu_experiments.py
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cpu_experiment_results.json")


def _savefig(name: str) -> None:
    """Save current figure as PNG and PDF with tight bounding box."""
    import matplotlib.pyplot as _plt

    for ext in ("png", "pdf"):
        _plt.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    _plt.close()


# Model + utilities (self-contained)
class DeepLinearNet(nn.Module):
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
    """Compute exact full Gauss-Newton matrix for MSE loss."""
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


def full_fisher(model, X, Y, params, damping=0.0):
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
    if damping > 0:
        F += damping * np.eye(d)
    return F


def measure_all_quantities(width, depth=3, N=200, seed=42, damping=1e-3, train_steps=50):
    """Train DLN, then compute full H, F, Q and all bound quantities."""
    dim = width
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=depth, width=width, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=depth, width=width, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=0.1)
    for _ in range(train_steps):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    pred = model(X)
    loss = nn.MSELoss()(pred, Y)
    H_np = full_hessian(loss, params)
    G_np = full_gauss_newton(model, X, params)
    F_np = full_fisher(model, X, Y, params, damping=0.0)
    F_reg = F_np + damping * np.eye(npar)

    Q = H_np - G_np
    eps = np.linalg.norm(Q, ord=2)
    delta = np.linalg.norm(G_np - F_np, ord=2)

    mu_min = np.min(np.linalg.eigvalsh(F_reg))
    mu_max = np.max(np.linalg.eigvalsh(F_reg))

    F_inv_H = np.linalg.solve(F_reg, H_np)
    seff = np.max(np.abs(np.linalg.eigvals(F_inv_H))).real
    bound_iv2 = 1.0 + eps / max(mu_min, 1e-12)
    bound_iv4 = 1.0 + (eps + delta) / max(mu_min, 1e-12)

    return {
        "n_params": npar,
        "width": width,
        "seed": seed,
        "eps": float(eps),
        "delta": float(delta),
        "mu_min": float(mu_min),
        "mu_max": float(mu_max),
        "seff": float(seff),
        "bound_iv2": float(bound_iv2),
        "bound_iv4": float(bound_iv4),
        "ratio_iv2": float(eps / max(mu_min, 1e-12)),
        "ratio_iv4": float((eps + delta) / max(mu_min, 1e-12)),
        "loss": float(loss.item()),
        "kappa_F": float(mu_max / max(mu_min, 1e-12)),
        # Return raw matrices for alignment computation
        "_H": H_np,
        "_F": F_np,
        "_F_reg": F_reg,
        "_Q": tuple(
            [Q, H_np - F_np]
        ),  # passing both for backward compatibility with alignment computation which might still use H - F for its theorem verification or we can update it too. We will see.
    }


# EXPERIMENT A: Scaling with 5 seeds + regression (Issue 4)


def run_scaling_experiment():
    print("-" * 60)
    print("EXPERIMENT A: Scaling Analysis (5 seeds per width)")
    print("-" * 60)

    widths = [5, 10, 15, 20, 30, 40]
    n_seeds = 5
    results_by_width = {}

    for w in widths:
        print(f"\n  Width {w}:", end=" ", flush=True)
        seed_results = []
        for s in range(n_seeds):
            r = measure_all_quantities(w, depth=3, N=200, seed=s, damping=1e-3, train_steps=50)
            seed_results.append(
                {
                    "seed": s,
                    "n_params": r["n_params"],
                    "ratio_iv2": r["ratio_iv2"],
                    "ratio_iv4": r["ratio_iv4"],
                    "seff": r["seff"],
                    "bound_iv2": r["bound_iv2"],
                    "bound_iv4": r["bound_iv4"],
                    "eps": r["eps"],
                    "delta": r["delta"],
                    "mu_min": r["mu_min"],
                    "kappa_F": r["kappa_F"],
                }
            )
            print(f"s{s}", end=" ", flush=True)

        ratios_iv2 = [sr["ratio_iv2"] for sr in seed_results]
        ratios_iv4 = [sr["ratio_iv4"] for sr in seed_results]
        seffs = [sr["seff"] for sr in seed_results]
        results_by_width[str(w)] = {
            "n_params": seed_results[0]["n_params"],
            "ratio_iv2_mean": float(np.mean(ratios_iv2)),
            "ratio_iv2_std": float(np.std(ratios_iv2)),
            "ratio_iv4_mean": float(np.mean(ratios_iv4)),
            "ratio_iv4_std": float(np.std(ratios_iv4)),
            "seff_mean": float(np.mean(seffs)),
            "seff_std": float(np.std(seffs)),
            "bound_iv2_satisfied_all": all(sr["seff"] <= sr["bound_iv2"] + 0.01 for sr in seed_results),
            "bound_iv4_satisfied_all": all(sr["seff"] <= sr["bound_iv4"] + 0.01 for sr in seed_results),
            "per_seed": seed_results,
        }
        print(
            f"→ Cor IV.4 = {np.mean(ratios_iv4):.0f} ± {np.std(ratios_iv4):.0f}, "
            f"S_eff = {np.mean(seffs):.0f} ± {np.std(seffs):.0f}"
        )

    # Regression: ε_true+delta/μ_min vs log(params)
    log_params = [np.log(results_by_width[str(w)]["n_params"]) for w in widths]
    mean_ratios_iv4 = [results_by_width[str(w)]["ratio_iv4_mean"] for w in widths]
    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_params, mean_ratios_iv4)

    regression = {
        "slope": float(slope),
        "intercept": float(intercept),
        "R_squared": float(r_value**2),
        "p_value": float(p_value),
        "std_err": float(std_err),
    }
    print("\n  Regression (Cor IV.4 vs log(params)):")
    print(f"    Slope = {slope:.1f}, R² = {r_value**2:.4f}, p = {p_value:.4f}")
    print(f"    Interpretation: {'No significant trend' if p_value > 0.05 else 'Significant trend'}")

    # Print summary table
    print(
        f"\n  {'Width':>6} {'Params':>7} {'Thm IV.2':>16} {'Cor IV.4':>16} {'S_eff':>16} {'IV.2 OK':>9} {'IV.4 OK':>9}"
    )
    print(f"  {'-' * 80}")
    for w in widths:
        r = results_by_width[str(w)]
        print(
            f"  {w:>6} {r['n_params']:>7} "
            f"{r['ratio_iv2_mean']:>7.0f} ± {r['ratio_iv2_std']:>5.0f} "
            f"{r['ratio_iv4_mean']:>7.0f} ± {r['ratio_iv4_std']:>5.0f} "
            f"{r['seff_mean']:>7.0f} ± {r['seff_std']:>5.0f} "
            f"{'Yes' if r['bound_iv2_satisfied_all'] else 'NO':>9} "
            f"{'Yes' if r['bound_iv4_satisfied_all'] else 'NO':>9}"
        )

    return {"by_width": results_by_width, "regression": regression}


# EXPERIMENT B: Alignment-Aware Bound Validation (Issue 6)


def compute_alignment_bound(H_np, F_np, F_reg, Q, damping=1e-3):
    """Compute Theorem IV.3 alignment-aware bound."""
    d = H_np.shape[0]

    # Eigendecompose Q and F_reg
    Q_eigvals, Q_eigvecs = np.linalg.eigh(Q)  # Q = Σ λ_Q u u^T
    F_eigvals, F_eigvecs = np.linalg.eigh(F_reg)  # F = Σ μ v v^T

    # Alignment matrix A_ij = (u_i^T v_j)^2
    A = (Q_eigvecs.T @ F_eigvecs) ** 2  # shape (d, d)

    # Theorem IV.3 bound (first form): max_j Σ_i |λ_Q_i| A_ij / μ_j
    bound_per_j = np.zeros(d)
    for j in range(d):
        bound_per_j[j] = np.sum(np.abs(Q_eigvals) * A[:, j]) / max(F_eigvals[j], 1e-12)
    bound_iv3 = 1.0 + np.max(bound_per_j)

    # Theorem IV.2 bound (worst case)
    eps = np.linalg.norm(Q, ord=2)
    mu_min = np.min(F_eigvals)
    bound_iv2 = 1.0 + eps / max(mu_min, 1e-12)

    # Proposition IV.5 lower bound (if Q has positive eigenvalues)
    Q_pos = Q_eigvals[Q_eigvals > 0]
    if len(Q_pos) > 0:
        eps_min = np.min(Q_pos)
        mu_max = np.max(F_eigvals)
        lower_bound = 1.0 + eps_min / mu_max
    else:
        lower_bound = 1.0

    # Actual S_eff
    F_inv_H = np.linalg.solve(F_reg, H_np)
    seff = np.max(np.abs(np.linalg.eigvals(F_inv_H))).real

    # Tightness improvement
    improvement = bound_iv2 / max(bound_iv3, 1e-12)

    return {
        "seff": float(seff),
        "bound_iv2": float(bound_iv2),
        "bound_iv3": float(bound_iv3),
        "lower_bound_iv5": float(lower_bound),
        "improvement_factor": float(improvement),
        "eps": float(eps),
        "mu_min": float(mu_min),
        "kappa_F": float(np.max(F_eigvals) / max(mu_min, 1e-12)),
        "Q_rank_eff": int(np.sum(np.abs(Q_eigvals) > 1e-6 * np.max(np.abs(Q_eigvals)))),
        "top_Q_eigvals": [float(v) for v in sorted(np.abs(Q_eigvals), reverse=True)[:10]],
        "top_F_eigvals": [float(v) for v in sorted(F_eigvals, reverse=True)[:10]],
        "alignment_max_per_j": [float(v) for v in sorted(bound_per_j, reverse=True)[:10]],
    }


def run_alignment_experiment():
    print("\n" + "-" * 60)
    print("EXPERIMENT B: Alignment-Aware Bound Validation (Theorem IV.3)")
    print("-" * 60)

    # Test on multiple widths and seeds
    configs = [
        {"width": 10, "depth": 2, "label": "110-param (depth 2, width 10)"},
        {"width": 20, "depth": 3, "label": "820-param (depth 3, width 20)"},
        {"width": 30, "depth": 3, "label": "1830-param (depth 3, width 30)"},
    ]

    alignment_results = {}
    for cfg in configs:
        print(f"\n  {cfg['label']}:")
        seed_results = []
        for s in range(5):
            raw = measure_all_quantities(cfg["width"], cfg["depth"], N=200, seed=s, damping=1e-3)
            # Unpack the tuple backward compatibility from _Q
            H_min_G, H_min_F = raw["_Q"]
            alg = compute_alignment_bound(raw["_H"], raw["_F"], raw["_F_reg"], H_min_G)
            seed_results.append(alg)

        # Average
        keys = ["seff", "bound_iv2", "bound_iv3", "lower_bound_iv5", "improvement_factor", "kappa_F"]
        avg = {k: float(np.mean([r[k] for r in seed_results])) for k in keys}
        std = {k: float(np.std([r[k] for r in seed_results])) for k in keys}

        alignment_results[cfg["label"]] = {
            "mean": avg,
            "std": std,
            "per_seed": seed_results,
        }

        print(f"    Actual S_eff:       {avg['seff']:.1f} ± {std['seff']:.1f}")
        print(f"    Bound IV.2 (worst): {avg['bound_iv2']:.1f} ± {std['bound_iv2']:.1f}")
        print(f"    Bound IV.3 (align): {avg['bound_iv3']:.1f} ± {std['bound_iv3']:.1f}")
        print(f"    Lower bound IV.5:   {avg['lower_bound_iv5']:.4f} ± {std['lower_bound_iv5']:.4f}")
        print(f"    Improvement factor: {avg['improvement_factor']:.2f}×")
        print(f"    κ(F):               {avg['kappa_F']:.0f}")

    return alignment_results


# EXPERIMENT C: Damping Rule-of-Thumb Validation (Issue 7)


def run_damping_experiment():
    print("\n" + "-" * 60)
    print("EXPERIMENT C: Damping Rule-of-Thumb Validation")
    print("-" * 60)

    # First, compute μ_median(F) for the DLN task
    print("\n  Computing μ_median(F) for DLN (width 20)...")
    raw = measure_all_quantities(20, depth=3, N=500, seed=42, damping=0.0, train_steps=0)
    F_eigvals = np.linalg.eigvalsh(raw["_F"])
    mu_median = float(np.median(F_eigvals[F_eigvals > 1e-10]))
    recommended_gamma = 0.1 * mu_median
    print(f"    μ_median(F) = {mu_median:.6f}")
    print(f"    Recommended γ ≈ 0.1 · μ_median = {recommended_gamma:.6f}")

    # Sweep damping values
    gammas = np.logspace(-5, 0, 20)
    n_seeds = 5
    steps = 150
    lr = 0.1
    N = 500

    damping_results = {}
    print(f"\n  Sweeping {len(gammas)} γ values × {n_seeds} seeds...")
    for gamma in gammas:
        finals = []
        for s in range(n_seeds):
            torch.manual_seed(s)
            np.random.seed(s)
            X = torch.randn(N, 20)
            teacher = DeepLinearNet(depth=3, width=20, input_dim=20)
            with torch.no_grad():
                Y = teacher(X)
            model = DeepLinearNet(depth=3, width=20, input_dim=20)
            params = list(model.parameters())
            for _ in range(steps):
                pred = model(X)
                loss = nn.MSELoss()(pred, Y)
                gs = torch.autograd.grad(loss, params)
                gf = torch.cat([g.view(-1) for g in gs])
                fi = 1.0 / (torch.dot(gf, gf) + gamma)
                with torch.no_grad():
                    for p, g in zip(params, gs):
                        p.sub_(g * fi * lr)
            with torch.no_grad():
                final_loss = nn.MSELoss()(model(X), Y).item()
            finals.append(final_loss)

        damping_results[f"{gamma:.6f}"] = {
            "gamma": float(gamma),
            "mean_loss": float(np.mean(finals)),
            "std_loss": float(np.std(finals)),
            "median_loss": float(np.median(finals)),
        }

    # Find best and recommended
    best_gamma = min(damping_results.values(), key=lambda x: x["mean_loss"])
    print(f"\n  Best γ (lowest mean MSE): {best_gamma['gamma']:.6f} → MSE = {best_gamma['mean_loss']:.6f}")
    print(f"  Recommended γ (0.1·μ_median): {recommended_gamma:.6f}")

    # Print table
    print(f"\n  {'γ':>12} {'Mean MSE':>12} {'Std':>10} {'Note':>20}")
    print(f"  {'-' * 60}")
    for key in sorted(damping_results.keys(), key=lambda k: damping_results[k]["gamma"]):
        r = damping_results[key]
        note = ""
        if abs(r["gamma"] - recommended_gamma) / max(recommended_gamma, 1e-10) < 2:
            note = "← near recommended"
        if r["gamma"] == best_gamma["gamma"]:
            note = "← BEST"
        print(f"  {r['gamma']:>12.6f} {r['mean_loss']:>12.6f} {r['std_loss']:>10.6f} {note:>20}")

    return {
        "mu_median_F": mu_median,
        "recommended_gamma": recommended_gamma,
        "best_gamma": best_gamma["gamma"],
        "best_gamma_loss": best_gamma["mean_loss"],
        "sweep": damping_results,
    }


# MAIN


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "text.usetex": False,
            "font.family": "serif",
            "lines.linewidth": 1.5,
            "savefig.dpi": 300,
        }
    )

    results = {}
    t0 = time.time()

    # ── Experiment A: Scaling ──
    results["scaling"] = run_scaling_experiment()

    # ── Experiment B: Alignment ──
    results["alignment"] = run_alignment_experiment()

    # ── Experiment C: Damping ──
    results["damping"] = run_damping_experiment()

    # ── Generate figures ──
    print("\n" + "-" * 60)
    print("Generating figures...")
    print("-" * 60)

    # Fig: Scaling analysis (updated with error bars)
    scaling = results["scaling"]
    widths = [5, 10, 15, 20, 30, 40]
    nparams = [scaling["by_width"][str(w)]["n_params"] for w in widths]
    ratio_iv2_means = [scaling["by_width"][str(w)]["ratio_iv2_mean"] for w in widths]
    ratio_iv2_stds = [scaling["by_width"][str(w)]["ratio_iv2_std"] for w in widths]
    ratio_iv4_means = [scaling["by_width"][str(w)]["ratio_iv4_mean"] for w in widths]
    ratio_iv4_stds = [scaling["by_width"][str(w)]["ratio_iv4_std"] for w in widths]
    seff_means = [scaling["by_width"][str(w)]["seff_mean"] for w in widths]
    seff_stds = [scaling["by_width"][str(w)]["seff_std"] for w in widths]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
    ax1.errorbar(
        nparams,
        ratio_iv2_means,
        yerr=ratio_iv2_stds,
        fmt="s-",
        color="tab:red",
        capsize=5,
        markersize=6,
        alpha=0.6,
        label=r"$\epsilon_{\mathrm{true}} / \mu_{\min}(F)$ (IV.2)",
    )
    ax1.errorbar(
        nparams,
        ratio_iv4_means,
        yerr=ratio_iv4_stds,
        fmt="o-",
        color="tab:orange",
        capsize=5,
        markersize=6,
        label=r"$(\epsilon_{\mathrm{true}}+\delta) / \mu_{\min}(F)$ (IV.4)",
    )
    # Regression line
    reg = scaling["regression"]
    log_x = np.linspace(np.log(min(nparams)), np.log(max(nparams)), 50)
    ax1.plot(
        np.exp(log_x),
        reg["slope"] * log_x + reg["intercept"],
        "--",
        color="gray",
        alpha=0.5,
        label=f"Regr. IV.4 (R\u00b2={reg['R_squared']:.3f})",
    )
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel("Bound Ratio")
    ax1.set_title("(a) Bound Ratio vs. Model Size (5 seeds)")
    ax1.legend(fontsize=8)
    ax1.grid(True, which="both", alpha=0.2)

    ax2.errorbar(
        nparams,
        seff_means,
        yerr=seff_stds,
        fmt="d-",
        color="tab:blue",
        capsize=5,
        markersize=8,
        label=r"$S_{\mathrm{eff}}$ (actual)",
    )
    ax2.errorbar(
        nparams,
        [r + 1 for r in ratio_iv4_means],
        yerr=ratio_iv4_stds,
        fmt="o-",
        color="tab:orange",
        capsize=5,
        markersize=7,
        alpha=0.7,
        label=r"Cor. IV.4 Bound",
    )
    ax2.errorbar(
        nparams,
        [r + 1 for r in ratio_iv2_means],
        yerr=ratio_iv2_stds,
        fmt="s-",
        color="tab:red",
        capsize=5,
        markersize=7,
        alpha=0.7,
        label=r"Thm. IV.2 Bound",
    )
    ax2.errorbar(
        nparams,
        bounds,
        yerr=ratio_stds,
        fmt="s--",
        color="tab:orange",
        capsize=5,
        markersize=8,
        label=r"Bound $1 + \epsilon/\mu_{\min}$",
    )
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of Parameters")
    ax2.set_ylabel("Effective Sharpness")
    ax2.set_title("(b) Actual vs. Bound (5 seeds, mean ± s.d.)")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    _savefig("scaling_analysis")
    print("  Saved scaling_analysis.png/.pdf")

    # Fig: Alignment validation (3-panel: per-model comparison)
    alignment = results["alignment"]
    labels = list(alignment.keys())
    seffs = [alignment[l]["mean"]["seff"] for l in labels]
    b_iv2 = [alignment[l]["mean"]["bound_iv2"] for l in labels]
    b_iv3 = [alignment[l]["mean"]["bound_iv3"] for l in labels]
    b_iv5 = [alignment[l]["mean"]["lower_bound_iv5"] for l in labels]

    fig, ax = plt.subplots(figsize=(7.16, 3.0))
    x = np.arange(len(labels))
    w_ = 0.2
    ax.bar(x - 1.5 * w_, seffs, w_, label=r"$S_{\mathrm{eff}}$ (actual)", color="tab:blue")
    ax.bar(x - 0.5 * w_, b_iv3, w_, label="Thm IV.3 (alignment)", color="tab:green")
    ax.bar(x + 0.5 * w_, b_iv2, w_, label="Thm IV.2 (worst-case)", color="tab:orange")
    ax.bar(x + 1.5 * w_, b_iv5, w_, label="Prop IV.5 (lower)", color="tab:purple")
    ax.set_xticks(x)
    ax.set_xticklabels([l.split("(")[0].strip() for l in labels], fontsize=11)
    ax.set_ylabel("Value")
    ax.set_title("Theorem IV.3 Alignment-Aware Bound Validation")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    ax.set_yscale("log")
    plt.tight_layout()
    _savefig("alignment_validation")
    print("  Saved alignment_validation.png/.pdf")

    # Fig: Damping validation
    damp = results["damping"]
    gammas_sorted = sorted(damp["sweep"].values(), key=lambda x: x["gamma"])
    gs = [d["gamma"] for d in gammas_sorted]
    means = [d["mean_loss"] for d in gammas_sorted]
    stds = [d["std_loss"] for d in gammas_sorted]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ax.errorbar(gs, means, yerr=stds, fmt="o-", color="tab:blue", capsize=3, markersize=6, label="SP-GD final MSE")
    ax.axvline(
        damp["recommended_gamma"],
        color="tab:red",
        ls="--",
        alpha=0.7,
        label=f"Recommended $\\gamma = 0.1 \\cdot \\mu_{{median}}$\n= {damp['recommended_gamma']:.2e}",
    )
    ax.axvline(
        damp["best_gamma"], color="tab:green", ls=":", alpha=0.7, label=f"Best $\\gamma$ = {damp['best_gamma']:.2e}"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Damping $\gamma$")
    ax.set_ylabel("Final MSE")
    ax.set_title("Damping Rule-of-Thumb Validation (DLN, 820 params)")
    ax.legend(fontsize=10)
    ax.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    _savefig("damping_validation")
    print("  Saved damping_validation.png/.pdf")

    # ── Save results ──
    # Strip numpy arrays for JSON serialization
    def strip_arrays(obj):
        if isinstance(obj, dict):
            return {k: strip_arrays(v) for k, v in obj.items() if not k.startswith("_")}
        elif isinstance(obj, list):
            return [strip_arrays(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        return obj

    results_clean = strip_arrays(results)
    results_clean["total_runtime_seconds"] = time.time() - t0
    with open(RESULTS_FILE, "w") as f:
        json.dump(results_clean, f, indent=2)

    print(f"\nResults saved to {RESULTS_FILE}")
    print(f"Total runtime: {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
