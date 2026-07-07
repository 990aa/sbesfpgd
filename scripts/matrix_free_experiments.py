"""
Matrix-Free Estimation Experiments for "Spectral Stability of NGD"

Implements:
  1. GGN-vector product (= true Fisher-vector product for MSE/CE losses)
  2. Empirical Fisher-vector product
  3. Hessian-vector product (Pearlmutter's trick)
  4. CG solve for (F + gamma I)^{-1} u
  5. Power iteration for spectral norms ||H-G||, ||G-F_emp||
  6. Lanczos-based S_eff estimation via scipy eigsh

Validates matrix-free estimates against exact eigendecomposition on the 110-param DLN
(the "sanity check" that the matrix-free code recovers exact values).

Then measures delta, epsilon, S_eff on MNIST MLP (50,890 params) at several checkpoints.

Run via: uv run python scripts/matrix_free_experiments.py
"""

import json
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse.linalg import LinearOperator, eigsh

FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)
RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "matrix_free_results.json"
)


# ── Models ──────────────────────────────────────────────────────────────────


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


class TanhMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


# ── Vector Product Primitives ───────────────────────────────────────────────


def hvp(loss, params, v_list):
    """Hessian-vector product H @ v via Pearlmutter's trick (double backprop)."""
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # dot product of grads with v
    gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v_list))
    Hv = torch.autograd.grad(gv, params, create_graph=False)
    return [h.detach() for h in Hv]


def ggn_vp(model, X, Y, params, v_list, loss_type="mse"):
    """Gauss-Newton-vector product G @ v, batch-averaged.
    For softmax+CE or Gaussian-MSE, this equals the TRUE Fisher-vector product.
    Uses a vectorized double-backward pass for efficiency.
    """
    device = params[0].device
    X = X.to(device)
    Y = Y.to(device)
    v_list = [v.to(device) for v in v_list]
    
    pred = model(X)
    N = X.shape[0]
    C = pred.shape[1]
    
    # Step 1: Compute J v
    u = torch.zeros_like(pred, requires_grad=True)
    u_pred = torch.sum(u * pred)
    grads = torch.autograd.grad(u_pred, params, create_graph=True)
    
    grads_flat = torch.cat([g.reshape(-1) for g in grads])
    v_flat = torch.cat([v.reshape(-1) for v in v_list])
    v_Jt_u = torch.sum(grads_flat * v_flat)
    
    Jv = torch.autograd.grad(v_Jt_u, u)[0]
    
    # Step 2: Multiply by loss Hessian w.r.t prediction
    if loss_type == "mse":
        # Hessian of 1/(N*C) * ||pred - Y||^2 is (2 / (N * C)) * I
        HJv = Jv * (2.0 / (N * C))
    elif loss_type == "ce":
        # Hessian of CE w.r.t pred is (diag(p) - p p^T) / N
        p = torch.softmax(pred.detach(), dim=-1)
        HJv = (p * Jv - p * torch.sum(p * Jv, dim=-1, keepdim=True)) / N
    else:
        raise ValueError(loss_type)
        
    # Step 3: Compute J^T @ (H @ J_v)
    HJv_detached = HJv.detach()
    JtHJv_list = torch.autograd.grad(torch.sum(HJv_detached * pred), params, retain_graph=True)
    result_flat = torch.cat([g.reshape(-1) for g in JtHJv_list])
    
    return result_flat


def empirical_fvp(model, X, Y, params, v_flat, loss_type="mse"):
    """Empirical Fisher-vector product: F_emp @ v = (1/N) sum_i (g_i g_i^T) v.
    Uses a vectorized formulation to avoid explicit sample-by-sample loops.
    """
    device = params[0].device
    X = X.to(device)
    Y = Y.to(device)
    v_flat = v_flat.to(device)
    
    pred = model(X)
    N = X.shape[0]
    C = pred.shape[1]
    
    v_list = []
    offset = 0
    for p in params:
        n = p.numel()
        v_list.append(v_flat[offset : offset + n].view_as(p))
        offset += n
        
    # Step 1: Compute J v
    u = torch.zeros_like(pred, requires_grad=True)
    u_pred = torch.sum(u * pred)
    grads = torch.autograd.grad(u_pred, params, create_graph=True)
    grads_flat = torch.cat([g.reshape(-1) for g in grads])
    v_Jt_u = torch.sum(grads_flat * v_flat)
    Jv = torch.autograd.grad(v_Jt_u, u)[0]
    
    # Step 2: Compute individual d_i = \nabla_{pred_i} L_i
    if loss_type == "mse":
        d = (2.0 / C) * (pred - Y)
        individual_losses = torch.mean((pred - Y)**2, dim=-1)
    elif loss_type == "ce":
        p = torch.softmax(pred, dim=-1)
        Y_one_hot = torch.nn.functional.one_hot(Y.view(-1).long(), num_classes=C).float()
        d = p - Y_one_hot
        individual_losses = nn.CrossEntropyLoss(reduction="none")(pred, Y.view(-1).long())
    else:
        raise ValueError(loss_type)
        
    # Step 3: Compute a_i = d_i^T (J_i v)
    a = torch.sum(d * Jv, dim=-1)
    
    # Step 4: Compute Fv = (1/N) * sum_i a_i * \nabla_{\theta} L_i
    weighted_loss = torch.sum(a.detach() * individual_losses) / N
    Fv_list = torch.autograd.grad(weighted_loss, params, retain_graph=True)
    result_flat = torch.cat([g.reshape(-1) for g in Fv_list])
    
    return result_flat


def hvp_flat(model, X, Y, params, v_flat, loss_type="mse"):
    """Hessian-vector product with flat vector interface."""
    device = params[0].device
    X = X.to(device)
    Y = Y.to(device)
    v_flat = v_flat.to(device)

    v_list = []
    offset = 0
    for p in params:
        n = p.numel()
        v_list.append(v_flat[offset : offset + n].view_as(p))
        offset += n

    # Compute loss
    pred = model(X)
    if loss_type == "mse":
        loss = nn.MSELoss()(pred, Y)
    else:
        loss = nn.CrossEntropyLoss()(pred, Y.view(-1).long())

    Hv = hvp(loss, params, v_list)
    return torch.cat([h.reshape(-1) for h in Hv])


# ── CG Solver ───────────────────────────────────────────────────────────────


def cg_solve(matvec, b, damping, tol=1e-6, max_iter=50):
    """Solve (A + damping * I) x = b via conjugate gradient."""
    x = torch.zeros_like(b)
    r = b - (matvec(x) + damping * x)
    p = r.clone()
    rs_old = r @ r
    for it in range(max_iter):
        Ap = matvec(p) + damping * p
        pAp = p @ Ap
        if pAp.abs() < 1e-30:
            break
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


# ── Spectral Norm Estimation ───────────────────────────────────────────────


def power_iteration_spectral_norm(matvec_fn, dim, num_iters=50, seed=0, device="cpu"):
    """Estimate ||A||_2 via power iteration on A^T A.
    For symmetric A, this gives the largest absolute eigenvalue.
    For general A, handles indefiniteness by tracking both extremes.
    """
    torch.manual_seed(seed)
    v = torch.randn(dim, device=device)
    v = v / v.norm()

    eigenvalue = 0.0
    for _ in range(num_iters):
        Av = matvec_fn(v)
        eigenvalue = (v @ Av).item()
        Av_norm = Av.norm()
        if Av_norm < 1e-30:
            break
        v = Av / Av_norm

    # Also try negative direction
    torch.manual_seed(seed + 1000)
    v2 = torch.randn(dim, device=device)
    v2 = v2 / v2.norm()
    neg_eigenvalue = 0.0
    neg_matvec = lambda x: -matvec_fn(x)
    for _ in range(num_iters):
        Av2 = neg_matvec(v2)
        neg_eigenvalue = (v2 @ Av2).item()
        Av2_norm = Av2.norm()
        if Av2_norm < 1e-30:
            break
        v2 = Av2 / Av2_norm

    return max(abs(eigenvalue), abs(neg_eigenvalue))


def estimate_seff_lanczos(
    hvp_fn, fvp_fn, dim, damping=1e-3, k=20, tol=1e-8, device="cpu"
):
    """Estimate S_eff = lambda_max(F^{-1} H) via Lanczos on the operator
    v -> (F + gamma I)^{-1} H v, using CG for the Fisher inverse.

    Uses scipy.sparse.linalg.eigsh with a LinearOperator wrapper.
    """

    def finv_h_matvec(v_np):
        v = torch.from_numpy(v_np).float().to(device)
        Hv = hvp_fn(v)
        x = cg_solve(fvp_fn, Hv, damping, tol=1e-7, max_iter=100)
        return x.cpu().numpy()

    A_op = LinearOperator((dim, dim), matvec=finv_h_matvec)
    try:
        eigenvalues, _ = eigsh(A_op, k=min(k, dim - 2), which="LM", tol=tol)
        return float(np.max(np.abs(eigenvalues)))
    except Exception as e:
        print(f"  Warning: eigsh failed ({e}), falling back to power iteration")
        return power_iteration_spectral_norm(
            lambda v: torch.from_numpy(
                finv_h_matvec(v.cpu().numpy())
            ).float().to(device),
            dim,
            num_iters=50,
            device=device
        )


# ── Exact computations for validation ──────────────────────────────────────


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
        gvec = torch.cat([g.view(-1) for g in gi]).detach().cpu().numpy()
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
        gvec = torch.cat([g.view(-1) for g in gi]).detach().cpu().numpy()
        F += np.outer(gvec, gvec)
    F /= N
    return F


# ── EXPERIMENT 1: DLN Sanity Check ─────────────────────────────────────────


def run_dln_validation():
    """Validate matrix-free estimates against exact eigendecomposition on 110-param DLN."""
    print("=" * 70)
    print("EXPERIMENT 1: DLN Matrix-Free Validation (110 parameters)")
    print("=" * 70)

    depth, width, input_dim, N = 2, 10, 10, 200
    damping = 1e-3

    # Exact computations are performed on CPU
    device = torch.device("cpu")

    torch.manual_seed(42)
    np.random.seed(42)
    X = torch.randn(N, input_dim).to(device)
    teacher = DeepLinearNet(depth=depth, width=width, input_dim=input_dim).to(device)
    with torch.no_grad():
        Y = teacher(X)

    model = DeepLinearNet(depth=depth, width=width, input_dim=input_dim).to(device)
    params = list(model.parameters())
    d = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=0.1)

    print(f"  Parameters: {d}")
    print()

    checkpoints = [0, 25, 50, 75, 100]
    results = []

    for target_step in checkpoints:
        # Train to target_step
        current_step = results[-1]["step"] if results else 0
        for _ in range(target_step - current_step):
            pred = model(X)
            loss = nn.MSELoss()(pred, Y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        pred = model(X)
        loss = nn.MSELoss()(pred, Y)

        print(f"  Step {target_step}: loss = {loss.item():.6f}")

        # ── Exact computation ──
        t0 = time.time()
        H_np = full_hessian(loss, params)
        G_np = full_gauss_newton(model, X, params)
        F_np = full_empirical_fisher(model, X, Y, params)
        F_reg = F_np + damping * np.eye(d)

        eps_exact = float(np.linalg.norm(H_np - G_np, ord=2))
        delta_exact = float(np.linalg.norm(G_np - F_np, ord=2))
        F_inv_H = np.linalg.solve(F_reg, H_np)
        seff_exact = float(np.max(np.abs(np.linalg.eigvals(F_inv_H))).real)
        exact_time = time.time() - t0

        # ── Matrix-free estimation ──
        t0 = time.time()

        # epsilon = ||H - G||_2 via power iteration
        def hg_matvec(v):
            Hv = hvp_flat(model, X, Y, params, v, loss_type="mse")
            v_list = []
            offset = 0
            for p in params:
                n = p.numel()
                v_list.append(v[offset : offset + n].view_as(p))
                offset += n
            Gv = ggn_vp(model, X, Y, params, v_list, loss_type="mse")
            return Hv - Gv

        eps_mf = power_iteration_spectral_norm(hg_matvec, d, num_iters=50, device=device)

        # delta = ||G - F_emp||_2 via power iteration
        def gf_matvec(v):
            v_list = []
            offset = 0
            for p in params:
                n = p.numel()
                v_list.append(v[offset : offset + n].view_as(p))
                offset += n
            Gv = ggn_vp(model, X, Y, params, v_list, loss_type="mse")
            Fv = empirical_fvp(model, X, Y, params, v, loss_type="mse")
            return Gv - Fv

        delta_mf = power_iteration_spectral_norm(gf_matvec, d, num_iters=50, device=device)

        # S_eff via Lanczos
        def hvp_fn(v):
            return hvp_flat(model, X, Y, params, v, loss_type="mse")

        def fvp_fn(v):
            return empirical_fvp(model, X, Y, params, v, loss_type="mse")

        seff_mf = estimate_seff_lanczos(hvp_fn, fvp_fn, d, damping=damping, k=10, device=device)
        mf_time = time.time() - t0

        # Compute relative errors
        eps_err = abs(eps_mf - eps_exact) / max(eps_exact, 1e-12) * 100
        delta_err = abs(delta_mf - delta_exact) / max(delta_exact, 1e-12) * 100
        seff_err = abs(seff_mf - seff_exact) / max(seff_exact, 1e-12) * 100

        print(f"    Exact:       eps={eps_exact:.4f}, delta={delta_exact:.4f}, S_eff={seff_exact:.1f}  ({exact_time:.1f}s)")
        print(f"    Matrix-free: eps={eps_mf:.4f}, delta={delta_mf:.4f}, S_eff={seff_mf:.1f}  ({mf_time:.1f}s)")
        print(f"    Rel errors:  eps={eps_err:.1f}%, delta={delta_err:.1f}%, S_eff={seff_err:.1f}%")
        print()

        results.append({
            "step": target_step,
            "loss": float(loss.item()),
            "eps_exact": eps_exact,
            "eps_mf": eps_mf,
            "eps_rel_err_pct": eps_err,
            "delta_exact": delta_exact,
            "delta_mf": delta_mf,
            "delta_rel_err_pct": delta_err,
            "seff_exact": seff_exact,
            "seff_mf": seff_mf,
            "seff_rel_err_pct": seff_err,
            "exact_time_s": exact_time,
            "mf_time_s": mf_time,
        })

    return results


# ── EXPERIMENT 2: MNIST Matrix-Free ────────────────────────────────────────


def run_mnist_matrix_free():
    """Matrix-free estimation of delta, epsilon, S_eff on MNIST MLP (50,890 params)."""
    print("=" * 70)
    print("EXPERIMENT 2: MNIST Matrix-Free Estimation (50,890 parameters)")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST
    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("  torchvision not available, skipping MNIST experiment")
        return []

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    try:
        trainset = torchvision.datasets.MNIST(
            root=data_dir, train=True, download=True, transform=transform
        )
    except Exception as e:
        print(f"  Failed to load MNIST: {e}")
        return []

    # Use 2000-sample subset (matching paper)
    torch.manual_seed(42)
    indices = torch.randperm(len(trainset))[:2000]
    subset = torch.utils.data.Subset(trainset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=2000, shuffle=False)
    X_full, Y_full = next(iter(loader))
    X_full = X_full.view(-1, 784).to(device)
    Y_full = Y_full.to(device)

    model = TanhMLP(input_dim=784, hidden_dim=64, output_dim=10).to(device)
    params = list(model.parameters())
    d = sum(p.numel() for p in params)
    print(f"  Parameters: {d}")

    opt = torch.optim.SGD(params, lr=0.01)
    damping = 1e-3

    checkpoints = [1, 25, 50, 100, 200]
    results = []
    current_step = 0

    # Use a smaller subset for matrix-free ops
    N_mf = 200  # use 200 samples for matrix-free estimation (tractable)

    for target_step in checkpoints:
        # Train to target step
        for step in range(current_step, target_step):
            pred = model(X_full)
            loss = nn.CrossEntropyLoss()(pred, Y_full)
            opt.zero_grad()
            loss.backward()
            opt.step()
        current_step = target_step

        # Evaluate
        with torch.no_grad():
            pred = model(X_full)
            loss_val = nn.CrossEntropyLoss()(pred, Y_full).item()
            acc = (pred.argmax(dim=1) == Y_full).float().mean().item() * 100

        print(f"  Step {target_step}: loss={loss_val:.4f}, acc={acc:.1f}%")

        # Matrix-free estimation on subset
        X_sub = X_full[:N_mf].to(device)
        Y_sub = Y_full[:N_mf].to(device)

        t0 = time.time()

        # epsilon = ||H - G||_2
        def hg_matvec_ce(v):
            Hv = hvp_flat(model, X_sub, Y_sub, params, v, loss_type="ce")
            v_list = []
            offset = 0
            for p in params:
                n = p.numel()
                v_list.append(v[offset : offset + n].view_as(p))
                offset += n
            Gv = ggn_vp(model, X_sub, Y_sub, params, v_list, loss_type="ce")
            return Hv - Gv

        eps_est = power_iteration_spectral_norm(hg_matvec_ce, d, num_iters=30, device=device)

        # delta = ||G - F_emp||_2
        def gf_matvec_ce(v):
            v_list = []
            offset = 0
            for p in params:
                n = p.numel()
                v_list.append(v[offset : offset + n].view_as(p))
                offset += n
            Gv = ggn_vp(model, X_sub, Y_sub, params, v_list, loss_type="ce")
            Fv = empirical_fvp(model, X_sub, Y_sub, params, v, loss_type="ce")
            return Gv - Fv

        delta_est = power_iteration_spectral_norm(gf_matvec_ce, d, num_iters=30, device=device)

        # S_eff via Lanczos
        def hvp_fn_ce(v):
            return hvp_flat(model, X_sub, Y_sub, params, v, loss_type="ce")

        def fvp_fn_ce(v):
            return empirical_fvp(model, X_sub, Y_sub, params, v, loss_type="ce")

        seff_est = estimate_seff_lanczos(
            hvp_fn_ce, fvp_fn_ce, d, damping=damping, k=10, device=device
        )
        
        # Also estimate ||F_emp||_2 for relative delta
        F_norm_est = power_iteration_spectral_norm(fvp_fn_ce, d, num_iters=30, device=device)

        elapsed = time.time() - t0
        delta_rel = delta_est / max(F_norm_est, 1e-12)

        print(f"    eps={eps_est:.4f}, delta={delta_est:.4f}, delta/||F||={delta_rel:.2f}, S_eff={seff_est:.1f}  ({elapsed:.1f}s)")
        print()

        results.append({
            "step": target_step,
            "loss": loss_val,
            "accuracy": acc,
            "eps_est": float(eps_est),
            "delta_est": float(delta_est),
            "F_norm_est": float(F_norm_est),
            "delta_relative": float(delta_rel),
            "seff_est": float(seff_est),
            "n_params": d,
            "n_samples_mf": N_mf,
            "time_s": elapsed,
        })

    return results


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    all_results = {}

    # Experiment 1: DLN validation
    dln_results = run_dln_validation()
    all_results["dln_validation"] = dln_results

    # Experiment 2: MNIST matrix-free
    mnist_results = run_mnist_matrix_free()
    all_results["mnist_matrix_free"] = mnist_results

    # Save results
    with open(RESULTS_FILE, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")

    # Print summary table
    print("\n" + "=" * 70)
    print("SUMMARY: DLN Matrix-Free Validation")
    print("=" * 70)
    print(f"{'Step':>6} {'eps_exact':>10} {'eps_mf':>10} {'err%':>6} {'delta_exact':>12} {'delta_mf':>10} {'err%':>6} {'seff_exact':>11} {'seff_mf':>9} {'err%':>6}")
    for r in dln_results:
        print(f"{r['step']:>6d} {r['eps_exact']:>10.4f} {r['eps_mf']:>10.4f} {r['eps_rel_err_pct']:>5.1f}% {r['delta_exact']:>12.4f} {r['delta_mf']:>10.4f} {r['delta_rel_err_pct']:>5.1f}% {r['seff_exact']:>11.1f} {r['seff_mf']:>9.1f} {r['seff_rel_err_pct']:>5.1f}%")

    if mnist_results:
        print("\n" + "=" * 70)
        print("SUMMARY: MNIST Matrix-Free Estimation")
        print("=" * 70)
        print(f"{'Step':>6} {'Loss':>8} {'Acc%':>6} {'eps':>10} {'delta':>10} {'delta/||F||':>12} {'S_eff':>10} {'Time(s)':>8}")
        for r in mnist_results:
            print(f"{r['step']:>6d} {r['loss']:>8.4f} {r['accuracy']:>5.1f}% {r['eps_est']:>10.4f} {r['delta_est']:>10.4f} {r['delta_relative']:>12.2f} {r['seff_est']:>10.1f} {r['time_s']:>8.1f}")


if __name__ == "__main__":
    main()
    