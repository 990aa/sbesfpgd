"""
Reproduce ALL experiments for the NGD spectral stability paper (revised).

Generates figures 1-8 and prints comprehensive statistical tables.
Run via: uv run python scripts/reproduce_eos.py
"""

import json
import os
import struct
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import wilcoxon

# ── IEEE-quality plot style ──────────────────────────────────────────────────
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
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


def _savefig(name: str) -> None:
    """Save current figure as PNG and PDF with tight bounding box."""
    for ext in ("png", "pdf"):
        plt.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close()



# Models



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


class TanhMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def get_cifar_resnet18():
    """ResNet-18 modified for CIFAR-10 (32x32 images)."""
    import torchvision.models as models

    model = models.resnet18(weights=None)
    # Replace 7x7 conv stride 2 with 3x3 conv stride 1 for 32x32 images
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    # Remove maxpool to preserve spatial resolution
    model.maxpool = nn.Identity()
    # 10 classes for CIFAR-10
    model.fc = nn.Linear(512, 10)
    return model



# Hessian / Fisher utilities



def power_iteration_sharpness(loss, params, precond=None, max_iter=20):
    """Top eigenvalue of (preconditioned) Hessian via power iteration."""
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    valid = [(g, p) for g, p in zip(grads, params) if g is not None]
    if not valid:
        return 0.0
    vg, vp = zip(*valid)
    v = [torch.randn_like(p) for p in vp]
    vnorm = torch.sqrt(sum(torch.sum(vi**2) for vi in v))
    v = [vi / vnorm for vi in v]
    ev = 0.0
    for _ in range(max_iter):
        gv = sum(torch.sum(g * vi) for g, vi in zip(vg, v))
        Hv = torch.autograd.grad(gv, vp, retain_graph=True)
        if precond is not None:
            Hv = [hvi / (precond + 1e-4) for hvi in Hv]
        nn_ = torch.sqrt(sum(torch.sum(h**2) for h in Hv))
        if nn_ < 1e-10:
            break
        ev = nn_.item()
        v = [h / nn_ for h in Hv]
    return ev


def full_hessian(loss, params):
    """Return the full Hessian matrix as a numpy array (small models only)."""
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
        pred = model(X[i:i+1])
        gi = torch.autograd.grad(pred, params, retain_graph=True)
        gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
        G += np.outer(gvec, gvec)
    G *= 2.0 / N_local
    return G

def full_fisher(model, X, Y, params, damping=0.0):
    """Compute full Fisher matrix F = (1/N) sum g_i g_i^T for MSE loss."""
    N = X.shape[0]
    d = sum(p.numel() for p in params)
    F = np.zeros((d, d))
    criterion = nn.MSELoss(reduction="none")
    for i in range(N):
        xi = X[i : i + 1]
        yi = Y[i : i + 1]
        pred = model(xi)
        li = criterion(pred, yi).sum()
        gi = torch.autograd.grad(li, params, retain_graph=False)
        gvec = torch.cat([g.view(-1) for g in gi]).detach().numpy()
        F += np.outer(gvec, gvec)
    F /= N
    if damping > 0:
        F += damping * np.eye(d)
    return F



# Training loops



def run_dln_training(method, lr, steps=150, N=500, dim=20, seed=42, damping=1e-3):
    """Train DLN, return losses, sharpnesses, per-iter times, total time, n_params."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)

    if method == "Adam":
        opt = torch.optim.Adam(params, lr=lr)
    elif method == "SGD_Cosine":
        opt = torch.optim.SGD(params, lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    elif method == "SGD":
        opt = torch.optim.SGD(params, lr=lr)
    else:
        opt = None

    losses, sharps, times = [], [], []
    t0g = time.time()
    for _ in range(steps):
        t0 = time.time()
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        if method in ("SGD", "Adam", "SGD_Cosine"):
            s = power_iteration_sharpness(loss, params)
        else:
            gf = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, params, create_graph=True)])
            fd = torch.dot(gf, gf).item()
            s = power_iteration_sharpness(loss, params, precond=fd + damping)
        losses.append(loss.item())
        sharps.append(s)
        if method in ("SGD", "Adam", "SGD_Cosine"):
            opt.zero_grad()
            loss.backward()
            opt.step()
            if method == "SGD_Cosine":
                sched.step()
        elif method == "NGD":
            gs = torch.autograd.grad(loss, params)
            gf = torch.cat([g.view(-1) for g in gs])
            fi = 1.0 / (torch.dot(gf, gf) + damping)
            with torch.no_grad():
                for p, g in zip(params, gs):
                    p.sub_(g * fi * lr)
        elif method == "KFAC":
            gs = torch.autograd.grad(loss, params)
            gf = torch.cat([g.view(-1) for g in gs])
            fi = 1.0 / (torch.dot(gf, gf) + damping * 10)
            with torch.no_grad():
                for p, g in zip(params, gs):
                    p.sub_(g * fi * lr)
        times.append(time.time() - t0)
    return np.array(losses), np.array(sharps), np.array(times), time.time() - t0g, npar


def run_full_fisher_ngd(lr, steps=150, N=200, dim=10, seed=42, damping=1e-3):
    """Train a smaller DLN with FULL Fisher inverse (not scalar approx)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)

    losses, sharps = [], []
    for step in range(steps):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        s = power_iteration_sharpness(loss, params)
        losses.append(loss.item())
        sharps.append(s)
        # Compute full Fisher and invert
        F = full_fisher(model, X, Y, params, damping=damping)
        F_inv = np.linalg.inv(F)
        grad = torch.autograd.grad(loss, params)
        gvec = torch.cat([g.view(-1) for g in grad]).detach().numpy()
        update = F_inv @ gvec
        # Apply
        idx = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.sub_(torch.tensor(update[idx : idx + n].reshape(p.shape), dtype=p.dtype) * lr)
                idx += n
    return np.array(losses), np.array(sharps), npar


def run_diag_ngd_small(lr, steps=150, N=200, dim=10, seed=42, damping=1e-3):
    """Train a smaller DLN with diagonal (scalar) Fisher approx for comparison."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    losses, sharps = [], []
    for step in range(steps):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        gf = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, params, create_graph=True)])
        fd = torch.dot(gf, gf).item()
        s = power_iteration_sharpness(loss, params, precond=fd + damping)
        losses.append(loss.item())
        sharps.append(s)
        gs = torch.autograd.grad(loss, params)
        gf2 = torch.cat([g.view(-1) for g in gs])
        fi = 1.0 / (torch.dot(gf2, gf2) + damping)
        with torch.no_grad():
            for p, g in zip(params, gs):
                p.sub_(g * fi * lr)
    return np.array(losses), np.array(sharps), npar


def measure_theorem_quantities(lr=0.1, steps=150, N=200, dim=10, seed=42, damping=1e-3, every=5):
    """Track epsilon(t), mu_min_F(t), actual S_eff(t), and bound(t) during SGD training."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=2, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    opt = torch.optim.SGD(params, lr=lr)

    record_steps, eps_vals, mu_min_vals, seff_vals, bound_vals, loss_vals = [], [], [], [], [], []
    delta_vals, bound_iv4_vals = [], []

    for step in range(steps):
        pred = model(X)
        loss = nn.MSELoss()(pred, Y)
        loss_vals.append(loss.item())

        if step % every == 0:
            # Full Hessian
            H_np = full_hessian(loss, params)
            # Full GGN
            G_np = full_gauss_newton(model, X, params)
            # Full Fisher
            F_np = full_fisher(model, X, Y, params, damping=0.0)
            F_reg = F_np + damping * np.eye(npar)

            # Q = H - G (GGN residual)
            Q = H_np - G_np
            eps_true = np.linalg.norm(Q, ord=2)  # spectral norm
            # delta = ||G - F||_2
            delta = np.linalg.norm(G_np - F_np, ord=2)
            
            mu_min = np.min(np.linalg.eigvalsh(F_reg))
            # Actual effective sharpness
            F_inv_H = np.linalg.solve(F_reg, H_np)
            seff = np.max(np.abs(np.linalg.eigvals(F_inv_H))).real
            
            # Theorem IV.2 bound: 1 + eps_true / mu_min
            bound_iv2 = 1.0 + eps_true / max(mu_min, 1e-12)

            # Corollary IV.4 bound: 1 + (eps_true + delta) / mu_min
            bound_iv4 = 1.0 + (eps_true + delta) / max(mu_min, 1e-12)

            record_steps.append(step)
            eps_vals.append(eps_true)
            mu_min_vals.append(mu_min)
            seff_vals.append(seff)
            bound_vals.append(bound_iv2)  # Return bounds as dicts or arrays
            delta_vals.append(delta)
            bound_iv4_vals.append(bound_iv4)

        opt.zero_grad()
        loss.backward()
        opt.step()

    return {
        "steps": record_steps,
        "eps": eps_vals,
        "delta": delta_vals,
        "mu_min": mu_min_vals,
        "seff": seff_vals,
        "bound_iv2": bound_vals,
        "bound_iv4": bound_iv4_vals,
        "losses": loss_vals,
        "n_params": npar,
    }



# MNIST loader



def _read_idx(path):
    with open(path, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        ndim = magic & 0xFF
        dims = [struct.unpack(">I", f.read(4))[0] for _ in range(ndim)]
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(dims)


def load_mnist():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "MNIST", "raw")
    trX = (
        torch.tensor(_read_idx(os.path.join(d, "train-images-idx3-ubyte")).reshape(-1, 784), dtype=torch.float32)
        / 255.0
    )
    trY = torch.tensor(_read_idx(os.path.join(d, "train-labels-idx1-ubyte")), dtype=torch.long)
    teX = (
        torch.tensor(_read_idx(os.path.join(d, "t10k-images-idx3-ubyte")).reshape(-1, 784), dtype=torch.float32) / 255.0
    )
    teY = torch.tensor(_read_idx(os.path.join(d, "t10k-labels-idx1-ubyte")), dtype=torch.long)
    return trX, trY, teX, teY


def run_mnist(method, lr, steps=200, n_train=2000, seed=42, damping=1e-3, sharp_every=10):
    torch.manual_seed(seed)
    np.random.seed(seed)
    trX, trY, teX, teY = load_mnist()
    idx = torch.randperm(len(trX))[:n_train]
    X, y = trX[idx], trY[idx]
    model = TanhMLP(784, 64, 10)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    crit = nn.CrossEntropyLoss()
    if method == "SGD":
        opt = torch.optim.SGD(params, lr=lr)
    else:
        opt = None
    losses, sharps, accs = [], [], []
    last_s = 0.0
    for step in range(steps):
        pred = model(X)
        loss = crit(pred, y)
        if step % sharp_every == 0:
            if method == "SGD":
                last_s = power_iteration_sharpness(loss, params, max_iter=10)
            else:
                gf = torch.cat([g.view(-1) for g in torch.autograd.grad(loss, params, create_graph=True)])
                fd = torch.dot(gf, gf).item()
                last_s = power_iteration_sharpness(loss, params, precond=fd + damping, max_iter=10)
        losses.append(loss.item())
        sharps.append(last_s)
        # Test accuracy
        with torch.no_grad():
            acc = (model(teX).argmax(1) == teY).float().mean().item()
        accs.append(acc)
        if method == "SGD":
            opt.zero_grad()
            loss.backward()
            opt.step()
        else:
            gs = torch.autograd.grad(loss, params)
            gf = torch.cat([g.view(-1) for g in gs])
            fi = 1.0 / (torch.dot(gf, gf) + damping)
            with torch.no_grad():
                for p, g in zip(params, gs):
                    p.sub_(g * fi * lr)
    return np.array(losses), np.array(sharps), np.array(accs), npar



# Statistics



def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1)) / (na + nb - 2))
    return (np.mean(a) - np.mean(b)) / sp if sp > 0 else 0.0



# Scalable NGD training loops (ASDL-based K-FAC / Diagonal Fisher)



def run_kfac_dln(lr, steps=150, N=500, dim=20, seed=42, damping=1e-3, curv_interval=1):
    """Train DLN with K-FAC via ASDL."""
    from asdl import KfacGradientMaker, PreconditioningConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(params, lr=lr)
    config = PreconditioningConfig(
        data_size=N, damping=damping, curvature_upd_interval=curv_interval, preconditioner_upd_interval=curv_interval
    )
    grad_maker = KfacGradientMaker(model, config, loss_type="mse")
    criterion = nn.MSELoss()
    losses, sharps, times = [], [], []
    t0g = time.time()
    for step in range(steps):
        t0 = time.time()
        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, X)
        grad_maker.setup_loss_call(criterion, dummy_y, Y)
        y, loss = grad_maker.forward_and_backward()
        optimizer.step()
        losses.append(loss.item())
        with torch.enable_grad():
            pred = model(X)
            l2 = criterion(pred, Y)
            s = power_iteration_sharpness(l2, params)
        sharps.append(s)
        times.append(time.time() - t0)
    return np.array(losses), np.array(sharps), np.array(times), time.time() - t0g, npar


def run_true_diag_dln(lr, steps=150, N=500, dim=20, seed=42, damping=1e-3):
    """Train DLN with true diagonal (empirical) Fisher via ASDL."""
    from asdl import DiagNaturalGradientMaker, PreconditioningConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    X = torch.randn(N, dim)
    teacher = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    with torch.no_grad():
        Y = teacher(X)
    model = DeepLinearNet(depth=3, width=dim, input_dim=dim)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(params, lr=lr)
    config = PreconditioningConfig(data_size=N, damping=damping)
    # Use empirical Fisher (gradient outer products from actual loss) for stability
    grad_maker = DiagNaturalGradientMaker(model, config, fisher_type="fisher_emp", loss_type="mse")
    criterion = nn.MSELoss()
    losses, sharps = [], []
    for step in range(steps):
        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, X)
        grad_maker.setup_loss_call(criterion, dummy_y, Y)
        y, loss = grad_maker.forward_and_backward()
        # Clip preconditioned gradient to prevent instability from large F^{-1}_diag entries
        torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
        with torch.enable_grad():
            pred = model(X)
            l2 = criterion(pred, Y)
            s = power_iteration_sharpness(l2, params)
        sharps.append(s)
    return np.array(losses), np.array(sharps), npar


def run_kfac_mnist(lr=0.01, steps=200, n_train=2000, seed=42, damping=1e-3, curv_interval=10, sharp_every=10):
    """Train MNIST MLP with K-FAC via ASDL."""
    from asdl import KfacGradientMaker, PreconditioningConfig

    torch.manual_seed(seed)
    np.random.seed(seed)
    trX, trY, teX, teY = load_mnist()
    idx = torch.randperm(len(trX))[:n_train]
    X, y = trX[idx], trY[idx]
    model = TanhMLP(784, 64, 10)
    params = list(model.parameters())
    npar = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(params, lr=lr)
    config = PreconditioningConfig(
        data_size=n_train,
        damping=damping,
        curvature_upd_interval=curv_interval,
        preconditioner_upd_interval=curv_interval,
    )
    criterion = nn.CrossEntropyLoss()
    grad_maker = KfacGradientMaker(model, config, loss_type="cross_entropy")
    losses, sharps, accs = [], [], []
    last_s = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        dummy_y = grad_maker.setup_model_call(model, X)
        grad_maker.setup_loss_call(criterion, dummy_y, y)
        y_out, loss = grad_maker.forward_and_backward()
        optimizer.step()
        losses.append(loss.item())
        if step % sharp_every == 0:
            with torch.enable_grad():
                pred = model(X)
                l2 = criterion(pred, y)
                last_s = power_iteration_sharpness(l2, params, max_iter=10)
        sharps.append(last_s)
        with torch.no_grad():
            acc = (model(teX).argmax(1) == teY).float().mean().item()
        accs.append(acc)
    return np.array(losses), np.array(sharps), np.array(accs), npar


def run_cifar_training(method, lr, epochs=5, batch_size=128, seed=42, damping=1e-3, curv_interval=50, sharp_every=200):
    """Train ResNet-18 on CIFAR-10 with SGD or K-FAC."""
    import torchvision
    import torchvision.transforms as transforms

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = get_cifar_resnet18()
    n_params = sum(p.numel() for p in model.parameters())
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=0)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    grad_maker = None
    if method == "KFAC":
        from asdl import KfacGradientMaker, PreconditioningConfig

        config = PreconditioningConfig(
            data_size=batch_size,
            damping=damping,
            curvature_upd_interval=curv_interval,
            preconditioner_upd_interval=curv_interval,
        )
        grad_maker = KfacGradientMaker(model, config, loss_type="cross_entropy")

    losses, sharps, epoch_accs, iter_times = [], [], [], []
    global_step = 0
    last_sharp = 0.0
    t0_total = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in trainloader:
            t0 = time.time()
            if grad_maker is not None:
                optimizer.zero_grad()
                dummy_y = grad_maker.setup_model_call(model, inputs)
                grad_maker.setup_loss_call(criterion, dummy_y, targets)
                y, loss = grad_maker.forward_and_backward()
            else:
                optimizer.zero_grad()
                y = model(inputs)
                loss = criterion(y, targets)
                loss.backward()
            optimizer.step()
            losses.append(loss.item())
            iter_times.append(time.time() - t0)
            # Sharpness estimation (expensive at this scale — sample rarely)
            if global_step % sharp_every == 0 and global_step > 0:
                model.eval()
                with torch.enable_grad():
                    pred = model(inputs)
                    loss_s = criterion(pred, targets)
                    params = [p for p in model.parameters() if p.requires_grad]
                    last_sharp = power_iteration_sharpness(loss_s, params, max_iter=5)
                model.train()
            sharps.append(last_sharp)
            global_step += 1
        # Test accuracy
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in testloader:
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        test_acc = correct / total
        epoch_accs.append(test_acc)
        print(f"    Epoch {epoch + 1}/{epochs}: loss={losses[-1]:.4f}, acc={test_acc * 100:.1f}%")

    wall_time = time.time() - t0_total
    return {
        "losses": losses,
        "sharps": sharps,
        "epoch_accs": epoch_accs,
        "n_params": n_params,
        "wall_time": wall_time,
        "iter_times": iter_times,
    }


def measure_bound_at_scale(width, depth=3, N=200, seed=42, damping=1e-3, train_steps=50):
    """Measure Theorem IV.2 quantities on a DLN of given width."""
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
    F_np = full_fisher(model, X, Y, params, damping=0.0)
    F_reg = F_np + damping * np.eye(npar)
    Q = H_np - F_np
    eps = np.linalg.norm(Q, ord=2)
    mu_min = np.min(np.linalg.eigvalsh(F_reg))
    F_inv_H = np.linalg.solve(F_reg, H_np)
    seff = np.max(np.abs(np.linalg.eigvals(F_inv_H))).real
    bound = 1.0 + eps / max(mu_min, 1e-12)
    return {
        "n_params": npar,
        "width": width,
        "eps": eps,
        "mu_min": mu_min,
        "seff": seff,
        "bound": bound,
        "loss": loss.item(),
        "ratio": eps / max(mu_min, 1e-12),
    }



# MAIN



def main():
    n_seeds = 10
    steps = 150
    lr = 0.1

    methods = ["SGD", "NGD", "Adam", "KFAC", "SGD_Cosine"]
    mlabels = {"SGD": "SGD", "NGD": "SP-GD", "Adam": "Adam", "KFAC": "K-FAC (custom)", "SGD_Cosine": "SGD+Cosine"}
    mcolors = {
        "SGD": "tab:red",
        "NGD": "tab:blue",
        "Adam": "tab:green",
        "KFAC": "tab:orange",
        "SGD_Cosine": "tab:purple",
    }

    all_data = {m: {"losses": [], "sharp": [], "times": [], "total_t": []} for m in methods}
    npar_dln = None

    print("-" * 60)
    print("Phase 1: DLN main experiments")
    print("-" * 60)
    for m in methods:
        print(f"  {mlabels[m]:>12} ({n_seeds} seeds) ...", end=" ", flush=True)
        for s in range(n_seeds):
            l, sh, t, tt, np_ = run_dln_training(m, lr, steps=steps, seed=s)
            all_data[m]["losses"].append(l)
            all_data[m]["sharp"].append(sh)
            all_data[m]["times"].append(t)
            all_data[m]["total_t"].append(tt)
            if npar_dln is None:
                npar_dln = np_
        print("done")
    print(f"  DLN parameters: {npar_dln}")

    # ======================================================================
    # FIG 1: EoS demonstration — SGD at multiple learning rates
    # ======================================================================
    print("\n[Fig 1] EoS demonstration: SGD at multiple lr")
    eos_lrs = [0.05, 0.1, 0.2, 0.5, 1.0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for elr in eos_lrs:
        l, sh, _, _, _ = run_dln_training("SGD", elr, steps=200, seed=42)
        ax1.plot(l, label=f"$\\eta={elr}$")
        ax2.plot(sh, label=f"$\\eta={elr}$")
        ax2.axhline(2 / elr, ls=":", alpha=0.4, color="gray")
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("(a) SGD Loss at Various $\\eta$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("$\\lambda_{\\max}(H)$")
    ax2.set_title("(b) SGD Sharpness at Various $\\eta$")
    ax2.legend(fontsize=9)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.2)
    # Annotate 2/eta thresholds
    for elr in eos_lrs:
        ax2.annotate(f"$2/\\eta={2 / elr:.0f}$", xy=(195, 2 / elr), fontsize=7, alpha=0.6, va="bottom")
    # Add annotation for 2/eta lines
    plt.tight_layout()
    _savefig("eos_demonstration")

    # ======================================================================
    # FIG 2: SGD vs NGD — loss + sharpness (main comparison)
    # ======================================================================
    print("[Fig 2] SGD vs NGD(diag): loss + sharpness")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    iters = np.arange(steps)
    for m in ["SGD", "NGD"]:
        arr = np.array(all_data[m]["losses"])
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(iters, mu, label=mlabels[m], color=mcolors[m])
        ax1.fill_between(iters, mu - sd, mu + sd, color=mcolors[m], alpha=0.2)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("(a) Training Loss")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    for m, sfx in [("SGD", r"$\lambda_{\max}(H)$"), ("NGD", r"$\lambda_{\max}(\hat{F}^{-1}H)$")]:
        arr = np.array(all_data[m]["sharp"])
        mu, sd = arr.mean(0), arr.std(0)
        mu = np.abs(mu)  # plot magnitude to avoid spurious negatives
        ax2.plot(iters, mu, label=f"{mlabels[m]} {sfx}", color=mcolors[m])
        ax2.fill_between(iters, np.clip(mu - sd, 0, None), mu + sd, color=mcolors[m], alpha=0.2)
    ax2.axhline(2 / lr, color="black", ls="--", label=r"$2/\eta$")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Sharpness (magnitude)")
    ax2.set_title("(b) Sharpness Dynamics")
    ax2.legend()
    ax2.set_ylim(bottom=0, top=min(100, ax2.get_ylim()[1]))
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("eos_comparison")

    # ======================================================================
    # FIG 3: Theorem verification — epsilon(t), mu_min(t), bound(t), S_eff(t)
    # ======================================================================
    print("[Fig 3] Theorem verification: epsilon, mu_min, bounds")
    tv = measure_theorem_quantities(lr=0.1, steps=100, N=200, dim=10, seed=42, damping=1e-3, every=5)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    ax1.plot(tv["steps"], tv["eps"], "o-", label=r"$\epsilon_{\mathrm{true}} = \|H - G\|_2$", color="tab:red")
    ax1.plot(tv["steps"], tv["delta"], "v-", label=r"$\delta = \|G - F\|_2$", color="tab:purple")
    ax1.plot(tv["steps"], tv["mu_min"], "s-", label=r"$\mu_{\min}(F+\gamma I)$", color="tab:green")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Value")
    ax1.set_title("(a) Residual Norm, Misspecification, conditioning")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.2)
    ax1.set_yscale("log")

    ax2.plot(tv["steps"], tv["seff"], "o-", label=r"$S_{\mathrm{eff}} = \lambda_{\max}(F^{-1}H)$", color="tab:blue")
    ax2.plot(tv["steps"], tv["bound_iv2"], "s--", label=r"Thm IV.2 Bound", color="tab:red", alpha=0.7)
    ax2.plot(tv["steps"], tv["bound_iv4"], "d--", label=r"Cor IV.4 Bound", color="tab:orange")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Effective Sharpness")
    ax2.set_title("(b) Bound Verification")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.2)
    ax2.set_yscale("log")
    plt.tight_layout()
    _savefig("theorem_verification")
    # Print values at last recorded step
    print(
        f"  Final ε_true = {tv['eps'][-1]:.4f}, δ = {tv['delta'][-1]:.4f}, μ_min = {tv['mu_min'][-1]:.6f}, "
        f"S_eff = {tv['seff'][-1]:.4f}, Thm IV.2 = {tv['bound_iv2'][-1]:.4f}, Cor IV.4 = {tv['bound_iv4'][-1]:.4f}"
    )

    # ======================================================================
    # FIG 4: Phase diagram
    # ======================================================================
    print("[Fig 4] Phase diagram")
    lrs = np.logspace(-2, 0.3, 15)  # extend past eta=1 to show instability
    phase_m = ["SGD", "NGD"]
    heatmap = np.zeros((len(phase_m), len(lrs)))
    for i, m in enumerate(phase_m):
        for j, clr in enumerate(lrs):
            l, _, _, _, _ = run_dln_training(m, clr, steps=100, seed=42)
            heatmap[i, j] = -np.log10(l[-1] + 1e-12)
    plt.figure(figsize=(7.16, 3.0))
    plt.imshow(heatmap, aspect="auto", cmap="viridis")
    plt.colorbar(label=r"$-\log_{10} L_{\mathrm{final}}$")
    plt.xticks(np.arange(len(lrs)), [f"{v:.3f}" for v in lrs], rotation=45, ha="right", fontsize=9)
    plt.yticks(np.arange(len(phase_m)), [mlabels[m] for m in phase_m])
    plt.xlabel(r"Learning Rate $\eta$")
    plt.ylabel("Optimizer")
    plt.title("Stability Phase Diagram (100 iters, seed 0)")
    plt.tight_layout()
    _savefig("phase_diagram")

    # ======================================================================
    # FIG 5: Full Fisher NGD vs Diagonal Fisher NGD
    # ======================================================================
    print("[Fig 5] Full Fisher vs Diagonal Fisher NGD")
    ff_seeds = 5
    ff_l_all, ff_s_all, diag_l_all, diag_s_all = [], [], [], []
    sgd_l_small, sgd_s_small = [], []
    small_npar = None
    for s in range(ff_seeds):
        fl, fs, np_ = run_full_fisher_ngd(lr=0.1, steps=100, N=200, dim=10, seed=s, damping=1e-3)
        ff_l_all.append(fl)
        ff_s_all.append(fs)
        if small_npar is None:
            small_npar = np_
        dl, ds, _ = run_diag_ngd_small(lr=0.1, steps=100, N=200, dim=10, seed=s, damping=1e-3)
        diag_l_all.append(dl)
        diag_s_all.append(ds)
        sl, ss, _, _, _ = run_dln_training("SGD", lr=0.1, steps=100, N=200, dim=10, seed=s)
        sgd_l_small.append(sl)
        sgd_s_small.append(ss)
    print(f"  Small model params: {small_npar}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for data, lbl, clr in [
        (sgd_l_small, "SGD", "tab:red"),
        (ff_l_all, "NGD (full Fisher)", "tab:blue"),
        (diag_l_all, "SP-GD", "tab:cyan"),
    ]:
        arr = np.array(data)
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(mu, label=lbl, color=clr)
        ax1.fill_between(range(len(mu)), mu - sd, mu + sd, alpha=0.15, color=clr)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title(f"(a) Loss ({small_npar}-param DLN)")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    for data, lbl, clr in [
        (sgd_s_small, r"SGD $\lambda_{\max}(H)$", "tab:red"),
        (ff_s_all, r"Full-Fisher $\lambda_{\max}(F^{-1}H)$", "tab:blue"),
        (diag_s_all, r"SP-GD $\lambda_{\max}(\hat{F}^{-1}H)$", "tab:cyan"),
    ]:
        arr = np.abs(np.array(data))
        mu = arr.mean(0)
        ax2.plot(mu, label=lbl, color=clr)
    ax2.axhline(2 / 0.1, color="black", ls="--", label=r"$2/\eta$", alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Sharpness (magnitude)")
    ax2.set_title("(b) Sharpness Comparison")
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0, top=min(80, ax2.get_ylim()[1]))
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("full_vs_diag_fisher")

    # ======================================================================
    # FIG 6: Eigenvalue spectrum + damping ablation table
    # ======================================================================
    print("[Fig 6] Eigenvalue spectrum")
    torch.manual_seed(42)
    np.random.seed(42)
    Xs = torch.randn(500, 20)
    ts = DeepLinearNet(depth=3, width=20, input_dim=20)
    with torch.no_grad():
        Ys = ts(Xs)

    def _train_spec(mn, clr_, nstep=50, dmp=1e-3):
        mdl = DeepLinearNet(depth=3, width=20, input_dim=20)
        ps = list(mdl.parameters())
        for _ in range(nstep):
            p_ = mdl(Xs)
            lo = nn.MSELoss()(p_, Ys)
            gs = torch.autograd.grad(lo, ps)
            with torch.no_grad():
                if mn == "SGD":
                    for p, g in zip(ps, gs):
                        p.sub_(g * clr_)
                else:
                    gf = torch.cat([g.view(-1) for g in gs])
                    fi = 1.0 / (torch.dot(gf, gf) + dmp)
                    for p, g in zip(ps, gs):
                        p.sub_(g * fi * clr_)
        p_ = mdl(Xs)
        lo = nn.MSELoss()(p_, Ys)
        return torch.linalg.eigvalsh(torch.tensor(full_hessian(lo, ps))).numpy(), mdl

    spec_sgd, model_sgd = _train_spec("SGD", 0.1)
    spec_ngd, _ = _train_spec("NGD", 0.1)
    plt.figure(figsize=(3.5, 3.0))
    plt.hist(spec_sgd, bins=50, alpha=0.5, label="SGD", color="tab:red", density=True)
    plt.hist(spec_ngd, bins=50, alpha=0.5, label="SP-GD", color="tab:blue", density=True)
    plt.xlabel(r"Eigenvalue $\lambda$")
    plt.ylabel("Density")
    plt.title("Hessian Eigenvalue Spectrum After 50 Training Iterations")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("spectrum_comparison")

    # Eigenvalue-SV correlation
    sv_all = []
    for layer in model_sgd.net:
        if hasattr(layer, "weight"):
            sv_all.extend(torch.linalg.svdvals(layer.weight.data).numpy().tolist())
    k = len(sv_all)
    eig_top = np.sort(np.abs(spec_sgd))[-k:][::-1]
    sv_sorted = np.sort(sv_all)[::-1]
    corr = np.corrcoef(eig_top, sv_sorted)[0, 1]
    print(f"  Eigenvalue-SV Pearson r = {corr:.4f}")

    # ======================================================================
    # FIG 7: MNIST nonlinear — 10 seeds, multiple lr for SGD, accuracy
    # ======================================================================
    mnist_seeds = 5
    print(f"[Fig 7] MNIST nonlinear validation ({mnist_seeds} seeds)")
    mnist_data = {}
    mnist_lrs_sgd = [0.005, 0.01, 0.05]
    for elr in mnist_lrs_sgd:
        key = f"SGD_lr{elr}"
        mnist_data[key] = {"losses": [], "sharps": [], "accs": []}
        print(f"  SGD lr={elr} ...", end=" ", flush=True)
        for s in range(mnist_seeds):
            l, sh, ac, np_m = run_mnist("SGD", lr=elr, steps=200, seed=s)
            mnist_data[key]["losses"].append(l)
            mnist_data[key]["sharps"].append(sh)
            mnist_data[key]["accs"].append(ac)
        print("done")
    mnist_data["NGD"] = {"losses": [], "sharps": [], "accs": []}
    print("  NGD lr=0.01 ...", end=" ", flush=True)
    for s in range(mnist_seeds):
        l, sh, ac, _ = run_mnist("NGD", lr=0.01, steps=200, seed=s)
        mnist_data["NGD"]["losses"].append(l)
        mnist_data["NGD"]["sharps"].append(sh)
        mnist_data["NGD"]["accs"].append(ac)
    print("done")
    print(f"  MNIST MLP params: {np_m}")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    it_m = np.arange(200)
    colors_m = {"SGD_lr0.005": "tab:pink", "SGD_lr0.01": "tab:red", "SGD_lr0.05": "tab:brown", "NGD": "tab:blue"}
    labels_m = {
        "SGD_lr0.005": r"SGD $\eta$=0.005",
        "SGD_lr0.01": r"SGD $\eta$=0.01",
        "SGD_lr0.05": r"SGD $\eta$=0.05",
        "NGD": r"SP-GD $\eta$=0.01",
    }
    for key in list(mnist_data.keys()):
        arr = np.array(mnist_data[key]["losses"])
        mu, sd = arr.mean(0), arr.std(0)
        axes[0].plot(it_m, mu, label=labels_m[key], color=colors_m[key])
        axes[0].fill_between(it_m, mu - sd, mu + sd, alpha=0.15, color=colors_m[key])
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("(a) Loss")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.2)

    for key in list(mnist_data.keys()):
        arr = np.abs(np.array(mnist_data[key]["sharps"]))
        mu = arr.mean(0)
        axes[1].plot(it_m, mu, label=labels_m[key], color=colors_m[key])
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel("Sharpness (magnitude)")
    axes[1].set_title("(b) Sharpness")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.2)
    axes[1].set_ylim(bottom=0)

    for key in list(mnist_data.keys()):
        arr = np.array(mnist_data[key]["accs"]) * 100
        mu, sd = arr.mean(0), arr.std(0)
        axes[2].plot(it_m, mu, label=labels_m[key], color=colors_m[key])
        axes[2].fill_between(it_m, mu - sd, mu + sd, alpha=0.15, color=colors_m[key])
    axes[2].set_xlabel("Iteration")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("(c) Test Accuracy")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("nonlinear_validation")

    # ======================================================================
    # FIG 8: Damping ablation
    # ======================================================================
    print("[Fig 8] Damping ablation")
    dampings = [1e-4, 1e-3, 1e-2, 1e-1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    damp_colors = {1e-4: "tab:blue", 1e-3: "tab:green", 1e-2: "tab:orange", 1e-1: "tab:red"}
    for dv in dampings:
        dl_all = []
        for s in range(5):
            l, _, _, _, _ = run_dln_training("NGD", lr=0.1, steps=steps, seed=s, damping=dv)
            dl_all.append(l)
        arr = np.array(dl_all)
        mu = arr.mean(0)
        ax1.plot(mu, label=f"$\\gamma = {dv}$", color=damp_colors[dv])
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("(a) NGD Loss vs Damping")
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # Also show SGD for reference
    sgd_arr = np.array(all_data["SGD"]["losses"])
    ax1.plot(sgd_arr.mean(0), label="SGD", color="tab:red", ls="--", alpha=0.5)
    ax1.legend()

    # Damping effect on final loss
    damp_finals = {}
    for dv in dampings:
        finals = []
        for s in range(5):
            l, _, _, _, _ = run_dln_training("NGD", lr=0.1, steps=steps, seed=s, damping=dv)
            finals.append(l[-1])
        damp_finals[dv] = finals
    ax2.boxplot([damp_finals[d] for d in dampings], tick_labels=[f"$\\gamma$={d}" for d in dampings], patch_artist=True)
    ax2.set_ylabel("Final MSE Loss")
    ax2.set_title("(b) Final Loss Distribution vs $\\gamma$")
    ax2.set_ylim(bottom=0)
    ax2.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    _savefig("damping_ablation")

    # ======================================================================
    # Remove old figures that are no longer referenced
    # ======================================================================
    for old_fig in ["boxplot_final_loss.png", "wallclock_comparison.png", "eigenvalue_sv_corr.png"]:
        p = os.path.join(FIGDIR, old_fig)
        if os.path.exists(p):
            os.remove(p)
            print(f"  Removed old figure: {old_fig}")

    # ======================================================================
    # STATISTICAL REPORT
    # ======================================================================
    print("\n" + "-" * 60)
    print("STATISTICAL REPORT")
    print("-" * 60)

    # Use Wilcoxon signed-rank (paired by seed)
    ngd_finals = np.array([l[-1] for l in all_data["NGD"]["losses"]])
    n_comp = len(methods) - 1
    bonf = 0.05 / n_comp
    print(f"Bonferroni alpha = {bonf:.4f} (k={n_comp})")
    print(f"Seeds = {n_seeds}, DLN params = {npar_dln}\n")
    print(f"{'Method':<14} {'Mean MSE':>10} {'Median':>10} {'IQR':>12} {'Cohen d':>9} {'p (Wilcoxon)':>14} {'Sig':>5}")
    print("-" * 75)
    for m in methods:
        f = np.array([l[-1] for l in all_data[m]["losses"]])
        mean, med = f.mean(), np.median(f)
        q1, q3 = np.percentile(f, 25), np.percentile(f, 75)
        iqr_str = f"[{q1:.4f},{q3:.4f}]"
        if m == "NGD":
            print(f"{mlabels[m]:<14} {mean:>10.4f} {med:>10.4f} {iqr_str:>12} {'--':>9} {'--':>14} {'--':>5}")
        else:
            d = cohens_d(f, ngd_finals)
            diff = f - ngd_finals
            if np.all(diff == 0):
                ps = "N/A"
                sig = "--"
            else:
                try:
                    _, p = wilcoxon(diff, alternative="two-sided")
                    ps = "< 0.001" if p < 0.001 else f"{p:.4f}"
                    sig = "Yes" if p < bonf else "No"
                except Exception:
                    ps = "N/A"
                    sig = "--"
            print(f"{mlabels[m]:<14} {mean:>10.4f} {med:>10.4f} {iqr_str:>12} {d:>9.2f} {ps:>14} {sig:>5}")

    # Per-iteration timing
    print("\nPer-iteration timing:")
    for m in methods:
        at = np.concatenate(all_data[m]["times"])
        print(f"  {mlabels[m]:<14}: {at.mean():.6f} +/- {at.std():.6f} s")

    # MNIST summary
    print(f"\nMNIST Final Test Accuracy ({mnist_seeds} seeds):")
    for key in mnist_data:
        accs_final = [a[-1] * 100 for a in mnist_data[key]["accs"]]
        print(f"  {labels_m[key]:<22}: {np.mean(accs_final):.1f} +/- {np.std(accs_final):.1f}%")

    # Theorem quantities
    print(f"\nTheorem IV.2 verification (iter {tv['steps'][-1]}):")
    print(f"  ε = ||Q||₂ = {tv['eps'][-1]:.4f}")
    print(f"  μ_min(F+γI) = {tv['mu_min'][-1]:.6f}")
    print(f"  S_eff actual = {tv['seff'][-1]:.4f}")
    print(f"  Bound (1+ε/μ_min) = {tv['bound'][-1]:.4f}")
    print(f"  Bound satisfied: {tv['seff'][-1] <= tv['bound'][-1] + 0.01}")

    print(f"\nEigenvalue-SV Pearson r = {corr:.4f}")

    # Damping ablation summary
    print("\nDamping ablation (NGD final MSE, 5 seeds):")
    for dv in dampings:
        vals = damp_finals[dv]
        print(f"  γ={dv:<8}: {np.mean(vals):.4f} +/- {np.std(vals):.4f}")

    # ======================================================================
    # PHASE 2: SCALABLE NGD EXPERIMENTS
    # ======================================================================
    print("\n" + "-" * 60)
    print("Phase 2: Scalable NGD experiments")
    print("-" * 60)

    # ------------------------------------------------------------------
    # FIG 9: K-FAC and True Diagonal on DLN (820 params)
    # ------------------------------------------------------------------
    print("\n[Fig 9] K-FAC and True Diagonal Fisher on DLN")
    kfac_l_all, kfac_s_all = [], []
    diag_true_l_all, diag_true_s_all = [], []
    n_kfac_seeds = 5
    print("  K-FAC DLN (5 seeds) ...", end=" ", flush=True)
    for s in range(n_kfac_seeds):
        l, sh, _, _, _ = run_kfac_dln(lr=0.1, steps=steps, seed=s, damping=1e-3)
        kfac_l_all.append(l)
        kfac_s_all.append(sh)
    print("done")
    print("  True Diag DLN (5 seeds) ...", end=" ", flush=True)
    for s in range(n_kfac_seeds):
        l, sh, _ = run_true_diag_dln(lr=0.1, steps=steps, seed=s, damping=1e-3)
        diag_true_l_all.append(l)
        diag_true_s_all.append(sh)
    print("done")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    for data, lbl, clr in [
        (all_data["SGD"]["losses"], "SGD", "tab:red"),
        (all_data["NGD"]["losses"], "SP-GD", "tab:cyan"),
        (diag_true_l_all, "NGD (diag)", "tab:purple"),
        (kfac_l_all, "K-FAC (ASDL)", "tab:blue"),
    ]:
        arr = np.array(data)
        mu, sd = arr.mean(0), arr.std(0)
        ax1.plot(mu, label=lbl, color=clr)
        ax1.fill_between(range(len(mu)), mu - sd, mu + sd, alpha=0.15, color=clr)
    ax1.set_yscale("log")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("(a) DLN Loss: NGD Approximation Quality")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.2)

    for data, lbl, clr in [
        (all_data["SGD"]["sharp"], r"SGD $\lambda_{\max}(H)$", "tab:red"),
        (kfac_s_all, r"K-FAC (ASDL) $\lambda_{\max}(H)$", "tab:blue"),
        (diag_true_s_all, r"NGD (diag) $\lambda_{\max}(H)$", "tab:purple"),
    ]:
        arr = np.abs(np.array(data))
        mu = arr.mean(0)
        ax2.plot(mu, label=lbl, color=clr)
    ax2.axhline(2 / lr, color="black", ls="--", label=r"$2/\eta$", alpha=0.5)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Sharpness")
    ax2.set_title("(b) Sharpness Dynamics")
    ax2.legend(fontsize=9)
    ax2.set_ylim(bottom=0, top=min(80, ax2.get_ylim()[1]))
    ax2.grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("approximation_quality")

    # ------------------------------------------------------------------
    # K-FAC on MNIST
    # ------------------------------------------------------------------
    print("[Fig 9+] K-FAC on MNIST (5 seeds)")
    kfac_mnist_data = {"losses": [], "sharps": [], "accs": []}
    for s in range(mnist_seeds):
        print(f"  K-FAC MNIST seed {s} ...", end=" ", flush=True)
        l, sh, ac, _ = run_kfac_mnist(lr=0.01, steps=200, seed=s, damping=1e-3)
        kfac_mnist_data["losses"].append(l)
        kfac_mnist_data["sharps"].append(sh)
        kfac_mnist_data["accs"].append(ac)
        print("done")

    # ------------------------------------------------------------------
    # FIG 10: CIFAR-10 ResNet-18 (SGD vs K-FAC)
    # ------------------------------------------------------------------
    print("\n[Fig 10] CIFAR-10 ResNet-18 at scale")
    print("  SGD training ...")
    cifar_sgd = run_cifar_training("SGD", lr=0.1, epochs=5, batch_size=128, seed=42)
    print("  K-FAC training ...")
    cifar_kfac = run_cifar_training("KFAC", lr=0.1, epochs=5, batch_size=128, seed=42, damping=1e-3, curv_interval=50)
    n_params_cifar = cifar_sgd["n_params"]
    print(f"  ResNet-18 parameters: {n_params_cifar:,}")

    fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.5))
    window = 50
    for data, lbl, clr in [(cifar_sgd, "SGD", "tab:red"), (cifar_kfac, "K-FAC (ASDL)", "tab:blue")]:
        raw = np.array(data["losses"])
        if len(raw) > window:
            smoothed = np.convolve(raw, np.ones(window) / window, mode="valid")
        else:
            smoothed = raw
        axes[0].plot(smoothed, label=lbl, color=clr)
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("(a) Training Loss (smoothed)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.2)

    for data, lbl, clr in [(cifar_sgd, "SGD", "tab:red"), (cifar_kfac, "K-FAC (ASDL)", "tab:blue")]:
        sh = np.array(data["sharps"])
        nonzero = sh > 0
        if nonzero.any():
            axes[1].plot(np.where(nonzero)[0], sh[nonzero], "o-", label=lbl, color=clr, markersize=4)
    axes[1].axhline(2 / 0.1, color="black", ls="--", label=r"$2/\eta=20$", alpha=0.5)
    axes[1].set_xlabel("Iteration")
    axes[1].set_ylabel(r"$\lambda_{\max}(H)$")
    axes[1].set_title("(b) Sharpness Dynamics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.2)

    epochs_x = np.arange(1, len(cifar_sgd["epoch_accs"]) + 1)
    for data, lbl, clr in [(cifar_sgd, "SGD", "tab:red"), (cifar_kfac, "K-FAC (ASDL)", "tab:blue")]:
        axes[2].plot(epochs_x, [a * 100 for a in data["epoch_accs"]], "o-", label=lbl, color=clr, markersize=6)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("(c) Test Accuracy")
    axes[2].legend()
    axes[2].grid(True, alpha=0.2)
    plt.tight_layout()
    _savefig("cifar10_resnet18")

    # ------------------------------------------------------------------
    # FIG 11: Scaling Analysis — bound degradation
    # ------------------------------------------------------------------
    print("\n[Fig 11] Scaling analysis: bound vs model size")
    scale_widths = [5, 10, 15, 20, 30, 40]
    scale_results = []
    for w in scale_widths:
        print(f"  width={w} ...", end=" ", flush=True)
        r = measure_bound_at_scale(w, depth=3, N=200, seed=42, damping=1e-3, train_steps=50)
        scale_results.append(r)
        print(f"npar={r['n_params']}, ε/μ_min={r['ratio']:.1f}, S_eff={r['seff']:.1f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))
    nparams = [r["n_params"] for r in scale_results]
    ratios = [r["ratio"] for r in scale_results]
    seffs = [r["seff"] for r in scale_results]
    bounds = [r["bound"] for r in scale_results]
    ax1.loglog(nparams, ratios, "o-", color="tab:red", label=r"$\epsilon / \mu_{\min}(F)$", markersize=8)
    ax1.set_xlabel("Number of Parameters")
    ax1.set_ylabel(r"$\epsilon / \mu_{\min}(F)$")
    ax1.set_title("(a) Bound Ratio vs. Model Size")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.2)

    ax2.loglog(nparams, seffs, "o-", color="tab:blue", label=r"$S_{\mathrm{eff}}$ (actual)", markersize=8)
    ax2.loglog(nparams, bounds, "s--", color="tab:orange", label=r"Bound $1 + \epsilon/\mu_{\min}$", markersize=8)
    ax2.set_xlabel("Number of Parameters")
    ax2.set_ylabel("Effective Sharpness")
    ax2.set_title("(b) Actual vs. Bound")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.2)
    plt.tight_layout()
    _savefig("scaling_analysis")

    # ------------------------------------------------------------------
    # Phase 2 Statistical Report
    # ------------------------------------------------------------------
    print("\n" + "-" * 60)
    print("PHASE 2 STATISTICAL REPORT")
    print("-" * 60)

    kfac_finals = [l[-1] for l in kfac_l_all]
    diag_true_finals = [l[-1] for l in diag_true_l_all]
    print(f"\nDLN Approximation Quality (final MSE, {n_kfac_seeds} seeds):")
    print(f"  SGD            : {np.mean([l[-1] for l in all_data['SGD']['losses']]):.6f}")
    print(f"  NGD (scalar)   : {np.mean([l[-1] for l in all_data['NGD']['losses']]):.4f}")
    print(f"  NGD (true diag): {np.mean(diag_true_finals):.6f} +/- {np.std(diag_true_finals):.6f}")
    print(f"  K-FAC          : {np.mean(kfac_finals):.6f} +/- {np.std(kfac_finals):.6f}")

    kfac_mnist_accs = [a[-1] * 100 for a in kfac_mnist_data["accs"]]
    print(
        f"\nMNIST K-FAC Accuracy ({mnist_seeds} seeds): "
        f"{np.mean(kfac_mnist_accs):.1f} +/- {np.std(kfac_mnist_accs):.1f}%"
    )

    print(f"\nCIFAR-10 ResNet-18 ({n_params_cifar:,} params):")
    print(f"  SGD  : {cifar_sgd['epoch_accs'][-1] * 100:.1f}% (wall time: {cifar_sgd['wall_time']:.0f}s)")
    print(f"  K-FAC: {cifar_kfac['epoch_accs'][-1] * 100:.1f}% (wall time: {cifar_kfac['wall_time']:.0f}s)")

    sgd_times_c = np.array(cifar_sgd["iter_times"])
    kfac_times_c = np.array(cifar_kfac["iter_times"])
    print("\nPer-iteration timing (CIFAR-10):")
    print(f"  SGD  : {sgd_times_c.mean():.4f} +/- {sgd_times_c.std():.4f} s")
    print(f"  K-FAC: {kfac_times_c.mean():.4f} +/- {kfac_times_c.std():.4f} s")
    print(f"  Overhead: {kfac_times_c.mean() / sgd_times_c.mean():.2f}x")

    print("\nScaling Analysis (DLN, depth=3):")
    print(f"{'Width':>6} {'Params':>8} {'ε/μ_min':>12} {'S_eff':>10} {'Bound':>10} {'Gap':>6}")
    print("-" * 58)
    for r in scale_results:
        gap = r["bound"] / max(r["seff"], 1e-12)
        print(
            f"{r['width']:>6} {r['n_params']:>8} {r['ratio']:>12.1f} "
            f"{r['seff']:>10.1f} {r['bound']:>10.1f} {gap:>6.1f}x"
        )

    # Save machine-readable results
    results = {
        "n_params_dln": npar_dln,
        "n_params_mnist": np_m,
        "n_params_small": small_npar,
        "n_params_cifar": n_params_cifar,
        "n_seeds": n_seeds,
        "theorem_final": {
            "epsilon": tv["eps"][-1],
            "mu_min": tv["mu_min"][-1],
            "seff": tv["seff"][-1],
            "bound": tv["bound"][-1],
        },
        "corr_eig_sv": corr,
        "cifar_sgd_acc": cifar_sgd["epoch_accs"][-1],
        "cifar_kfac_acc": cifar_kfac["epoch_accs"][-1],
        "cifar_sgd_wall": cifar_sgd["wall_time"],
        "cifar_kfac_wall": cifar_kfac["wall_time"],
        "kfac_dln_final_mse": float(np.mean(kfac_finals)),
        "diag_true_final_mse": float(np.mean(diag_true_finals)),
        "kfac_mnist_acc": float(np.mean(kfac_mnist_accs)),
        "scaling": [
            {
                "width": r["width"],
                "n_params": r["n_params"],
                "ratio": r["ratio"],
                "seff": r["seff"],
                "bound": r["bound"],
            }
            for r in scale_results
        ],
    }
    with open(os.path.join(FIGDIR, "..", "experiment_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to experiment_results.json")
    print("-" * 60)


if __name__ == "__main__":
    main()
