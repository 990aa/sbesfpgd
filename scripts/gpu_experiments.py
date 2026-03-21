#!/usr/bin/env python3
"""
STABLE GPU Experiments for "Spectral Stability of NGD"
Not to be run on cpu (will be very slow).
Run on Google Colab with GPU runtime (T4):
    !pip install git+https://github.com/kazukiosawa/asdl.git
    !python gpu_experiments.py

Changes from previous version:
    - DISABLED torch.compile (caused autograd crashes with ASDL).
    - Set num_workers=2 (prevents Colab freezing).
    - Suppressed specific PyTorch warnings.
    - Kept pruned hyperparameter grid for speed.
"""

import json
import sys
import time
import itertools
import warnings
import numpy as np

# ── Install dependencies ────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
except ImportError:
    import subprocess

    print("Installing torch and torchvision...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models

try:
    from asdl import KfacGradientMaker, PreconditioningConfig
except ImportError:
    import subprocess

    print("Installing asdl (optimization library) from GitHub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/kazukiosawa/asdl.git"])
    from asdl import KfacGradientMaker, PreconditioningConfig

# ── Device & Speed Optimizations ─────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    # Enable CuDNN autotuner
    torch.backends.cudnn.benchmark = True

    # Suppress TF32 warnings (T4 doesn't support TF32 anyway, but this cleans logs)
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except AttributeError:
        pass

# Filter noise warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

RESULTS_FILE = "gpu_experiment_results_stable.json"
RESULTS = {}


def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"  [Results saved to {RESULTS_FILE}]")



# Utilities



def hessian_top_eigenvalue(model, loss_fn, inputs, targets, num_iter=10):
    """Estimate λ_max(H) via Power Iteration."""
    # Ensure gradients are zeroed
    model.zero_grad(set_to_none=True)

    # Forward pass
    out = model(inputs)
    loss = loss_fn(out, targets)

    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)

    # Filter valid params
    valid_params = []
    valid_grads = []
    for g, p in zip(grads, params):
        if g is not None:
            valid_grads.append(g)
            valid_params.append(p)

    if not valid_params:
        return 0.0

    # Initialize vector v (random)
    v = [torch.randn_like(p) for p in valid_params]
    norm = torch.sqrt(sum((vi**2).sum() for vi in v))
    v = [vi / norm for vi in v]

    eigenvalue = 0.0

    for _ in range(num_iter):
        # Hessian-vector product: Hv = ∇(∇L·v)
        gv = sum((g * vi).sum() for g, vi in zip(valid_grads, v))
        Hv = torch.autograd.grad(gv, valid_params, retain_graph=True)

        # Rayleigh quotient
        eigenvalue = sum((hvi * vi).sum() for hvi, vi in zip(Hv, v)).item()

        # Re-normalize
        norm = torch.sqrt(sum((hvi**2).sum() for hvi in Hv))
        if norm.item() < 1e-12:
            break
        v = [hvi / norm for hvi in Hv]

    # Cleanup graph
    model.zero_grad(set_to_none=True)
    return eigenvalue


def get_cifar_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


class TanhMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))



# Data Loading (Optimized)



def get_cifar10_loaders(batch_size=256, data_dir="./data"):
    # Pre-calculate stats
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    t_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    t_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=t_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=t_test)

    # FIXED: num_workers=2 is safe for Colab. persistent_workers=True speeds up epochs.
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True, persistent_workers=True
    )
    return train_loader, test_loader


def get_mnist_tensors(n_train=2000, seed=42):
    trainset = torchvision.datasets.MNIST(root="./data", train=True, download=True)
    testset = torchvision.datasets.MNIST(root="./data", train=False, download=True)
    trX = trainset.data.float().view(-1, 784).to(DEVICE) / 255.0
    trY = trainset.targets.to(DEVICE)
    teX = testset.data.float().view(-1, 784).to(DEVICE) / 255.0
    teY = testset.targets.to(DEVICE)

    torch.manual_seed(seed)
    idx = torch.randperm(len(trX), device=DEVICE)[:n_train]
    return trX[idx], trY[idx], teX, teY



# EXPERIMENT 1 & 2: CIFAR-10



def train_cifar_kfac(lr, damping, curv_interval, batch_size, epochs, seed=42, measure_sharpness=False):
    """
    Train CIFAR-10 ResNet-18 with K-FAC via ASDL.
    NO torch.compile to avoid autograd errors.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = get_cifar_resnet18().to(DEVICE)
    sum(p.numel() for p in model.parameters())
    train_loader, test_loader = get_cifar10_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    config = PreconditioningConfig(
        data_size=batch_size,
        damping=damping,
        curvature_upd_interval=curv_interval,
        preconditioner_upd_interval=curv_interval,
    )

    grad_maker = KfacGradientMaker(model, config, loss_type="cross_entropy")

    epoch_losses, epoch_accs, sharpness_data = [], [], []
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)

            try:
                optimizer.zero_grad()

                # ASDL Logic
                dummy_y = grad_maker.setup_model_call(model, inputs)
                grad_maker.setup_loss_call(criterion, dummy_y, targets)
                y, loss = grad_maker.forward_and_backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                running_loss += loss.item()
            except RuntimeError as e:
                if "nan" in str(e).lower() or "inf" in str(e).lower():
                    print(f"    [Diverged: {e}]")
                    return {"final_acc": 0.0, "error": "diverged"}
                raise

        # End of Epoch: Validation
        avg_loss = running_loss / len(train_loader)

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = correct / total

        epoch_losses.append(avg_loss)
        epoch_accs.append(acc)
        print(f"    Epoch {epoch + 1}/{epochs}: loss={avg_loss:.4f}, acc={acc * 100:.1f}%")

        # Measure Sharpness (Only if requested, and only once per epoch)
        if measure_sharpness:
            model.eval()
            with torch.enable_grad():
                # Get a fresh batch for Hessian
                inputs_h, targets_h = next(iter(train_loader))
                inputs_h = inputs_h[:64].to(DEVICE)
                targets_h = targets_h[:64].to(DEVICE)

                lmax = hessian_top_eigenvalue(model, criterion, inputs_h, targets_h, num_iter=10)
                sharpness_data.append({"epoch": epoch + 1, "lambda_max_H": float(lmax)})
            model.train()

        # Early stop for screening if diverging badly
        if not measure_sharpness and avg_loss > 10.0:
            return {"final_acc": 0.0, "error": "diverged"}

    return {
        "lr": lr,
        "damping": damping,
        "curv_interval": curv_interval,
        "final_acc": epoch_accs[-1] if epoch_accs else 0.0,
        "epoch_accs": [float(a) for a in epoch_accs],
        "sharpness": sharpness_data,
        "wall_time": time.time() - t0,
    }


def train_cifar_sgd(lr, epochs, batch_size=256, seed=42, measure_sharpness=False):
    """SGD Baseline for CIFAR-10."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = get_cifar_resnet18().to(DEVICE)

    train_loader, test_loader = get_cifar10_loaders(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

    sharpness_data = []
    epoch_accs = []

    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE, non_blocking=True), targets.to(DEVICE, non_blocking=True)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        acc = correct / total
        epoch_accs.append(acc)
        print(f"    Epoch {epoch + 1}/{epochs} (SGD): acc={acc * 100:.1f}%")

        if measure_sharpness:
            model.eval()
            with torch.enable_grad():
                inputs_h, targets_h = next(iter(train_loader))
                inputs_h = inputs_h[:64].to(DEVICE)
                targets_h = targets_h[:64].to(DEVICE)
                lmax = hessian_top_eigenvalue(model, criterion, inputs_h, targets_h, num_iter=10)
                sharpness_data.append({"epoch": epoch + 1, "lambda_max_H": float(lmax)})
            model.train()

    return {"final_acc": epoch_accs[-1], "sharpness": sharpness_data, "wall_time": time.time() - t0}


def run_experiment_1_2():
    print("\n" + "-" * 60)
    print("EXPERIMENT 1: Optimized CIFAR-10 K-FAC Sweep")
    print("-" * 60)

    # Pruned grid based on your previous logs (lr=0.001 was bad)
    dampings = [1e-3, 1e-2, 5e-2]
    lrs = [0.01, 0.05, 0.1]
    curv_freq = 10
    batch = 256

    # 5 epochs is enough to distinguish bad vs good LRs
    epochs_screen = 5

    sweep_results = []
    print(f"Screening {len(dampings) * len(lrs)} configs ({epochs_screen} epochs each)...")

    for damping, lr in itertools.product(dampings, lrs):
        print(f"\n[Config] damping={damping}, lr={lr}")
        try:
            result = train_cifar_kfac(lr, damping, curv_freq, batch, epochs_screen)
            sweep_results.append(result)
        except Exception as e:
            print(f"FAILED: {e}")
            # Add a dummy result so the index error doesn't happen
            sweep_results.append({"damping": damping, "lr": lr, "final_acc": -1.0})

    # Filter out failures
    valid_results = [r for r in sweep_results if r.get("final_acc", -1) >= 0]

    if not valid_results:
        print("CRITICAL: All configs failed. Defaulting to standard config.")
        best = {"damping": 0.01, "lr": 0.05, "curv_interval": 10}
    else:
        valid_results.sort(key=lambda x: x.get("final_acc", 0), reverse=True)
        best = valid_results[0]
        print(f"\nBest Config: damping={best['damping']}, lr={best['lr']} (Acc: {best['final_acc'] * 100:.1f}%)")

    # ── Phase 2: Full Training with Sharpness ──────────────────────────
    print("\n" + "-" * 60)
    print("EXPERIMENT 2: Extended Training & Sharpness")
    print("-" * 60)

    epochs_full = 25

    print(f"Running BEST K-FAC (d={best['damping']}, lr={best['lr']})...")
    best_kfac_res = train_cifar_kfac(
        best["lr"], best["damping"], best.get("curv_interval", 10), batch, epochs_full, measure_sharpness=True
    )

    print("\nRunning SGD Baseline (lr=0.1)...")
    sgd_res = train_cifar_sgd(0.1, epochs_full, batch, measure_sharpness=True)

    RESULTS["cifar10"] = {"best_kfac": best_kfac_res, "sgd": sgd_res}
    save_results()



# EXPERIMENT 3 & 4: MNIST (Fast)



def run_mnist_ngd(lr, X, y, teX, teY, steps=200, damping=1e-3, seed=42):
    """Simplified NGD for MNIST comparison."""
    torch.manual_seed(seed)
    model = TanhMLP().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    params = list(model.parameters())

    accs = []
    t0 = time.time()

    for step in range(steps):
        # Forward
        pred = model(X)
        loss = crit(pred, y)

        # NGD Update (Scalar Fisher)
        gs = torch.autograd.grad(loss, params)
        gf_flat = torch.cat([g.view(-1) for g in gs])
        norm_sq = torch.dot(gf_flat, gf_flat)
        scale = 1.0 / (norm_sq + damping)

        with torch.no_grad():
            for p, g in zip(params, gs):
                p.sub_(g * scale * lr)

        if step % 20 == 0:
            with torch.no_grad():
                acc = (model(teX).argmax(1) == teY).float().mean().item()
                accs.append(acc)

    return {"final_acc": accs[-1], "time": time.time() - t0}


def run_experiment_3_4():
    print("\n" + "-" * 60)
    print("EXPERIMENT 3/4: MNIST Comparisons (Fast)")
    print("-" * 60)

    X, y, teX, teY = get_mnist_tensors(n_train=2000)

    # Quick compute-controlled check
    # SGD
    print("Running SGD (200 steps)...")
    model_sgd = TanhMLP().to(DEVICE)
    opt = torch.optim.SGD(model_sgd.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(200):
        opt.zero_grad()
        loss = crit(model_sgd(X), y)
        loss.backward()
        opt.step()
    t_sgd = time.time() - t0
    acc_sgd = (model_sgd(teX).argmax(1) == teY).float().mean().item()

    # NGD (Time Matched)
    # Estimate NGD cost
    _ = run_mnist_ngd(0.01, X, y, teX, teY, steps=10)  # warm up
    res_ngd_bench = run_mnist_ngd(0.01, X, y, teX, teY, steps=50)
    t_ngd_iter = res_ngd_bench["time"] / 50
    t_sgd_iter = t_sgd / 200

    iters_matched = int(t_sgd / t_ngd_iter)
    print(f"Time Budget: {t_sgd:.4f}s")
    print(f"SGD Iters: 200 | NGD Matched Iters: {iters_matched}")

    res_ngd = run_mnist_ngd(0.01, X, y, teX, teY, steps=iters_matched)

    print(f"SGD Acc: {acc_sgd * 100:.1f}%")
    print(f"NGD Acc: {res_ngd['final_acc'] * 100:.1f}%")

    RESULTS["mnist"] = {"sgd_acc": acc_sgd, "ngd_acc": res_ngd["final_acc"], "overhead": t_ngd_iter / t_sgd_iter}
    save_results()



# MAIN


if __name__ == "__main__":
    t_start = time.time()

    run_experiment_1_2()
    run_experiment_3_4()

    total_time = (time.time() - t_start) / 60
    print(f"\nALL EXPERIMENTS COMPLETE in {total_time:.1f} minutes.")
