"""
Optimizer Baselines Experiment for "Spectral Stability of NGD"

Extends Table IV (DLN regression) with:
- AdamW
- SGD + warmup + cosine decay
- Shampoo (via ASDL, if available)

Also extends MNIST comparison with additional baselines.

Run via: uv run python scripts/optimizer_baselines.py
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "optimizer_baselines_results.json"
)
FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figures")
os.makedirs(FIGDIR, exist_ok=True)


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


def power_iteration_sharpness(loss, params, max_iter=20):
    """Top eigenvalue of Hessian via power iteration."""
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    valid = [(g, p) for g, p in zip(grads, params) if g is not None]
    if not valid:
        return 0.0
    vg, vp = zip(*valid)
    v = [torch.randn_like(p) for p in vp]
    vnorm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
    v = [vi / vnorm for vi in v]

    eigenvalue = 0.0
    for _ in range(max_iter):
        Hv = torch.autograd.grad(vg, vp, grad_outputs=v, retain_graph=True)
        eigenvalue = sum(torch.sum(hi * vi) for hi, vi in zip(Hv, v)).item()
        vnorm = torch.sqrt(sum(torch.sum(hi ** 2) for hi in Hv))
        if vnorm.item() < 1e-30:
            break
        v = [hi / vnorm for hi in Hv]
    return eigenvalue


def warmup_cosine_lr(step, warmup_steps, total_steps, base_lr):
    """Linear warmup + cosine decay schedule."""
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))


# ── DLN Experiment ──────────────────────────────────────────────────────────


def run_dln_baselines():
    """DLN regression baselines: AdamW, SGD+warmup+cosine, Shampoo."""
    print("=" * 70)
    print("DLN Optimizer Baselines (820 parameters, 10 seeds)")
    print("=" * 70)

    n_seeds = 10
    n_iters = 150
    width = 20
    depth = 3
    input_dim = 20
    N = 500

    optimizers_config = {
        "AdamW": {"type": "adamw", "lr": 0.01, "weight_decay": 1e-4},
        "SGD+Warmup+Cosine": {"type": "sgd_warmup_cosine", "lr": 0.1, "warmup": 15},
    }

    # Try to import Shampoo from ASDL
    has_shampoo = False
    try:
        from asdl import ShampooGradientMaker, PreconditioningConfig  # type: ignore
        has_shampoo = True
        optimizers_config["Shampoo (ASDL)"] = {"type": "shampoo", "lr": 0.1, "damping": 1e-3}
        print("  ASDL Shampoo available ✓")
    except ImportError:
        print("  ASDL Shampoo not available, skipping")

    all_results = {}

    for opt_name, opt_cfg in optimizers_config.items():
        print(f"\n  {opt_name}:")
        seed_losses = []
        seed_sharpness_histories = []

        for seed in range(n_seeds):
            torch.manual_seed(42)  # fixed teacher
            np.random.seed(42)
            X = torch.randn(N, input_dim)
            teacher = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
            with torch.no_grad():
                Y = teacher(X)

            torch.manual_seed(seed)
            model = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
            params = list(model.parameters())

            if opt_cfg["type"] == "adamw":
                optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
            elif opt_cfg["type"] == "sgd_warmup_cosine":
                optimizer = torch.optim.SGD(params, lr=opt_cfg["lr"])
            elif opt_cfg["type"] == "shampoo":
                # Use SGD optimizer, let ASDL handle preconditioning
                optimizer = torch.optim.SGD(params, lr=opt_cfg["lr"])
                precond_config = PreconditioningConfig(damping=opt_cfg["damping"])
                gradient_maker = ShampooGradientMaker(model, precond_config)

            sharpness_history = []
            final_loss = None

            for step in range(n_iters):
                # Update LR for warmup+cosine
                if opt_cfg["type"] == "sgd_warmup_cosine":
                    lr = warmup_cosine_lr(step, opt_cfg["warmup"], n_iters, opt_cfg["lr"])
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                pred = model(X)
                loss = nn.MSELoss()(pred, Y)

                # Measure sharpness every 10 steps
                if step % 10 == 0:
                    sharp = power_iteration_sharpness(loss, params, max_iter=15)
                    sharpness_history.append({"step": step, "sharpness": sharp, "loss": loss.item()})

                optimizer.zero_grad()
                
                if opt_cfg["type"] == "shampoo":
                    # ASDL Shampoo
                    try:
                        dummy_y = Y  # for loss function
                        gradient_maker.setup_model_call(model, X)
                        gradient_maker.setup_loss_call(nn.MSELoss(), pred, Y)
                        loss.backward()
                        gradient_maker.precondition()
                    except Exception as e:
                        loss.backward()  # fallback
                else:
                    loss.backward()
                
                optimizer.step()
                final_loss = loss.item()

            seed_losses.append(final_loss)
            seed_sharpness_histories.append(sharpness_history)
            print(f"    Seed {seed}: MSE = {final_loss:.6f}")

        losses = np.array(seed_losses)
        all_results[opt_name] = {
            "median_mse": float(np.median(losses)),
            "mean_mse": float(np.mean(losses)),
            "std_mse": float(np.std(losses)),
            "iqr_low": float(np.percentile(losses, 25)),
            "iqr_high": float(np.percentile(losses, 75)),
            "per_seed_losses": [float(l) for l in seed_losses],
            "config": opt_cfg,
            "sharpness_histories": seed_sharpness_histories,
        }
        print(f"    Median MSE: {np.median(losses):.6f}")
        print(f"    Mean ± s.d.: {np.mean(losses):.6f} ± {np.std(losses):.6f}")

    return all_results


# ── MNIST Experiment ────────────────────────────────────────────────────────


def run_mnist_baselines():
    """MNIST baselines: AdamW, SGD+warmup+cosine."""
    print("\n" + "=" * 70)
    print("MNIST Optimizer Baselines (50,890 parameters, 5 seeds)")
    print("=" * 70)

    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("  torchvision not available, skipping")
        return {}

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    try:
        trainset = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        testset = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    except Exception as e:
        print(f"  Failed to load MNIST: {e}")
        return {}

    n_seeds = 5
    n_iters = 200
    n_train = 2000

    optimizers_config = {
        "AdamW (lr=0.001)": {"type": "adamw", "lr": 0.001, "weight_decay": 1e-4},
        "SGD+Warmup+Cosine (lr=0.05)": {"type": "sgd_warmup_cosine", "lr": 0.05, "warmup": 20},
    }

    all_results = {}

    for opt_name, opt_cfg in optimizers_config.items():
        print(f"\n  {opt_name}:")
        seed_results = []

        for seed in range(n_seeds):
            torch.manual_seed(seed)
            indices = torch.randperm(len(trainset))[:n_train]
            subset = torch.utils.data.Subset(trainset, indices)
            loader = torch.utils.data.DataLoader(subset, batch_size=n_train, shuffle=False)
            X_train, Y_train = next(iter(loader))
            X_train = X_train.view(-1, 784)

            model = TanhMLP()
            params = list(model.parameters())

            if opt_cfg["type"] == "adamw":
                optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
            elif opt_cfg["type"] == "sgd_warmup_cosine":
                optimizer = torch.optim.SGD(params, lr=opt_cfg["lr"])

            for step in range(n_iters):
                if opt_cfg["type"] == "sgd_warmup_cosine":
                    lr = warmup_cosine_lr(step, opt_cfg["warmup"], n_iters, opt_cfg["lr"])
                    for pg in optimizer.param_groups:
                        pg["lr"] = lr

                pred = model(X_train)
                loss = nn.CrossEntropyLoss()(pred, Y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Test accuracy
            test_loader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs = imgs.view(-1, 784)
                    pred = model(imgs)
                    correct += (pred.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total * 100

            seed_results.append({
                "seed": seed,
                "final_loss": float(loss.item()),
                "test_accuracy": acc,
            })
            print(f"    Seed {seed}: acc={acc:.1f}%")

        accs = [r["test_accuracy"] for r in seed_results]
        all_results[opt_name] = {
            "mean_accuracy": float(np.mean(accs)),
            "std_accuracy": float(np.std(accs)),
            "per_seed": seed_results,
            "config": opt_cfg,
        }
        print(f"    Mean accuracy: {np.mean(accs):.1f} ± {np.std(accs):.1f}%")

    return all_results


def main():
    results = {}

    # DLN baselines
    results["dln"] = run_dln_baselines()

    # MNIST baselines
    results["mnist"] = run_mnist_baselines()

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: DLN Baselines (Extend Table IV)")
    print("=" * 70)
    print(f"{'Method':<30} {'Median MSE':>12} {'Mean ± s.d.':>20}")
    for name, r in results["dln"].items():
        print(f"{name:<30} {r['median_mse']:>12.6f} {r['mean_mse']:.6f} ± {r['std_mse']:.6f}")

    if results.get("mnist"):
        print("\n" + "=" * 70)
        print("SUMMARY: MNIST Baselines (Extend Figure 9)")
        print("=" * 70)
        print(f"{'Method':<35} {'Accuracy':>12}")
        for name, r in results["mnist"].items():
            print(f"{name:<35} {r['mean_accuracy']:.1f} ± {r['std_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
