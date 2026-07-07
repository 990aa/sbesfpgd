"""
SOPHIA Baselines Experiment for "Spectral Stability of NGD"

Implements SOPHIA (Liu et al., 2024) from scratch using Hutchinson's
randomized trace estimator for the diagonal Hessian approximation.

Extends Table IV (DLN regression) and Table XIV (MNIST) with SOPHIA results.

Run via: uv run python scripts/sophia_baselines.py
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "sophia_baselines_results.json"
)


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


class Sophia:
    """
    SOPHIA optimizer (Liu et al., ICLR 2024).
    
    Algorithm:
    1. Compute gradient g_t
    2. Estimate diagonal Hessian h_t via Hutchinson
    3. Update EMAs:
       m_t = β1 * m_{t-1} + (1-β1) * g_t
       h_ema_t = β2 * h_ema_{t-1} + (1-β2) * h_t
    4. Update:
       θ_{t+1} = θ_t - η * clip(m_t / max(h_ema_t, ε), -c, c)
    """
    
    def __init__(self, params, lr=0.01, beta1=0.9, beta2=0.99, eps=1e-4,
                 rho=0.04, weight_decay=0.0, n_hutchinson=1):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.rho = rho  # clip threshold
        self.weight_decay = weight_decay
        self.n_hutchinson = n_hutchinson
        
        self.m = [torch.zeros_like(p) for p in self.params]
        self.h = [torch.zeros_like(p) for p in self.params]
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def _hutchinson_diag_hessian(self, loss):
        grads = torch.autograd.grad(
            loss, self.params, create_graph=True, retain_graph=True
        )
        
        diag_estimates = [torch.zeros_like(p) for p in self.params]
        
        for _ in range(self.n_hutchinson):
            z = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in self.params]
            gz = sum(torch.sum(g * zi) for g, zi in zip(grads, z))
            Hz = torch.autograd.grad(gz, self.params, retain_graph=True)
            for i, (hi, zi) in enumerate(zip(Hz, z)):
                diag_estimates[i] += zi * hi
        
        for i in range(len(diag_estimates)):
            diag_estimates[i] /= self.n_hutchinson
        
        return grads, diag_estimates
    
    def step(self, loss):
        grads, diag_h = self._hutchinson_diag_hessian(loss)
        
        for i, (p, g, d) in enumerate(zip(self.params, grads, diag_h)):
            g_detach = g.detach()
            d_detach = d.detach().abs()
            
            if self.weight_decay > 0:
                p.data.mul_(1 - self.lr * self.weight_decay)
            
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_detach
            self.h[i] = self.beta2 * self.h[i] + (1 - self.beta2) * d_detach
            
            # Sophia update: clip(m / max(h, eps), -rho, rho)
            ratio = self.m[i] / torch.clamp(self.h[i], min=self.eps)
            update = torch.clamp(ratio, -self.rho, self.rho)
            
            p.data -= self.lr * update


def power_iteration_sharpness(loss, params, max_iter=20):
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


# ── DLN Experiment ──────────────────────────────────────────────────────────

def run_dln_sophia():
    print("=" * 70)
    print("DLN SOPHIA Baseline (820 parameters, 10 seeds)")
    print("=" * 70)

    n_seeds = 10
    n_iters = 150
    width = 20
    depth = 3
    input_dim = 20
    N = 500

    configs = {
        "SOPHIA (lr=0.01)": {"lr": 0.01, "beta1": 0.9, "beta2": 0.99, "eps": 1e-4, "rho": 0.04},
        "SOPHIA (lr=0.05)": {"lr": 0.05, "beta1": 0.9, "beta2": 0.99, "eps": 1e-4, "rho": 0.04},
    }

    all_results = {}

    for opt_name, cfg in configs.items():
        print(f"\n  {opt_name}:")
        seed_losses = []

        for seed in range(n_seeds):
            torch.manual_seed(42)
            np.random.seed(42)
            X = torch.randn(N, input_dim)
            teacher = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
            with torch.no_grad():
                Y = teacher(X)

            torch.manual_seed(seed)
            model = DeepLinearNet(depth=depth, width=width, input_dim=input_dim)
            params = list(model.parameters())

            optimizer = Sophia(
                params, lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"],
                eps=cfg["eps"], rho=cfg["rho"]
            )

            final_loss = None

            for step in range(n_iters):
                pred = model(X)
                loss = nn.MSELoss()(pred, Y)
                
                optimizer.zero_grad()
                optimizer.step(loss)

                final_loss = loss.item()

            seed_losses.append(final_loss)
            print(f"    Seed {seed}: MSE = {final_loss:.6f}")

        losses = np.array(seed_losses)
        all_results[opt_name] = {
            "median_mse": float(np.median(losses)),
            "mean_mse": float(np.mean(losses)),
            "std_mse": float(np.std(losses)),
            "per_seed_losses": [float(l) for l in seed_losses],
            "config": cfg,
        }
        print(f"    Median MSE: {np.median(losses):.6f}")

    return all_results


# ── MNIST Experiment ────────────────────────────────────────────────────────

def run_mnist_sophia():
    print("\n" + "=" * 70)
    print("MNIST SOPHIA Baseline (50,890 parameters, 5 seeds)")
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

    cfg = {"lr": 0.05, "beta1": 0.9, "beta2": 0.99, "eps": 1e-4, "rho": 0.04}

    print(f"\n  SOPHIA (lr={cfg['lr']}):")
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

        optimizer = Sophia(
            params, lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"],
            eps=cfg["eps"], rho=cfg["rho"]
        )

        t0 = time.time()
        for step in range(n_iters):
            pred = model(X_train)
            loss = nn.CrossEntropyLoss()(pred, Y_train)
            optimizer.zero_grad()
            optimizer.step(loss)

        elapsed = time.time() - t0

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
    result = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_seed": seed_results,
        "config": cfg,
    }
    print(f"    Mean accuracy: {np.mean(accs):.1f} ± {np.std(accs):.1f}%")

    return {"SOPHIA (lr=0.05)": result}


def main():
    results = {}
    results["dln"] = run_dln_sophia()
    results["mnist"] = run_mnist_sophia()

    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()
