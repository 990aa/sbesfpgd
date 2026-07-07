"""
ADAHESSIAN Baselines Experiment for "Spectral Stability of NGD"

Implements ADAHESSIAN (Yao et al., 2021) from scratch using Hutchinson's
randomized trace estimator for the diagonal Hessian approximation.

Extends Table IV (DLN regression) and Table XIV (MNIST) with ADAHESSIAN results.

Run via: uv run python scripts/adahessian_baselines.py
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn

RESULTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "adahessian_baselines_results.json"
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


class AdaHessian:
    """
    ADAHESSIAN optimizer (Yao et al., AAAI 2021).
    
    Uses Hutchinson's trace estimator to approximate diagonal Hessian elements,
    then uses these as per-parameter adaptive learning rates (like Adam but with
    true curvature information instead of gradient-magnitude-based scaling).
    
    Algorithm:
    1. Compute gradient g_t
    2. Estimate diagonal Hessian D_t via Hutchinson: D_t ≈ z ⊙ (Hz) where z ~ Rademacher
    3. Update exponential moving averages: m_t = β1*m_{t-1} + (1-β1)*g_t
                                            v_t = β2*v_{t-1} + (1-β2)*D_t^2
    4. Bias-correct: m_hat = m_t/(1-β1^t), v_hat = v_t/(1-β2^t)
    5. Update: θ_{t+1} = θ_t - η * m_hat / (sqrt(v_hat) + ε)
    
    The key difference from Adam is step 2: Adam uses g_t^2 (gradient magnitude)
    while ADAHESSIAN uses D_t^2 (diagonal Hessian), providing true curvature info.
    """
    
    def __init__(self, params, lr=0.1, beta1=0.9, beta2=0.999, eps=1e-4,
                 weight_decay=0.0, n_hutchinson=1):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.n_hutchinson = n_hutchinson  # number of Hutchinson samples
        self.step_count = 0
        
        # Initialize state
        self.m = [torch.zeros_like(p) for p in self.params]  # first moment (gradient)
        self.v = [torch.zeros_like(p) for p in self.params]  # second moment (Hessian diag)
    
    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
    
    def _hutchinson_diag_hessian(self, loss):
        """
        Estimate diagonal Hessian via Hutchinson's method.
        
        For each Rademacher vector z:
          Hz = autograd(grad · z, params)   [Hessian-vector product]
          diag_estimate = z ⊙ Hz            [diagonal extraction]
        
        Average over n_hutchinson samples.
        """
        # Get gradients with graph retained for Hessian computation
        grads = torch.autograd.grad(
            loss, self.params, create_graph=True, retain_graph=True
        )
        
        diag_estimates = [torch.zeros_like(p) for p in self.params]
        
        for _ in range(self.n_hutchinson):
            # Rademacher random vectors
            z = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in self.params]
            
            # Compute gradient-vector product: g^T z
            gz = sum(torch.sum(g * zi) for g, zi in zip(grads, z))
            
            # Hessian-vector product: Hz = d(g^T z)/dθ
            Hz = torch.autograd.grad(
                gz, self.params, retain_graph=True
            )
            
            # Diagonal estimate: diag(H) ≈ z ⊙ Hz
            for i, (hi, zi) in enumerate(zip(Hz, z)):
                diag_estimates[i] += zi * hi
        
        # Average over Hutchinson samples
        for i in range(len(diag_estimates)):
            diag_estimates[i] /= self.n_hutchinson
        
        return grads, diag_estimates
    
    def step(self, loss):
        """
        Perform one ADAHESSIAN update step.
        
        Args:
            loss: The loss tensor (must have create_graph=True compatible computation)
        """
        self.step_count += 1
        
        # Get gradients and diagonal Hessian estimates
        grads, diag_h = self._hutchinson_diag_hessian(loss)
        
        for i, (p, g, d) in enumerate(zip(self.params, grads, diag_h)):
            # Detach for parameter updates
            g_detach = g.detach()
            d_detach = d.detach().abs()  # Use absolute value of Hessian diagonal
            
            # Weight decay
            if self.weight_decay > 0:
                g_detach = g_detach + self.weight_decay * p.data
            
            # Update exponential moving averages
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g_detach
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * d_detach.pow(2)
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            
            # Parameter update
            p.data -= self.lr * m_hat / (v_hat.sqrt() + self.eps)


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


# ── DLN Experiment ──────────────────────────────────────────────────────────


def run_dln_adahessian():
    """DLN regression with ADAHESSIAN (820 parameters, 10 seeds)."""
    print("=" * 70)
    print("DLN ADAHESSIAN Baseline (820 parameters, 10 seeds)")
    print("=" * 70)

    n_seeds = 10
    n_iters = 150
    width = 20
    depth = 3
    input_dim = 20
    N = 500

    # ADAHESSIAN hyperparameters tuned for DLN
    configs = {
        "ADAHESSIAN (lr=0.01)": {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-4, "n_hutchinson": 1},
        "ADAHESSIAN (lr=0.001)": {"lr": 0.001, "beta1": 0.9, "beta2": 0.999, "eps": 1e-4, "n_hutchinson": 1},
    }

    all_results = {}

    for opt_name, cfg in configs.items():
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

            optimizer = AdaHessian(
                params, lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"],
                eps=cfg["eps"], n_hutchinson=cfg["n_hutchinson"]
            )

            sharpness_history = []
            final_loss = None

            for step in range(n_iters):
                pred = model(X)
                loss = nn.MSELoss()(pred, Y)

                # Measure sharpness every 10 steps
                if step % 10 == 0:
                    sharp = power_iteration_sharpness(loss, params, max_iter=15)
                    sharpness_history.append({"step": step, "sharpness": sharp, "loss": loss.item()})

                optimizer.zero_grad()
                optimizer.step(loss)

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
            "config": cfg,
            "sharpness_histories": seed_sharpness_histories,
        }
        print(f"    Median MSE: {np.median(losses):.6f}")
        print(f"    Mean ± s.d.: {np.mean(losses):.6f} ± {np.std(losses):.6f}")

    return all_results


# ── MNIST Experiment ────────────────────────────────────────────────────────


def run_mnist_adahessian():
    """MNIST with ADAHESSIAN (50,890 parameters, 5 seeds)."""
    print("\n" + "=" * 70)
    print("MNIST ADAHESSIAN Baseline (50,890 parameters, 5 seeds)")
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

    # ADAHESSIAN config for MNIST
    cfg = {"lr": 0.01, "beta1": 0.9, "beta2": 0.999, "eps": 1e-4, "n_hutchinson": 1}

    print(f"\n  ADAHESSIAN (lr={cfg['lr']}):")
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

        optimizer = AdaHessian(
            params, lr=cfg["lr"], beta1=cfg["beta1"], beta2=cfg["beta2"],
            eps=cfg["eps"], n_hutchinson=cfg["n_hutchinson"]
        )

        t0 = time.time()
        for step in range(n_iters):
            pred = model(X_train)
            loss = nn.CrossEntropyLoss()(pred, Y_train)
            optimizer.zero_grad()
            optimizer.step(loss)

            if step % 50 == 0:
                print(f"      Seed {seed}, step {step}: loss={loss.item():.4f}")

        elapsed = time.time() - t0

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
            "wall_time": elapsed,
        })
        print(f"    Seed {seed}: acc={acc:.1f}%, time={elapsed:.1f}s")

    accs = [r["test_accuracy"] for r in seed_results]
    result = {
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy": float(np.std(accs)),
        "per_seed": seed_results,
        "config": cfg,
    }
    print(f"    Mean accuracy: {np.mean(accs):.1f} ± {np.std(accs):.1f}%")

    return {"ADAHESSIAN (lr=0.01)": result}


def main():
    results = {}

    # DLN baselines
    results["dln"] = run_dln_adahessian()

    # MNIST baselines
    results["mnist"] = run_mnist_adahessian()

    # Save
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nAll results saved to {RESULTS_FILE}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: DLN ADAHESSIAN Baselines (Extend Table IV)")
    print("=" * 70)
    print(f"{'Method':<30} {'Median MSE':>12} {'Mean ± s.d.':>20}")
    for name, r in results["dln"].items():
        print(f"{name:<30} {r['median_mse']:>12.6f} {r['mean_mse']:.6f} ± {r['std_mse']:.6f}")

    if results.get("mnist"):
        print("\n" + "=" * 70)
        print("SUMMARY: MNIST ADAHESSIAN Baselines (Extend Table XIV)")
        print("=" * 70)
        print(f"{'Method':<35} {'Accuracy':>12}")
        for name, r in results["mnist"].items():
            print(f"{name:<35} {r['mean_accuracy']:.1f} ± {r['std_accuracy']:.1f}%")


if __name__ == "__main__":
    main()
