import json
import sys
import time
import os
import warnings
import subprocess
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

RESULTS_FILE = "misspec_scale_gpu_results.json"
RESULTS = {}


# ── Hugging Face CDN CIFAR-10 Loader ────────────────────────────────────────

def get_cifar10_datasets(transform_train, transform_test):
    """Directly loads CIFAR-10 via Hugging Face CDN (uoft-cs/cifar10)."""
    print("Loading CIFAR-10 directly via Hugging Face CDN (uoft-cs/cifar10)...")
    try:
        import datasets
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "datasets"])
        import datasets

    hf_ds = datasets.load_dataset("uoft-cs/cifar10")

    class HFCIFAR10Dataset(torch.utils.data.Dataset):
        def __init__(self, hf_split, transform=None):
            self.data = hf_split
            self.transform = transform
        def __len__(self):
            return len(self.data)
        def __getitem__(self, idx):
            item = self.data[idx]
            img, label = item['img'], item['label']
            if img.mode != 'RGB':
                img = img.convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, label

    trainset = HFCIFAR10Dataset(hf_ds['train'], transform=transform_train)
    testset = HFCIFAR10Dataset(hf_ds['test'], transform=transform_test)
    print("CIFAR-10 loaded successfully from Hugging Face!")
    return trainset, testset


# ── Models ──────────────────────────────────────────────────────────────────

class TanhMLP(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def get_cifar_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


# ── Vector Product Primitives (using .reshape() to handle non-contiguous memory)

def hvp_flat(model, X, Y, params, v_flat, loss_type="ce"):
    """Hessian-vector product H @ v (Pearlmutter's trick)."""
    v_list = []
    offset = 0
    for p in params:
        n = p.numel()
        v_list.append(v_flat[offset:offset+n].reshape(p.shape))
        offset += n

    pred = model(X)
    if loss_type == "ce":
        loss = nn.CrossEntropyLoss()(pred, Y)
    else:
        loss = nn.MSELoss()(pred, Y)

    grads = torch.autograd.grad(loss, params, create_graph=True)
    gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v_list))
    Hv = torch.autograd.grad(gv, params, retain_graph=False)
    return torch.cat([h.detach().reshape(-1) for h in Hv])


def ggn_vp_batched(model, X, Y, params, v_flat, loss_type="ce", batch_size=64):
    """Fast GGN-vector product via simultaneous batch Jacobian-Vector Product (JVP)."""
    d = v_flat.shape[0]
    result = torch.zeros(d, device=v_flat.device)
    N = X.shape[0]

    v_list = []
    offset = 0
    for p in params:
        n = p.numel()
        v_list.append(v_flat[offset:offset+n].reshape(p.shape))
        offset += n

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xi = X[start:end]

        pred = model(xi)  # (B, C)

        # JVP u = J @ v via dual variable alpha
        alpha = torch.zeros_like(pred, requires_grad=True)
        grads = torch.autograd.grad(pred, params, grad_outputs=alpha, create_graph=True, retain_graph=True)
        gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v_list))
        u = torch.autograd.grad(gv, alpha, retain_graph=True)[0]  # shape (B, C)

        if loss_type == "ce":
            p = torch.softmax(pred.detach(), dim=-1)
            sum_pu = (p * u).sum(dim=-1, keepdim=True)
            h = p * u - p * sum_pu
        elif loss_type == "mse":
            h = 2.0 * u
        else:
            raise ValueError(loss_type)

        # VJP J^T @ h in a single backward pass
        vjp_grads = torch.autograd.grad(pred, params, grad_outputs=h.detach(), retain_graph=False)

        offset = 0
        for vg in vjp_grads:
            n = vg.numel()
            result[offset:offset+n] += vg.detach().reshape(-1)
            offset += n

    result /= N
    return result


def empirical_fvp_batched(model, X, Y, params, v_flat, loss_type="ce", batch_size=64):
    """Fast Empirical Fisher-vector product via simultaneous directional derivatives."""
    d = v_flat.shape[0]
    result = torch.zeros(d, device=v_flat.device)
    N = X.shape[0]

    v_list = []
    offset = 0
    for p in params:
        n = p.numel()
        v_list.append(v_flat[offset:offset+n].reshape(p.shape))
        offset += n

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        xi, yi = X[start:end], Y[start:end]

        pred = model(xi)
        if loss_type == "ce":
            losses = nn.CrossEntropyLoss(reduction="none")(pred, yi)
        else:
            losses = nn.MSELoss(reduction="none")(pred, yi).mean(dim=-1)

        # Directional derivatives d_i = g_i^T v for all samples simultaneously
        beta = torch.ones_like(losses, requires_grad=True)
        weighted_loss = torch.sum(losses * beta)

        grads = torch.autograd.grad(weighted_loss, params, create_graph=True, retain_graph=True)
        gv = sum(torch.sum(g * vi) for g, vi in zip(grads, v_list))
        d_vec = torch.autograd.grad(gv, beta, retain_graph=True)[0]  # shape (B,)

        # Weighted sum of gradients sum_i (d_i / N) * g_i via single VJP
        weights = (d_vec.detach() / N)
        weighted_sum = torch.sum(losses * weights)
        fvp_grads = torch.autograd.grad(weighted_sum, params, retain_graph=False)

        offset = 0
        for fg in fvp_grads:
            n = fg.numel()
            result[offset:offset+n] += fg.detach().reshape(-1)
            offset += n

    return result


# ── Spectral Norm Estimation & GPU Conjugate Gradient ───────────────────────

def power_iteration_spectral_norm(matvec_fn, dim, device, num_iters=30, seed=0):
    """Estimate ||A||_2 via power iteration, handling indefinite matrices."""
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

    torch.manual_seed(seed + 1000)
    v2 = torch.randn(dim, device=device)
    v2 = v2 / v2.norm()
    neg_eig = 0.0
    for _ in range(num_iters):
        Av2 = -matvec_fn(v2)
        neg_eig = (v2 @ Av2).item()
        Av2_norm = Av2.norm()
        if Av2_norm < 1e-30:
            break
        v2 = Av2 / Av2_norm

    return max(abs(eigenvalue), abs(neg_eig))


def cg_solve(matvec, b, damping, tol=1e-4, max_iter=20):
    """Solve (A + damping*I) x = b via Conjugate Gradient on GPU."""
    x = torch.zeros_like(b)
    r = b - (matvec(x) + damping * x)
    p = r.clone()
    rs_old = r @ r
    for _ in range(max_iter):
        Ap = matvec(p) + damping * p
        pAp = p @ Ap
        if pAp.abs() < 1e-30:
            break
        alpha = rs_old / pAp
        x += alpha * p
        r -= alpha * Ap
        rs_new = r @ r
        if rs_new.sqrt() < tol:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x


def estimate_seff(hvp_fn, fvp_fn, dim, device, damping=1e-3, num_iters=10, cg_max_iter=15):
    """Estimate S_eff = ||(F + γI)^{-1} H||_2 via 100% GPU Power Iteration + CG."""
    def combined_matvec(v):
        Hv = hvp_fn(v)
        x = cg_solve(fvp_fn, Hv, damping=damping, tol=1e-4, max_iter=cg_max_iter)
        return x
    return power_iteration_spectral_norm(combined_matvec, dim, device, num_iters=num_iters)


# ── MNIST Experiment ────────────────────────────────────────────────────────

def run_mnist_misspec():
    print("\n" + "=" * 70)
    print("MNIST MLP: Matrix-Free Misspecification Measurement")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    torch.manual_seed(42)
    indices = torch.randperm(len(trainset))[:2000]
    subset = torch.utils.data.Subset(trainset, indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=2000, shuffle=False)
    X_full, Y_full = next(iter(loader))
    X_full = X_full.reshape(-1, 784).to(DEVICE)
    Y_full = Y_full.to(DEVICE)

    model = TanhMLP().to(DEVICE)
    params = list(model.parameters())
    d = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(params, lr=0.01)
    damping = 1e-3

    print(f"  Parameters: {d:,}")

    N_mf = 100
    checkpoints = [1, 5, 10, 25]
    results = []
    current_step = 0

    for target_epoch in checkpoints:
        for _ in range(target_epoch - current_step):
            pred = model(X_full)
            loss = nn.CrossEntropyLoss()(pred, Y_full)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        current_step = target_epoch

        with torch.no_grad():
            pred = model(X_full)
            loss_val = nn.CrossEntropyLoss()(pred, Y_full).item()
            acc = (pred.argmax(1) == Y_full).float().mean().item() * 100

        print(f"\n  Epoch {target_epoch}: loss={loss_val:.4f}, acc={acc:.1f}%")

        X_sub = X_full[:N_mf]
        Y_sub = Y_full[:N_mf]

        t0 = time.time()

        def hg_mv(v):
            Hv = hvp_flat(model, X_sub, Y_sub, params, v, "ce")
            Gv = ggn_vp_batched(model, X_sub, Y_sub, params, v, "ce", batch_size=32)
            return Hv - Gv

        eps_est = power_iteration_spectral_norm(hg_mv, d, DEVICE, num_iters=20)

        def gf_mv(v):
            Gv = ggn_vp_batched(model, X_sub, Y_sub, params, v, "ce", batch_size=32)
            Fv = empirical_fvp_batched(model, X_sub, Y_sub, params, v, "ce", batch_size=32)
            return Gv - Fv

        delta_est = power_iteration_spectral_norm(gf_mv, d, DEVICE, num_iters=20)

        def hvp_fn(v):
            return hvp_flat(model, X_sub, Y_sub, params, v, "ce")
        def fvp_fn(v):
            return empirical_fvp_batched(model, X_sub, Y_sub, params, v, "ce", batch_size=32)

        seff_est = estimate_seff(hvp_fn, fvp_fn, d, DEVICE, damping=damping, num_iters=10, cg_max_iter=15)
        F_norm = power_iteration_spectral_norm(fvp_fn, d, DEVICE, num_iters=20)

        elapsed = time.time() - t0
        delta_rel = delta_est / max(F_norm, 1e-12)

        print(f"    ε={eps_est:.4f}, δ={delta_est:.4f}, δ/||F||={delta_rel:.2f}, S_eff≈{seff_est:.1f}  ({elapsed:.1f}s)")

        results.append({
            "task": "MNIST MLP",
            "params": d,
            "checkpoint": f"epoch {target_epoch}",
            "loss": loss_val,
            "accuracy": acc,
            "eps_est": float(eps_est),
            "delta_est": float(delta_est),
            "F_norm_est": float(F_norm),
            "delta_relative": float(delta_rel),
            "seff_est": float(seff_est),
            "time_s": elapsed,
        })

    return results


# ── CIFAR-10 ResNet-18 Experiment ──────────────────────────────────────────

def run_cifar_misspec():
    print("\n" + "=" * 70)
    print("CIFAR-10 ResNet-18: Matrix-Free Misspecification Measurement")
    print("=" * 70)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset, testset = get_cifar10_datasets(transform_train, transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

    model = get_cifar_resnet18().to(DEVICE)
    params = list(model.parameters())
    d = sum(p.numel() for p in params)
    optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9, weight_decay=5e-4)
    loss_fn = nn.CrossEntropyLoss()
    damping = 1e-3

    print(f"  Parameters: {d:,}")

    N_mf = 32
    checkpoints = [1, 5, 25]
    results = []

    for epoch in range(max(checkpoints)):
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

        if (epoch + 1) in checkpoints:
            model.eval()
            correct, total = 0, 0
            with torch.no_grad():
                for imgs, labels in testloader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    out = model(imgs)
                    correct += (out.argmax(1) == labels).sum().item()
                    total += labels.size(0)
            acc = correct / total * 100
            model.train()

            print(f"\n  Epoch {epoch+1}: acc={acc:.1f}%")

            mf_batch = next(iter(trainloader))
            X_mf = mf_batch[0][:N_mf].to(DEVICE)
            Y_mf = mf_batch[1][:N_mf].to(DEVICE)

            t0 = time.time()

            def hg_mv(v):
                Hv = hvp_flat(model, X_mf, Y_mf, params, v, "ce")
                Gv = ggn_vp_batched(model, X_mf, Y_mf, params, v, "ce", batch_size=8)
                return Hv - Gv

            eps_est = power_iteration_spectral_norm(hg_mv, d, DEVICE, num_iters=15)

            def gf_mv(v):
                Gv = ggn_vp_batched(model, X_mf, Y_mf, params, v, "ce", batch_size=8)
                Fv = empirical_fvp_batched(model, X_mf, Y_mf, params, v, "ce", batch_size=8)
                return Gv - Fv

            delta_est = power_iteration_spectral_norm(gf_mv, d, DEVICE, num_iters=15)

            def hvp_fn(v):
                return hvp_flat(model, X_mf, Y_mf, params, v, "ce")
            def fvp_fn(v):
                return empirical_fvp_batched(model, X_mf, Y_mf, params, v, "ce", batch_size=8)

            # Fully compute F_norm and S_eff without skipping
            print("    Computing ||F||_2 and estimating S_eff...")
            F_norm = power_iteration_spectral_norm(fvp_fn, d, DEVICE, num_iters=15)
            seff_est = estimate_seff(hvp_fn, fvp_fn, d, DEVICE, damping=damping, num_iters=8, cg_max_iter=12)

            elapsed = time.time() - t0
            delta_rel = delta_est / max(F_norm, 1e-12)

            print(f"    ε≈{eps_est:.4f}, δ≈{delta_est:.4f}, δ/||F||={delta_rel:.2f}, S_eff≈{seff_est:.1f}  ({elapsed:.1f}s)")

            results.append({
                "task": "CIFAR ResNet-18",
                "params": d,
                "checkpoint": f"epoch {epoch+1}",
                "accuracy": acc,
                "eps_est": float(eps_est),
                "delta_est": float(delta_est),
                "F_norm_est": float(F_norm),
                "delta_relative": float(delta_rel),
                "seff_est": float(seff_est),
                "time_s": elapsed,
            })

    return results


def main():
    # Direct execution without error swallowing
    mnist_results = run_mnist_misspec()
    RESULTS["mnist"] = mnist_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2)

    cifar_results = run_cifar_misspec()
    RESULTS["cifar"] = cifar_results

    with open(RESULTS_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2)

    print("\n" + "=" * 70)
    print("Misspecification at Scale (Table for Paper)")
    print("=" * 70)
    print(f"{'Task':<20} {'Params':>10} {'Checkpoint':>12} {'ε':>10} {'δ':>10} {'δ/||F||':>10} {'S_eff':>10}")
    for key in ["mnist", "cifar"]:
        if isinstance(RESULTS.get(key), list):
            for r in RESULTS[key]:
                delta_rel = r.get("delta_relative", "N/A")
                if isinstance(delta_rel, float):
                    delta_rel = f"{delta_rel:.2f}"
                seff_val = r.get("seff_est", "N/A")
                if isinstance(seff_val, float):
                    seff_val = f"{seff_val:.1f}"
                print(f"{r['task']:<20} {r['params']:>10,} {r['checkpoint']:>12} {r['eps_est']:>10.4f} {r['delta_est']:>10.4f} {delta_rel:>10} {seff_val:>10}")


if __name__ == "__main__":
    main()