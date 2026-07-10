"""
CIFAR-10 ADAHESSIAN + SOPHIA Baselines (GPU)

Trains ResNet-18 on CIFAR-10 for 25 epochs with:
  1. ADAHESSIAN (lr=0.15, weight_decay=5e-4, hessian_power=1)
  2. SOPHIA (lr=0.01, rho=0.04, weight_decay=5e-4)

Also measures per-epoch sharpness (lambda_max(H) via power iteration).

Run on Google Colab with GPU: !pip install torch torchvision datasets

Output: cifar_adahessian_sophia_gpu_results.json
"""

import json
import math
import sys
import time
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    import datasets
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "datasets"])
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models
    import datasets

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

RESULTS_FILE = "cifar_adahessian_sophia_gpu_results.json"
RESULTS = {}


def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"  [Results saved to {RESULTS_FILE}]")


def get_cifar_resnet18():
    """CIFAR-10-adapted ResNet-18 (same as cifar_baselines_gpu.py)."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def hessian_top_eigenvalue(model, loss_fn, inputs, targets, num_iter=10):
    """Estimate lambda_max(H) via Power Iteration."""
    model.zero_grad(set_to_none=True)
    out = model(inputs)
    loss = loss_fn(out, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    valid_params, valid_grads = [], []
    for g, p in zip(grads, params):
        if g is not None:
            valid_grads.append(g)
            valid_params.append(p)
    if not valid_params:
        return 0.0
    v = [torch.randn_like(p) for p in valid_params]
    vnorm = torch.sqrt(sum(torch.sum(vi ** 2) for vi in v))
    v = [vi / vnorm for vi in v]
    eigenvalue = 0.0
    for _ in range(num_iter):
        Hv = torch.autograd.grad(valid_grads, valid_params, grad_outputs=v, retain_graph=True)
        eigenvalue = sum(torch.sum(hi * vi) for hi, vi in zip(Hv, v)).item()
        vnorm = torch.sqrt(sum(torch.sum(hi ** 2) for hi in Hv))
        if vnorm.item() < 1e-30:
            break
        v = [hi / vnorm for hi in Hv]
    return eigenvalue


def get_cifar_data():
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

    # Use Hugging Face CDN to speed up downloading
    hf_ds = datasets.load_dataset("uoft-cs/cifar10")

    def train_transforms(examples):
        examples["pixel_values"] = [transform_train(img.convert("RGB")) for img in examples["img"]]
        return examples

    def test_transforms(examples):
        examples["pixel_values"] = [transform_test(img.convert("RGB")) for img in examples["img"]]
        return examples

    trainset = hf_ds["train"].with_transform(train_transforms)
    testset = hf_ds["test"].with_transform(test_transforms)

    def collate_fn(batch):
        imgs = torch.stack([x["pixel_values"] for x in batch])
        labels = torch.tensor([x["label"] for x in batch])
        return imgs, labels

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2, collate_fn=collate_fn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2, collate_fn=collate_fn)
    return trainloader, testloader


def evaluate(model, testloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return correct / total * 100


# ===========================================================================
# ADAHESSIAN Optimizer (self-contained implementation)
# ===========================================================================
class ADAHESSIAN(torch.optim.Optimizer):
    """
    ADAHESSIAN: An Adaptive Second Order Optimizer for Machine Learning
    (Yao et al., 2021, AAAI)
    Uses Hutchinson's estimator for diagonal Hessian with spatial averaging.
    """
    def __init__(self, params, lr=0.15, betas=(0.9, 0.999), eps=1e-4,
                 weight_decay=0.0, hessian_power=1):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        hessian_power=hessian_power)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _get_hessian_trace(self):
        """Compute diagonal Hessian via Hutchinson's estimator."""
        params_with_grad = []
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None and p.requires_grad:
                    params_with_grad.append(p)
                    grads.append(p.grad)

        if not params_with_grad:
            return

        zs = [torch.randint_like(p, 0, 2) * 2.0 - 1.0 for p in params_with_grad]

        with torch.enable_grad():
            hvps = torch.autograd.grad(
                grads, params_with_grad, grad_outputs=zs,
                only_inputs=True, retain_graph=True
            )

        for p, z, hvp in zip(params_with_grad, zs, hvps):
            diag_hessian = hvp * z
            if len(diag_hessian.shape) > 2:
                diag_hessian = diag_hessian.abs().mean(
                    dim=list(range(2, len(diag_hessian.shape))), keepdim=True
                ).expand_as(p)
            else:
                diag_hessian = diag_hessian.abs()

            state = self.state[p]
            if 'hessian_diag' not in state:
                state['hessian_diag'] = diag_hessian.clone()
            else:
                state['hessian_diag'].mul_(0.999).add_(diag_hessian, alpha=0.001)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._get_hessian_trace()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            hessian_power = group['hessian_power']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad
                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_hessian_diag_sq'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg = state['exp_avg']
                exp_hessian = state['exp_hessian_diag_sq']

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                hessian_diag = state.get('hessian_diag', torch.ones_like(p))
                exp_hessian.mul_(beta2).addcmul_(hessian_diag, hessian_diag, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                corrected_avg = exp_avg / bias_correction1
                corrected_hessian = (exp_hessian / bias_correction2).sqrt()

                if hessian_power < 1:
                    corrected_hessian = corrected_hessian ** hessian_power

                p.add_(corrected_avg / (corrected_hessian + eps), alpha=-lr)

        return loss


# ===========================================================================
# SOPHIA Optimizer (self-contained, SophiaG variant)
# ===========================================================================
class SophiaG(torch.optim.Optimizer):
    """
    SOPHIA: A Scalable Stochastic Second-order Optimizer (Liu et al., 2023)
    SophiaG variant using Gauss-Newton-Bartlett estimator.
    """
    def __init__(self, params, lr=0.01, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay)
        super().__init__(params, defaults)

    # REMOVED @torch.no_grad() from here so loss_sample.backward() works
    def update_hessian(self, model, loss_fn, data, target):
        """Update diagonal Hessian estimate using mini-batch."""
        model.zero_grad()
        output = model(data)
        samp_dist = torch.distributions.Categorical(logits=output)
        y_sample = samp_dist.sample()
        loss_sample = loss_fn(output, y_sample)
        loss_sample.backward()

        # Wrap just the state updates in torch.no_grad()
        with torch.no_grad():
            for group in self.param_groups:
                beta2 = group['betas'][1]
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    if 'hessian' not in state:
                        state['hessian'] = torch.zeros_like(p)
                    state['hessian'].mul_(beta2).addcmul_(
                        p.grad, p.grad, value=1 - beta2
                    )
        model.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['betas'][0]
            rho = group['rho']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad

                state = self.state[p]
                if 'step' not in state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    if 'hessian' not in state:
                        state['hessian'] = torch.zeros_like(p)

                state['step'] += 1
                exp_avg = state['exp_avg']
                hessian = state['hessian']

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                update = torch.where(
                    exp_avg.abs() > rho * hessian,
                    torch.sign(exp_avg),
                    exp_avg / (rho * hessian + 1e-15)
                )
                p.add_(update, alpha=-lr)

        return loss


def train_adahessian(trainloader, testloader, epochs=25):
    """ADAHESSIAN optimizer."""
    print("\n--- ADAHESSIAN ---")
    torch.manual_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    optimizer = ADAHESSIAN(
        model.parameters(), lr=0.15, weight_decay=5e-4,
        hessian_power=1, eps=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss()
    results = {"per_epoch": [], "method": "ADAHESSIAN"}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward(create_graph=True)
            optimizer.step()

        scheduler.step()
        acc = evaluate(model, testloader)

        sharp_batch = next(iter(trainloader))
        sharp_imgs = sharp_batch[0][:64].to(DEVICE)
        sharp_labels = sharp_batch[1][:64].to(DEVICE)
        sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_imgs, sharp_labels, num_iter=10)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: acc={acc:.1f}%, lambda_max(H)={sharpness:.0f}, time={elapsed:.0f}s")
        results["per_epoch"].append({
            "epoch": epoch + 1, "accuracy": acc, "sharpness": sharpness, "time": elapsed
        })

    results["final_acc"] = results["per_epoch"][-1]["accuracy"]
    results["best_acc"] = max(r["accuracy"] for r in results["per_epoch"])
    results["mean_sharpness"] = float(np.mean([r["sharpness"] for r in results["per_epoch"]]))
    results["peak_sharpness"] = float(np.max([r["sharpness"] for r in results["per_epoch"]]))
    results["wall_time"] = time.time() - t0
    return results


def train_sophia(trainloader, testloader, epochs=25):
    """SOPHIA optimizer (SophiaG variant)."""
    print("\n--- SOPHIA (SophiaG) ---")
    torch.manual_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    optimizer = SophiaG(
        model.parameters(), lr=0.01, rho=0.04, weight_decay=5e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss()
    results = {"per_epoch": [], "method": "SOPHIA (SophiaG)"}
    t0 = time.time()
    hessian_update_interval = 10

    for epoch in range(epochs):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            if batch_idx % hessian_update_interval == 0:
                optimizer.update_hessian(model, loss_fn, imgs, labels)

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

        scheduler.step()
        acc = evaluate(model, testloader)

        sharp_batch = next(iter(trainloader))
        sharp_imgs = sharp_batch[0][:64].to(DEVICE)
        sharp_labels = sharp_batch[1][:64].to(DEVICE)
        sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_imgs, sharp_labels, num_iter=10)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: acc={acc:.1f}%, lambda_max(H)={sharpness:.0f}, time={elapsed:.0f}s")
        results["per_epoch"].append({
            "epoch": epoch + 1, "accuracy": acc, "sharpness": sharpness, "time": elapsed
        })

    results["final_acc"] = results["per_epoch"][-1]["accuracy"]
    results["best_acc"] = max(r["accuracy"] for r in results["per_epoch"])
    results["mean_sharpness"] = float(np.mean([r["sharpness"] for r in results["per_epoch"]]))
    results["peak_sharpness"] = float(np.max([r["sharpness"] for r in results["per_epoch"]]))
    results["wall_time"] = time.time() - t0
    return results


def main():
    trainloader, testloader = get_cifar_data()

    print("=" * 70)
    print("CIFAR-10 ResNet-18 Baselines: ADAHESSIAN + SOPHIA (25 epochs)")
    print("=" * 70)

    try:
        RESULTS["adahessian"] = train_adahessian(trainloader, testloader, epochs=25)
        save_results()
    except Exception as e:
        print(f"  ADAHESSIAN failed: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["adahessian"] = {"error": str(e)}
        save_results()

    try:
        RESULTS["sophia"] = train_sophia(trainloader, testloader, epochs=25)
        save_results()
    except Exception as e:
        print(f"  SOPHIA failed: {e}")
        import traceback
        traceback.print_exc()
        RESULTS["sophia"] = {"error": str(e)}
        save_results()

    print("\n" + "=" * 70)
    print("CIFAR-10 ResNet-18 Baseline Results (25 epochs)")
    print("=" * 70)
    fmt = "{:<25} {:>10} {:>10} {:>14} {:>14} {:>10}"
    print(fmt.format("Method", "Final Acc", "Best Acc", "Mean lmax(H)", "Peak lmax(H)", "Wall Time"))
    for name, r in RESULTS.items():
        if "error" in r:
            print(f"{name:<25} {'ERROR':>10}  -- {r['error'][:40]}")
            continue
        print(fmt.format(
            r['method'],
            f"{r['final_acc']:.1f}%",
            f"{r['best_acc']:.1f}%",
            f"{r['mean_sharpness']:.0f}",
            f"{r['peak_sharpness']:.0f}",
            f"{r['wall_time']:.0f}s"
        ))

    print(f"\nResults saved to: {RESULTS_FILE}")
    print("Please report the table above and the JSON file contents back.")


if __name__ == "__main__":
    main()