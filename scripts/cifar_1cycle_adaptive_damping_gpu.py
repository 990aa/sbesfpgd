"""
CIFAR-10 ResNet-18 1cycle and adaptive-damping follow-up (GPU).

This standalone Colab-oriented script runs two missing follow-up baselines:
  1. SGD with PyTorch OneCycleLR.
  2. K-FAC via ASDL with an epoch-wise adaptive damping schedule.

It writes cifar_1cycle_adaptive_damping_gpu_results.json.

Run on Google Colab:
    !python scripts/cifar_1cycle_adaptive_damping_gpu.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import warnings

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.models as models
    import torchvision.transforms as transforms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.models as models
    import torchvision.transforms as transforms

try:
    from asdl import KfacGradientMaker, PreconditioningConfig
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "git+https://github.com/kazukiosawa/asdl.git"]
    )
    from asdl import KfacGradientMaker, PreconditioningConfig


warnings.filterwarnings("ignore")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_FILE = "cifar_1cycle_adaptive_damping_gpu_results.json"
RESULTS: dict[str, dict] = {}

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


def save_results() -> None:
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"[saved] {RESULTS_FILE}")


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar_resnet18() -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def get_cifar_data(batch_size: int = 256):
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
    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=512, shuffle=False, num_workers=2, pin_memory=True
    )
    return trainloader, testloader


def evaluate(model: nn.Module, testloader) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    model.train()
    return 100.0 * correct / total


def hessian_top_eigenvalue(model: nn.Module, loss_fn, inputs, targets, num_iter: int = 10) -> float:
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    params = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    valid_params = []
    valid_grads = []
    for grad, param in zip(grads, params):
        if grad is not None:
            valid_grads.append(grad)
            valid_params.append(param)
    if not valid_params:
        return 0.0

    vec = [torch.randn_like(param) for param in valid_params]
    norm = torch.sqrt(sum(torch.sum(v * v) for v in vec))
    vec = [v / norm for v in vec]
    eigenvalue = 0.0
    for _ in range(num_iter):
        hvec = torch.autograd.grad(valid_grads, valid_params, grad_outputs=vec, retain_graph=True)
        eigenvalue = sum(torch.sum(h * v) for h, v in zip(hvec, vec)).item()
        norm = torch.sqrt(sum(torch.sum(h * h) for h in hvec))
        if norm.item() < 1e-30:
            break
        vec = [h / norm for h in hvec]
    return float(eigenvalue)


def summarize(per_epoch: list[dict], method: str, wall_time: float) -> dict:
    sharpness = [row["sharpness"] for row in per_epoch]
    accs = [row["accuracy"] for row in per_epoch]
    return {
        "method": method,
        "per_epoch": per_epoch,
        "final_acc": float(accs[-1]),
        "best_acc": float(max(accs)),
        "mean_sharpness": float(np.mean(sharpness)),
        "peak_sharpness": float(np.max(sharpness)),
        "wall_time": float(wall_time),
    }


def measure_epoch(model, loss_fn, trainloader, testloader, epoch: int, t0: float, extra=None) -> dict:
    acc = evaluate(model, testloader)
    sharp_batch = next(iter(trainloader))
    sharp_images = sharp_batch[0][:64].to(DEVICE, non_blocking=True)
    sharp_labels = sharp_batch[1][:64].to(DEVICE, non_blocking=True)
    sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_images, sharp_labels, num_iter=10)
    row = {
        "epoch": epoch,
        "accuracy": float(acc),
        "sharpness": float(sharpness),
        "time": float(time.time() - t0),
    }
    if extra:
        row.update(extra)
    return row


def train_sgd_onecycle(trainloader, testloader, epochs: int = 25) -> dict:
    print("\n--- SGD + OneCycleLR ---")
    set_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.2,
        epochs=epochs,
        steps_per_epoch=len(trainloader),
        pct_start=0.2,
        anneal_strategy="cos",
        div_factor=10.0,
        final_div_factor=100.0,
    )
    per_epoch = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        for images, labels in trainloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        row = measure_epoch(model, loss_fn, trainloader, testloader, epoch, t0, {"lr": float(lr)})
        per_epoch.append(row)
        print(
            f"epoch {epoch:02d}: acc={row['accuracy']:.2f}% "
            f"lambda_max={row['sharpness']:.2f} lr={lr:.5f}"
        )
    return summarize(per_epoch, "SGD + OneCycleLR", time.time() - t0)


def adaptive_damping(epoch: int, gamma0: float = 1e-2, gamma_min: float = 1e-4, decay: float = 0.15) -> float:
    return max(gamma_min, gamma0 / (1.0 + decay * (epoch - 1)))


def set_asdl_damping(config, grad_maker, value: float) -> None:
    # ASDL versions differ in whether damping is read from config or cached
    # on the gradient maker; update every obvious location and record it.
    if hasattr(config, "damping"):
        config.damping = value
    if hasattr(grad_maker, "damping"):
        grad_maker.damping = value
    if hasattr(grad_maker, "_damping"):
        grad_maker._damping = value


def train_kfac_adaptive_damping(trainloader, testloader, epochs: int = 25) -> dict:
    print("\n--- K-FAC + adaptive damping ---")
    set_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    config = PreconditioningConfig(
        data_size=256,
        damping=adaptive_damping(1),
        curvature_upd_interval=10,
        preconditioner_upd_interval=10,
    )
    grad_maker = KfacGradientMaker(model, config, loss_type="cross_entropy")
    per_epoch = []
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        gamma = adaptive_damping(epoch)
        set_asdl_damping(config, grad_maker, gamma)
        model.train()
        for images, labels in trainloader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            dummy = grad_maker.setup_model_call(model, images)
            grad_maker.setup_loss_call(loss_fn, dummy, labels)
            _, loss = grad_maker.forward_and_backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if not math.isfinite(float(loss.detach().cpu())):
                raise RuntimeError("non-finite K-FAC loss")
        row = measure_epoch(
            model, loss_fn, trainloader, testloader, epoch, t0, {"damping": float(gamma)}
        )
        per_epoch.append(row)
        print(
            f"epoch {epoch:02d}: acc={row['accuracy']:.2f}% "
            f"lambda_max={row['sharpness']:.2f} damping={gamma:.6f}"
        )
    return summarize(per_epoch, "K-FAC + adaptive damping", time.time() - t0)


def main() -> None:
    print(f"device: {DEVICE}")
    trainloader, testloader = get_cifar_data(batch_size=256)

    RESULTS["sgd_onecycle"] = train_sgd_onecycle(trainloader, testloader, epochs=25)
    save_results()

    try:
        RESULTS["kfac_adaptive_damping"] = train_kfac_adaptive_damping(
            trainloader, testloader, epochs=25
        )
    except Exception as exc:
        RESULTS["kfac_adaptive_damping"] = {"error": repr(exc)}
        print(f"K-FAC adaptive damping failed: {exc!r}")
    save_results()

    print("\nSummary")
    print("Method                         Final    Best     Mean lmax   Peak lmax   Time")
    for result in RESULTS.values():
        if "error" in result:
            print(f"{result['error']}")
            continue
        print(
            f"{result['method']:<30} {result['final_acc']:>6.1f}% "
            f"{result['best_acc']:>6.1f}% {result['mean_sharpness']:>10.1f} "
            f"{result['peak_sharpness']:>10.1f} {result['wall_time']:>7.0f}s"
        )


if __name__ == "__main__":
    main()
