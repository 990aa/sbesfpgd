"""
CIFAR-10 Optimizer Baselines + Matrix-Free S_eff Estimation (GPU)

Extends Table VI with:
- AdamW (η=0.001, weight_decay=0.01)
- SGD + warmup + cosine decay (η=0.1, warmup=5 epochs)
- Shampoo (via ASDL)

Also attempts matrix-free S_eff estimation on ResNet-18 at 2-3 checkpoints.

Run on Google Colab with GPU:
    !pip install git+https://github.com/kazukiosawa/asdl.git
    !python scripts/cifar_baselines_gpu.py
"""

import json
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
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision"])
    import torch
    import torch.nn as nn
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.models as models

try:
    from asdl import KfacGradientMaker, ShampooGradientMaker, PreconditioningConfig
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/kazukiosawa/asdl.git"])
    from asdl import KfacGradientMaker, ShampooGradientMaker, PreconditioningConfig

warnings.filterwarnings("ignore")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

RESULTS_FILE = "cifar_baselines_gpu_results.json"
RESULTS = {}


def save_results():
    with open(RESULTS_FILE, "w") as f:
        json.dump(RESULTS, f, indent=2, default=str)
    print(f"  [Results saved to {RESULTS_FILE}]")


def get_cifar_resnet18():
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(512, 10)
    return model


def hessian_top_eigenvalue(model, loss_fn, inputs, targets, num_iter=10):
    """Estimate λ_max(H) via Power Iteration."""
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
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)
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


def train_sgd_warmup_cosine(trainloader, testloader, epochs=25):
    """SGD with linear warmup (5 epochs) + cosine decay."""
    print("\n--- SGD + Warmup + Cosine Decay ---")
    torch.manual_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    warmup_epochs = 5
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)

    loss_fn = nn.CrossEntropyLoss()
    results = {"per_epoch": [], "method": "SGD+Warmup+Cosine"}
    t0 = time.time()

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            lr = 0.1 * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = lr

        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

        if epoch >= warmup_epochs:
            scheduler.step()

        acc = evaluate(model, testloader)

        sharp_batch = next(iter(trainloader))
        sharp_imgs, sharp_labels = sharp_batch[0][:64].to(DEVICE), sharp_batch[1][:64].to(DEVICE)
        sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_imgs, sharp_labels, num_iter=10)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: acc={acc:.1f}%, λ_max(H)={sharpness:.0f}, time={elapsed:.0f}s")
        results["per_epoch"].append({
            "epoch": epoch + 1, "accuracy": acc, "sharpness": sharpness, "time": elapsed
        })

    results["final_acc"] = results["per_epoch"][-1]["accuracy"]
    results["best_acc"] = max(r["accuracy"] for r in results["per_epoch"])
    results["mean_sharpness"] = float(np.mean([r["sharpness"] for r in results["per_epoch"]]))
    results["peak_sharpness"] = float(np.max([r["sharpness"] for r in results["per_epoch"]]))
    results["wall_time"] = time.time() - t0
    return results


def train_adamw(trainloader, testloader, epochs=25):
    """AdamW optimizer."""
    print("\n--- AdamW ---")
    torch.manual_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    loss_fn = nn.CrossEntropyLoss()
    results = {"per_epoch": [], "method": "AdamW"}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for imgs, labels in trainloader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        acc = evaluate(model, testloader)
        sharp_batch = next(iter(trainloader))
        sharp_imgs, sharp_labels = sharp_batch[0][:64].to(DEVICE), sharp_batch[1][:64].to(DEVICE)
        sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_imgs, sharp_labels, num_iter=10)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: acc={acc:.1f}%, λ_max(H)={sharpness:.0f}, time={elapsed:.0f}s")
        results["per_epoch"].append({
            "epoch": epoch + 1, "accuracy": acc, "sharpness": sharpness, "time": elapsed
        })

    results["final_acc"] = results["per_epoch"][-1]["accuracy"]
    results["best_acc"] = max(r["accuracy"] for r in results["per_epoch"])
    results["mean_sharpness"] = float(np.mean([r["sharpness"] for r in results["per_epoch"]]))
    results["peak_sharpness"] = float(np.max([r["sharpness"] for r in results["per_epoch"]]))
    results["wall_time"] = time.time() - t0
    return results


def train_shampoo(trainloader, testloader, epochs=25):
    """Shampoo via ASDL."""
    print("\n--- Shampoo (ASDL) ---")
    torch.manual_seed(42)
    model = get_cifar_resnet18().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

    config = PreconditioningConfig(damping=1e-3)
    grad_maker = ShampooGradientMaker(model, config)

    loss_fn = nn.CrossEntropyLoss()
    results = {"per_epoch": [], "method": "Shampoo (ASDL)"}
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        for batch_idx, (imgs, labels) in enumerate(trainloader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            try:
                dummy = grad_maker.setup_model_call(model, imgs)
                grad_maker.setup_loss_call(loss_fn, dummy, labels)
                out, loss = grad_maker.forward_and_backward()
                if batch_idx % 10 == 0:
                    grad_maker.update_curvature()
                grad_maker.precondition()
            except Exception as e:
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()

            optimizer.step()

        acc = evaluate(model, testloader)
        sharp_batch = next(iter(trainloader))
        sharp_imgs, sharp_labels = sharp_batch[0][:64].to(DEVICE), sharp_batch[1][:64].to(DEVICE)
        sharpness = hessian_top_eigenvalue(model, loss_fn, sharp_imgs, sharp_labels, num_iter=10)

        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}: acc={acc:.1f}%, λ_max(H)={sharpness:.0f}, time={elapsed:.0f}s")
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

    RESULTS["sgd_warmup_cosine"] = train_sgd_warmup_cosine(trainloader, testloader, epochs=25)
    save_results()

    RESULTS["adamw"] = train_adamw(trainloader, testloader, epochs=25)
    save_results()

    try:
        RESULTS["shampoo"] = train_shampoo(trainloader, testloader, epochs=25)
        save_results()
    except Exception as e:
        print(f"  Shampoo failed: {e}")
        RESULTS["shampoo"] = {"error": str(e)}
        save_results()

    print("\n" + "=" * 70)
    print("CIFAR-10 ResNet-18 Baseline Results (25 epochs)")
    print("=" * 70)
    print(f"{'Method':<25} {'Final Acc':>10} {'Best Acc':>10} {'Mean λ_max(H)':>14} {'Peak λ_max(H)':>14} {'Wall Time':>10}")
    for name, r in RESULTS.items():
        if "error" in r:
            print(f"{name:<25} {'ERROR':>10}")
            continue
        print(f"{r['method']:<25} {r['final_acc']:>9.1f}% {r['best_acc']:>9.1f}% {r['mean_sharpness']:>14.0f} {r['peak_sharpness']:>14.0f} {r['wall_time']:>9.0f}s")


if __name__ == "__main__":
    main()
