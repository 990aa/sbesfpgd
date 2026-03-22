import json
import matplotlib.pyplot as plt
import os

with open("sbesfpgd-verify/gpu_experiment_results_stable.json", "r") as f:
    data = json.load(f)

cifar10 = data["cifar10"]
sgd_sharp_data = cifar10["sgd"]["sharpness"]
kfac_sharp_data = cifar10["best_kfac"]["sharpness"]

sgd_epochs = [item["epoch"] for item in sgd_sharp_data]
sgd_sharp = [item["lambda_max_H"] for item in sgd_sharp_data]

kfac_epochs = [item["epoch"] for item in kfac_sharp_data]
kfac_sharp = [item["lambda_max_H"] for item in kfac_sharp_data]

# Hardcoded SGD Accs from GPU logs
sgd_accs = [
    0.398,
    0.524,
    0.611,
    0.665,
    0.724,
    0.768,
    0.779,
    0.775,
    0.816,
    0.774,
    0.820,
    0.830,
    0.826,
    0.828,
    0.838,
    0.836,
    0.785,
    0.853,
    0.779,
    0.820,
    0.818,
    0.830,
    0.844,
    0.865,
    0.862,
]
sgd_accs = [x * 100 for x in sgd_accs]

kfac_accs = [x * 100 for x in cifar10["best_kfac"]["epoch_accs"]]

# Setup plot
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))

axes[0].plot(sgd_epochs, sgd_accs, "o-", label="SGD", color="tab:red", markersize=6)
axes[0].plot(kfac_epochs, kfac_accs, "o-", label="K-FAC (ASDL)", color="tab:blue", markersize=6)
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Test Accuracy (%)")
axes[0].set_title("(a) Test Accuracy")
axes[0].legend()
axes[0].grid(True, alpha=0.2)

axes[1].plot(sgd_epochs, sgd_sharp, "o-", label="SGD", color="tab:red", markersize=4)
axes[1].plot(kfac_epochs, kfac_sharp, "o-", label="K-FAC (ASDL)", color="tab:blue", markersize=4)
axes[1].axhline(2 / 0.1, color="black", ls="--", label=r"$2/\eta=20$", alpha=0.5)
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel(r"$\lambda_{\max}(H)$")
axes[1].set_title("(b) Sharpness Dynamics")
axes[1].legend()
axes[1].grid(True, alpha=0.2)
axes[1].set_yscale("linear")  # Keep linear or log? The KFAC peaks at 60k, maybe linear is ok or log scale

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/cifar10_resnet18.png", dpi=300, bbox_inches="tight")
print("Saved figures/cifar10_resnet18.png")
