"""
download_datasets.py

Download all datasets used in the experiments of:

  "A Spectral Bound on Effective Sharpness for Fisher-Preconditioned
   Gradient Descent"

Datasets:
  1. MNIST       — Used in Section VII.J (nonlinear validation, 2,000-sample subset)
  2. CIFAR-10    — Used in Section VII.M (ResNet-18 at scale, full 50,000 train)

The DLN (Deep Linear Network) experiments use synthetic teacher-student data
generated at runtime and do not require any download.

Usage:
  pip install torch torchvision
  python download_datasets.py [--data-dir DATA_DIR]

Default data directory: ../data/
"""

import argparse
import os

import torchvision
import torchvision.transforms as transforms


def download_mnist(data_dir: str) -> None:
    """Download MNIST dataset (train + test)."""
    print("Downloading MNIST...")
    torchvision.datasets.MNIST(root=data_dir, train=True, download=True,
                               transform=transforms.ToTensor())
    torchvision.datasets.MNIST(root=data_dir, train=False, download=True,
                               transform=transforms.ToTensor())
    print(f"  MNIST saved to {os.path.join(data_dir, 'MNIST')}")


def download_cifar10(data_dir: str) -> None:
    """Download CIFAR-10 dataset (train + test)."""
    print("Downloading CIFAR-10...")
    torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                 transform=transforms.ToTensor())
    torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                 transform=transforms.ToTensor())
    print(f"  CIFAR-10 saved to {os.path.join(data_dir, 'cifar-10-batches-py')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download datasets for sbesfpgd experiments")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "..", "data"),
                        help="Directory to store datasets (default: ../data/)")
    args = parser.parse_args()

    data_dir = os.path.abspath(args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Data directory: {data_dir}\n")

    download_mnist(data_dir)
    print()
    download_cifar10(data_dir)

    print("\nAll datasets downloaded successfully.")
    print(f"  MNIST:    ~11 MB  ({os.path.join(data_dir, 'MNIST')})")
    print(f"  CIFAR-10: ~170 MB ({os.path.join(data_dir, 'cifar-10-batches-py')})")


if __name__ == "__main__":
    main()
