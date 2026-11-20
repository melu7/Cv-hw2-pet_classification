"""
prepare_data.py - downloads the dataset, creates the stratified split,
counts images per breed and records the software versions.

Usage:  python src/prepare_data.py
Outputs (in results/): split_indices.json, class_counts.csv, environment.txt
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision

from common import NUM_CLASSES, RESULTS_DIR, SEED, get_data, set_seed


def save_environment():
    """Write the exact versions and seed to a text file (needed for the README and report)."""
    lines = [
        f"Python: {sys.version}",
        f"PyTorch: {torch.__version__}",
        f"TorchVision: {torchvision.__version__}",
        f"scikit-learn: {sklearn.__version__}",
        f"NumPy: {np.__version__}",
        f"Pandas: {pd.__version__}",
        f"GPU available: {torch.cuda.is_available()}",
        f"GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}",
        f"Random seed: {SEED}",
        "Deterministic algorithms: enabled with warn_only=True (cudnn.deterministic=True, benchmark=False)",
    ]
    with open(os.path.join(RESULTS_DIR, "environment.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    set_seed()
    save_environment()

    data = get_data()   # downloads the dataset and creates/loads the split
    trainval_labels = data["trainval_labels"]
    counts = pd.DataFrame({
        "breed": data["class_names"],
        "train": np.bincount(trainval_labels[data["train_idx"]], minlength=NUM_CLASSES),
        "validation": np.bincount(trainval_labels[data["val_idx"]], minlength=NUM_CLASSES),
        "test": np.bincount(data["test_labels"], minlength=NUM_CLASSES),
    })
    counts.to_csv(os.path.join(RESULTS_DIR, "class_counts.csv"), index=False)

    print("\nImages per breed in each split:")
    print(counts.to_string())
    print("\nTotals -> train:", counts["train"].sum(), " validation:", counts["validation"].sum(),
          " test:", counts["test"].sum())
    print("Images per breed in train: min", counts["train"].min(), "max", counts["train"].max())


if __name__ == "__main__":
    main()
