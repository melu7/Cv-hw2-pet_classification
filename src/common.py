"""
common.py - shared code used by all scripts of Homework 2.

It contains:
  * the fixed settings (seed, batch size, folders)
  * random seeds and deterministic settings
  * data loading for the Oxford-IIIT Pet dataset (37 breeds)
  * the stratified train/validation split
  * preprocessing / augmentation
  * the MobileNetV3-Small model with a new 37-output final layer
"""
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

# ----------------------------------------------------------------------
# Fixed settings (identical for all three approaches)
# ----------------------------------------------------------------------
SEED = 42
BATCH_SIZE = 64
NUM_CLASSES = 37
VAL_FRACTION = 0.2  # 80% of the official trainval set for training, 20% for validation

# Folders (relative to the repository root)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_DIR, "data")                  # dataset is downloaded here
RESULTS_DIR = os.path.join(REPO_DIR, "results")            # metrics, plots, split file
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")  # best neural-network weights
MODEL_DIR = os.path.join(RESULTS_DIR, "models")            # scikit-learn models + scaler
SPLIT_FILE = os.path.join(RESULTS_DIR, "split_indices.json")
for folder in (RESULTS_DIR, CHECKPOINT_DIR, MODEL_DIR):
    os.makedirs(folder, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS = MobileNet_V3_Small_Weights.DEFAULT  # ImageNet-1K pre-trained weights


# ----------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------
def set_seed(seed=SEED):
    """Fix the random numbers of Python, NumPy and PyTorch and ask for deterministic operations."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # warn_only=True: print a warning instead of crashing if an operation cannot be deterministic
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id):
    """Makes the random augmentation inside each data-loading worker repeatable."""
    np.random.seed(SEED + worker_id)
    random.seed(SEED + worker_id)


def sync():
    """Wait for the GPU to finish, so that time measurements are correct."""
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
def get_transforms():
    """Return (train_transform, eval_transform)."""
    # Official preprocessing of the pre-trained weights: resize 256, center crop 224, ImageNet normalization
    eval_tf = WEIGHTS.transforms()
    # Training only: random augmentation, then the SAME normalization
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=eval_tf.mean, std=eval_tf.std),
    ])
    return train_tf, eval_tf


def get_split(trainval_labels):
    """Return (train_idx, val_idx). Loads the saved split if it exists, otherwise creates and saves it."""
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            split = json.load(f)
        return np.array(split["train_indices"]), np.array(split["val_indices"])

    all_indices = np.arange(len(trainval_labels))
    train_idx, val_idx = train_test_split(
        all_indices,
        test_size=VAL_FRACTION,
        stratify=trainval_labels,   # same breed proportions in train and validation
        random_state=SEED,
    )
    with open(SPLIT_FILE, "w") as f:
        json.dump({"seed": SEED, "val_fraction": VAL_FRACTION,
                   "train_indices": train_idx.tolist(),
                   "val_indices": val_idx.tolist()}, f)
    return train_idx, val_idx


def get_data():
    """Download the dataset (first time only) and return datasets, labels, split and data loaders."""
    train_tf, eval_tf = get_transforms()
    options = dict(root=DATA_DIR, target_types="category", download=True)  # category = 37 breeds

    trainval_aug = OxfordIIITPet(split="trainval", transform=train_tf, **options)    # with augmentation
    trainval_plain = OxfordIIITPet(split="trainval", transform=eval_tf, **options)   # without augmentation
    test_plain = OxfordIIITPet(split="test", transform=eval_tf, **options)           # official test set

    trainval_labels = np.array(trainval_plain._labels)   # breed number (0-36) of each image
    test_labels = np.array(test_plain._labels)
    train_idx, val_idx = get_split(trainval_labels)

    g = torch.Generator()
    g.manual_seed(SEED)   # makes the shuffling order repeatable
    loaders = {
        # training images WITH augmentation, shuffled
        "train": DataLoader(Subset(trainval_aug, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=2, worker_init_fn=seed_worker, generator=g),
        # training images WITHOUT augmentation (used to measure training accuracy and extract features)
        "train_plain": DataLoader(Subset(trainval_plain, train_idx), batch_size=BATCH_SIZE,
                                  shuffle=False, num_workers=2),
        "val": DataLoader(Subset(trainval_plain, val_idx), batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2),
        "test": DataLoader(test_plain, batch_size=BATCH_SIZE, shuffle=False, num_workers=2),
    }
    return {
        "loaders": loaders,
        "class_names": trainval_plain.classes,
        "trainval_labels": trainval_labels,
        "test_labels": test_labels,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------
def build_model():
    """Pre-trained MobileNetV3-Small whose final layer is replaced by a new layer with 37 outputs."""
    torch.manual_seed(SEED)   # the new layer starts from the same random values every time
    model = mobilenet_v3_small(weights=WEIGHTS)
    # model.classifier = [Linear(576->1024), Hardswish, Dropout, Linear(1024->1000)]
    model.classifier[3] = nn.Linear(1024, NUM_CLASSES)
    return model.to(DEVICE)


def count_trainable_params(model):
    """Number of parameters that are updated during training."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def predict(model, loader):
    """Return the predicted breed number for every image of the loader."""
    model.eval()
    all_preds = []
    with torch.no_grad():
        for images, _ in loader:
            outputs = model(images.to(DEVICE))
            all_preds.append(outputs.argmax(dim=1).cpu().numpy())
    return np.concatenate(all_preds)
