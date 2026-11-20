"""
extract_features.py - Approach 3: frozen pre-trained features + scikit-learn classifiers.

Usage:  python src/extract_features.py

Steps:
  1. Run all images once through the FROZEN MobileNetV3-Small (convolutional layers + pooling)
     -> one vector of 576 numbers per image.
  2. Scale the features (scaler fitted on TRAINING features only).
  3. Train logistic regression and a linear SVM on the training features,
     choose the regularization strength C on the VALIDATION features.
The test features are extracted here but only used later, in evaluate.py.

Outputs: results/features.npz, results/models/*.joblib, results/feature_extraction_summary.json
"""
import json
import os
import time

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision.models import mobilenet_v3_small

from common import DEVICE, MODEL_DIR, RESULTS_DIR, SEED, WEIGHTS, get_data, set_seed, sync

C_VALUES = [0.001, 0.01, 0.1, 1]   # candidate regularization strengths (smaller C = simpler model)


def extract_features(feature_net, loader):
    """Return an array (N, 576) with the features of all images and an array with their labels."""
    all_features, all_labels = [], []
    with torch.no_grad():   # no training, so no gradients are needed
        for images, labels in loader:
            x = feature_net.features(images.to(DEVICE))   # convolutional layers
            x = feature_net.avgpool(x)                    # pooling -> (N, 576, 1, 1)
            x = torch.flatten(x, 1)                       # -> (N, 576)
            all_features.append(x.cpu().numpy())
            all_labels.append(labels.numpy())
    return np.concatenate(all_features), np.concatenate(all_labels)


def main():
    set_seed()
    data = get_data()
    loaders = data["loaders"]

    # Frozen pre-trained network: no new layers, nothing is trained
    feature_net = mobilenet_v3_small(weights=WEIGHTS).to(DEVICE)
    feature_net.eval()

    sync()
    start = time.time()
    X_train, y_train = extract_features(feature_net, loaders["train_plain"])   # no augmentation
    X_val, y_val = extract_features(feature_net, loaders["val"])
    X_test, y_test = extract_features(feature_net, loaders["test"])
    sync()
    extract_time = time.time() - start
    print("Train features:", X_train.shape, " Validation:", X_val.shape, " Test:", X_test.shape)
    print(f"Feature extraction time: {extract_time:.0f} seconds")

    np.savez(os.path.join(RESULTS_DIR, "features.npz"), X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val, X_test=X_test, y_test=y_test)

    # Scaling: the scaler learns mean/std from the TRAINING features only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.joblib"))

    classifiers = {
        "logistic_regression": lambda C: LogisticRegression(C=C, max_iter=2000, random_state=SEED),
        "linear_svm": lambda C: LinearSVC(C=C, max_iter=5000, random_state=SEED),
    }

    results = {}
    for name, make_clf in classifiers.items():
        print(f"\n{name}")
        best = {"val_acc": -1}
        best_model = None
        for C in C_VALUES:
            clf = make_clf(C)
            start = time.time()
            clf.fit(X_train_s, y_train)
            fit_time = time.time() - start

            train_acc = clf.score(X_train_s, y_train)
            val_acc = clf.score(X_val_s, y_val)
            print(f"  C={C:<6} train acc {train_acc:.3f} | val acc {val_acc:.3f} | fit time {fit_time:.1f}s")

            # Model-selection rule: highest VALIDATION accuracy
            if val_acc > best["val_acc"]:
                best = {"C": C, "train_acc": train_acc, "val_acc": val_acc, "fit_time_sec": fit_time}
                best_model = clf
        print(f"  -> chosen C = {best['C']} (val acc {best['val_acc']:.3f})")
        results[name] = best
        joblib.dump(best_model, os.path.join(MODEL_DIR, f"{name}.joblib"))

    summary = {"feature_extraction_time_sec": extract_time,
               "feature_dim": int(X_train.shape[1]),
               "classifiers": results}
    with open(os.path.join(RESULTS_DIR, "feature_extraction_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
