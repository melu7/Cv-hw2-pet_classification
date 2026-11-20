"""
evaluate.py - the single evaluation of all approaches on the official TEST set.

Usage:  python src/evaluate.py     (run it ONCE, after the other scripts)

Do not change any setting after looking at these results: that would be tuning on the test set.

Outputs (in results/): results_table.csv, test_predictions.npz, confusion_<approach>.png,
top_confusions.txt, examples_<approach>.png, pair_<approach>.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")   # draw figures into files (no window needed)
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torchvision.datasets import OxfordIIITPet

from common import (CHECKPOINT_DIR, DATA_DIR, DEVICE, MODEL_DIR, NUM_CLASSES, RESULTS_DIR, SEED,
                    build_model, get_data, predict, set_seed)

APPROACHES = {
    "full_finetune": "Full fine-tuning",
    "head_training": "Classifier-head training",
    "logistic_regression": "Features + logistic regression",
    "linear_svm": "Features + linear SVM",
}


def read_json(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run the training scripts first (see README).")
    with open(path) as f:
        return json.load(f)


def get_predictions(data):
    """Predict the test set with every approach. Also compute training accuracy and parameter counts."""
    loaders = data["loaders"]
    y_train = data["trainval_labels"][data["train_idx"]]
    predictions, train_acc, n_params = {}, {}, {}

    # Neural networks: load the best checkpoint (it was chosen on the validation set)
    for name in ["full_finetune", "head_training"]:
        model = build_model()
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, f"{name}_best.pth"),
                                         map_location=DEVICE))
        predictions[name] = predict(model, loaders["test"])
        train_acc[name] = float((predict(model, loaders["train_plain"]) == y_train).mean())

    # scikit-learn classifiers: saved features, scaler (fitted on train only) and models
    features = np.load(os.path.join(RESULTS_DIR, "features.npz"))
    assert np.array_equal(features["y_test"], data["test_labels"]), "test labels do not match"
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    X_test = scaler.transform(features["X_test"])
    feature_summary = read_json("feature_extraction_summary.json")
    for name in ["logistic_regression", "linear_svm"]:
        clf = joblib.load(os.path.join(MODEL_DIR, f"{name}.joblib"))
        predictions[name] = clf.predict(X_test)
        train_acc[name] = feature_summary["classifiers"][name]["train_acc"]
        n_params[name] = int(clf.coef_.size + clf.intercept_.size)
    return predictions, train_acc, n_params


def make_results_table(y_test, predictions, train_acc, n_params):
    full = read_json("full_finetune_summary.json")
    head = read_json("head_training_summary.json")
    feat = read_json("feature_extraction_summary.json")
    extract_time = feat["feature_extraction_time_sec"]
    lr_info = feat["classifiers"]["logistic_regression"]
    svm_info = feat["classifiers"]["linear_svm"]

    # name: (trainable parameters, time in seconds, validation accuracy)
    info = {
        "full_finetune": (full["trainable_params"], full["train_time_sec"], full["best_val_acc"]),
        "head_training": (head["trainable_params"], head["train_time_sec"], head["best_val_acc"]),
        "logistic_regression": (n_params["logistic_regression"],
                                extract_time + lr_info["fit_time_sec"], lr_info["val_acc"]),
        "linear_svm": (n_params["linear_svm"],
                       extract_time + svm_info["fit_time_sec"], svm_info["val_acc"]),
    }
    rows = []
    for name, (params, seconds, val_acc) in info.items():
        preds = predictions[name]
        rows.append({
            "Approach": APPROACHES[name],
            "Trainable params": params,
            "Time (s)": round(seconds, 1),
            "Train acc": round(train_acc[name], 3),
            "Val acc": round(val_acc, 3),
            "Test acc": round(accuracy_score(y_test, preds), 3),
            "Test macro F1": round(f1_score(y_test, preds, average="macro"), 3),
        })
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(RESULTS_DIR, "results_table.csv"), index=False)
    return table


def plot_confusion_matrices(y_test, predictions, class_names):
    """Save one confusion matrix (with breed names) per approach; return the raw count matrices."""
    matrices = {}
    for name, preds in predictions.items():
        cm = confusion_matrix(y_test, preds, labels=list(range(NUM_CLASSES)))
        matrices[name] = cm
        cm_fraction = cm / cm.sum(axis=1, keepdims=True)   # each row sums to 1

        plt.figure(figsize=(12, 10))
        plt.imshow(cm_fraction, cmap="Blues", vmin=0, vmax=1)
        plt.colorbar(label="Fraction of the true breed")
        plt.xticks(range(NUM_CLASSES), class_names, rotation=90, fontsize=7)
        plt.yticks(range(NUM_CLASSES), class_names, fontsize=7)
        plt.xlabel("Predicted breed")
        plt.ylabel("True breed")
        plt.title(f"Confusion matrix (test set): {APPROACHES[name]}")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"confusion_{name}.png"), dpi=200)
        plt.close()
    return matrices


def top_confusions(matrices, class_names, how_many=5):
    """Return {approach: [(true_breed_number, predicted_breed_number, count), ...]} and write a text file."""
    result, lines = {}, []
    for name, cm in matrices.items():
        errors = cm.copy()
        np.fill_diagonal(errors, 0)   # ignore correct predictions
        best = np.argsort(errors.flatten())[::-1][:how_many]
        result[name] = [(int(i // NUM_CLASSES), int(i % NUM_CLASSES), int(errors.flatten()[i])) for i in best]
        lines.append(f"{APPROACHES[name]}: top {how_many} confusions (true -> predicted)")
        for true_c, pred_c, count in result[name]:
            lines.append(f"  {class_names[true_c]} -> {class_names[pred_c]}: {count} images")
        lines.append("")
    text = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "top_confusions.txt"), "w") as f:
        f.write(text)
    print(text)
    return result


def show_examples(name, preds, y_test, class_names, test_images, n_each=4):
    """Random correct and wrong predictions (fixed seed, so the choice is repeatable and not cherry-picked)."""
    rng = np.random.RandomState(SEED)
    correct = rng.choice(np.where(preds == y_test)[0], n_each, replace=False)
    wrong = rng.choice(np.where(preds != y_test)[0], n_each, replace=False)

    fig, axes = plt.subplots(2, n_each, figsize=(3.2 * n_each, 7))
    for row, (indices, label) in enumerate([(correct, "Correct"), (wrong, "Wrong")]):
        for col, i in enumerate(indices):
            image, _ = test_images[int(i)]
            axes[row, col].imshow(image)
            axes[row, col].axis("off")
            axes[row, col].set_title(
                f"{label}\nTrue: {class_names[y_test[i]]}\nPred: {class_names[preds[i]]}",
                fontsize=8, color="green" if row == 0 else "red")
    fig.suptitle(f"{APPROACHES[name]}: random test examples")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"examples_{name}.png"), dpi=200)
    plt.close()


def show_pair(name, preds, y_test, class_names, test_images, true_c, pred_c, n=4):
    """Top row: images of the true breed predicted as another breed. Bottom row: real images of that breed."""
    wrong = np.where((y_test == true_c) & (preds == pred_c))[0][:n]   # the first n, not hand-picked
    reference = np.where(y_test == pred_c)[0][:n]

    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 7))
    for col in range(n):
        for row, (indices, text) in enumerate([
                (wrong, f"True: {class_names[true_c]}\nPred: {class_names[pred_c]}"),
                (reference, f"Real {class_names[pred_c]}")]):
            axes[row, col].axis("off")
            if col < len(indices):
                image, _ = test_images[int(indices[col])]
                axes[row, col].imshow(image)
                axes[row, col].set_title(text, fontsize=8)
    fig.suptitle(f"{APPROACHES[name]}: most frequent confusion")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"pair_{name}.png"), dpi=200)
    plt.close()


def main():
    set_seed()
    data = get_data()
    class_names = data["class_names"]
    y_test = data["test_labels"]

    predictions, train_acc, n_params = get_predictions(data)
    np.savez(os.path.join(RESULTS_DIR, "test_predictions.npz"), y_test=y_test, **predictions)

    table = make_results_table(y_test, predictions, train_acc, n_params)
    print(table.to_string(index=False))
    print()

    matrices = plot_confusion_matrices(y_test, predictions, class_names)
    most_confused = top_confusions(matrices, class_names)

    # Original test pictures (no preprocessing) for the example figures
    test_images = OxfordIIITPet(root=DATA_DIR, split="test", target_types="category", download=True)
    for name, preds in predictions.items():
        show_examples(name, preds, y_test, class_names, test_images)
        true_c, pred_c, _ = most_confused[name][0]
        show_pair(name, preds, y_test, class_names, test_images, true_c, pred_c)
    print("Figures saved in", RESULTS_DIR)


if __name__ == "__main__":
    main()
