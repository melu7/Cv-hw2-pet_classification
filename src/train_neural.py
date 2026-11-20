"""
train_neural.py - Approach 1 (full fine-tuning) and Approach 2 (classifier-head training).

Usage:
    python src/train_neural.py --approach full    # train the whole network
    python src/train_neural.py --approach head    # freeze the convolutional layers, train the classifier

Only the training and validation sets are used here. The test set is NOT touched.
Outputs: results/checkpoints/<name>_best.pth and results/<name>_summary.json
"""
import argparse
import json
import os
import time

import torch
import torch.nn as nn

from common import (BATCH_SIZE, CHECKPOINT_DIR, DEVICE, RESULTS_DIR, build_model,
                    count_trainable_params, get_data, set_seed, sync)

# Hyperparameters of each approach
SETTINGS = {
    "full": {"name": "full_finetune", "lr": 1e-4, "epochs": 10},
    "head": {"name": "head_training", "lr": 1e-3, "epochs": 10},
}


def run_one_epoch(model, loader, loss_fn, optimizer=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    if training:
        model.train()
        # If the feature extractor is frozen, keep it in eval mode so that
        # its BatchNorm statistics do not change either
        if not any(p.requires_grad for p in model.features.parameters()):
            model.features.eval()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):   # no gradients are needed when evaluating
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)              # forward pass: 37 scores per image
            loss = loss_fn(outputs, labels)      # how wrong are the scores?

            if training:
                optimizer.zero_grad()            # clear old gradients
                loss.backward()                  # compute how to change each weight
                optimizer.step()                 # update the weights

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total   # average loss, accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", choices=["full", "head"], required=True)
    args = parser.parse_args()
    cfg = SETTINGS[args.approach]
    name = cfg["name"]

    set_seed()
    data = get_data()
    train_loader = data["loaders"]["train"]
    val_loader = data["loaders"]["val"]

    model = build_model()   # pre-trained MobileNetV3-Small with a new 37-output last layer

    if args.approach == "head":
        # Freeze the convolutional feature extractor: its weights will not change
        for param in model.features.parameters():
            param.requires_grad = False

    if args.approach == "head":
        print("Trainable layers (everything else is frozen):")
        for layer_name, param in model.named_parameters():
            if param.requires_grad:
                print("  ", layer_name, tuple(param.shape))
    else:
        print("All layers are trainable (full fine-tuning).")
    n_trainable = count_trainable_params(model)
    print("Trainable parameters:", n_trainable)

    loss_fn = nn.CrossEntropyLoss()
    # The optimizer only receives the parameters that are allowed to change
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg["lr"])

    history = []
    best_val_acc, best_epoch = 0.0, 0
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{name}_best.pth")

    sync()
    start = time.time()
    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc = run_one_epoch(model, train_loader, loss_fn, optimizer)
        val_loss, val_acc = run_one_epoch(model, val_loader, loss_fn)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                        "val_loss": val_loss, "val_acc": val_acc})
        print(f"Epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} "
              f"| val loss {val_loss:.3f} acc {val_acc:.3f}")

        # Model-selection rule: keep the epoch with the best VALIDATION accuracy
        # (the earliest epoch if there is a tie). The test set plays no role.
        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, epoch
            torch.save(model.state_dict(), checkpoint_path)
    sync()
    train_time = time.time() - start   # includes the validation passes

    summary = {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "train_time_sec": train_time,
        "trainable_params": n_trainable,
        "settings": {
            "loss": "CrossEntropyLoss",
            "optimizer": "Adam",
            "learning_rate": cfg["lr"],
            "batch_size": BATCH_SIZE,
            "epochs": cfg["epochs"],
            "model_selection": "highest validation accuracy (earliest epoch on ties)",
        },
        "history": history,
    }
    with open(os.path.join(RESULTS_DIR, f"{name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nBest epoch: {best_epoch} (val acc {best_val_acc:.3f})")
    print(f"Training time: {train_time:.0f} seconds")


if __name__ == "__main__":
    main()
