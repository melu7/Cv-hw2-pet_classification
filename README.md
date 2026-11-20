# Homework 2: Image Classification with Transfer Learning

Computer Vision, Fall 2026 - Oxford-IIIT Pet (37 breeds) with a pre-trained MobileNetV3-Small.

Author: Melat Habtemariyam Ekubay (GitHub: melu7)

## What this project does

It compares three ways of adapting an ImageNet pre-trained **MobileNetV3-Small** to the 37 breed classes of the Oxford-IIIT Pet dataset.
All three use the same data splits, preprocessing and evaluation.

| # | Approach | What is trained |
|---|----------|-----------------|
| 1 | Full fine-tuning | All layers of the network (convolutional layers and classifier) |
| 2 | Classifier-head training | Only the two fully connected layers `classifier.0` (576->1024) and `classifier.3` (1024->37). `features` (convolutional layers) are frozen |
| 3 | Frozen features + scikit-learn | Nothing in the network. 576-dim features (after pooling) feed a logistic regression and a linear SVM |

The final layer (`classifier.3`) is replaced by a new layer with 37 outputs in approaches 1 and 2.

## Dataset

- **Oxford-IIIT Pet Dataset**, 37 cat and dog breeds. Official page: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Citation: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, IEEE CVPR 2012.
- License: **TODO: copy the license statement from the official dataset page before submitting.**
- The dataset is **not** included in this repository. `torchvision.datasets.OxfordIIITPet` downloads it automatically into `data/` the first time you run a script.
- Splits: the official `trainval` (3,680 images) and `test` (3,669 images) partitions are used.
  `trainval` is split 80/20 into **train (2,944)** and **validation (736)**, stratified by breed, with seed 42.
  The test set is never used for training or model selection.
- Images per breed in every split: `results/class_counts.csv`. The split itself: `results/split_indices.json`.

## Preprocessing and augmentation

- Pre-trained weights: `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K), from `torchvision`.
- Evaluation (validation, test, and feature extraction): the weights' own transform: resize 256, center crop 224, ImageNet mean/std normalization.
- Training (approaches 1 and 2 only): random resized crop to 224 (scale 0.7 to 1.0), random horizontal flip, then the same normalization. No augmentation on validation or test.
- No class balancing: the dataset is nearly balanced (74 to 80 training images per breed, see `class_counts.csv`).

## Training setup

| | Full fine-tuning | Head training | Features + logistic regression / linear SVM |
|---|---|---|---|
| Loss | cross entropy | cross entropy | logistic loss / squared hinge loss (scikit-learn) |
| Optimizer | Adam | Adam | scikit-learn solvers |
| Learning rate | 1e-4 | 1e-3 | - |
| Batch size | 64 | 64 | - |
| Epochs | 10 | 10 | - |
| Scaling | - | - | StandardScaler fitted on training features only |
| Hyperparameter | - | - | C in {0.001, 0.01, 0.1, 1}, chosen on validation accuracy |
| Model selection | best validation accuracy epoch | best validation accuracy epoch | best validation accuracy C |

## Environment

Run on Google Colab with a Tesla T4 GPU. The exact versions of the final run are written to `results/environment.txt`.
Versions used when developing: Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0), TorchVision 0.29.1, scikit-learn 1.6.1, NumPy 2.4.6.

```
pip install -r requirements.txt
```

## How to reproduce everything

Run these commands from the repository root, in this order:

```
python src/prepare_data.py                      # download data, create split, class counts, versions
python src/train_neural.py --approach full      # Approach 1
python src/train_neural.py --approach head      # Approach 2
python src/extract_features.py                  # Approach 3 (features + scikit-learn)
python src/evaluate.py                          # one evaluation on the test set, tables and figures
```

Outputs are written to `results/`:

- `results_table.csv`: test accuracy, macro F1, train/validation accuracy, trainable parameters, time
- `confusion_<approach>.png`: confusion matrices with breed names
- `top_confusions.txt`: most frequent confusions
- `examples_<approach>.png`, `pair_<approach>.png`: example predictions
- `*_summary.json`, `feature_extraction_summary.json`: selected epoch / C and hyperparameters
- `checkpoints/`: best network weights; `models/`: scikit-learn models and scaler
- `split_indices.json`, `class_counts.csv`, `environment.txt`

## Reproducibility notes

- Seed 42 is set for Python, NumPy and PyTorch (CPU and GPU), for the data split, and for the data loaders.
- Deterministic operations are enabled (`cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`).
  `warn_only=True` means some GPU operations may still be non-deterministic.
- Remaining sources of variation: GPU operations, the GPU type, library versions, and multi-process data loading.
  Two runs can differ slightly (small differences in loss and accuracy). Training times depend on the machine.
- The test set is used once, in `evaluate.py`. It is not used for early stopping, feature selection, threshold selection or hyperparameter tuning.

## Results

See `results/results_table.csv` and the figures in `results/`.

TODO: paste the final results table here after running `evaluate.py`.

## Limitations

TODO: write after the final results (single run per approach, one learning rate per approach, small validation set, one dataset).

## Use of AI assistance

TODO: adapt to your university's rules. Suggested wording: "I used Claude (Anthropic) to help plan the project, explain the concepts and draft the code structure. I ran all experiments myself and checked the results."

## References

- Parkhi et al., *Cats and Dogs*, CVPR 2012.
- Howard et al., *Searching for MobileNetV3*, ICCV 2019.
- PyTorch / TorchVision and scikit-learn documentation.
