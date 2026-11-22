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

## Repository structure

```
README.md             this file
requirements.txt      package versions
src/
  common.py           seeds, data loading, split, preprocessing, model
  prepare_data.py     downloads data, creates the split, class counts, versions
  train_neural.py     approach 1 (--approach full) and approach 2 (--approach head)
  extract_features.py approach 3 (features + scikit-learn)
  evaluate.py         single test evaluation, tables and figures
results/              metrics, figures, split file, checkpoints, scikit-learn models
```

## Dataset

- **Oxford-IIIT Pet Dataset**, 37 cat and dog breeds. Official page: https://www.robots.ox.ac.uk/~vgg/data/pets/
- Citation: O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, IEEE CVPR 2012.
- License: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0). The copyright of the individual images remains with their original owners. See the official page above for the terms.
- The dataset is **not** included in this repository. `torchvision.datasets.OxfordIIITPet` downloads it automatically into `data/` the first time you run a script.
- Splits: the official `trainval` (3,680 images) and `test` (3,669 images) partitions are used. `trainval` is split 80/20 into **train (2,944)** and **validation (736)**, stratified by breed, with seed 42. The test set is never used for training or model selection.
- Images per breed in every split: `results/class_counts.csv`. The split itself: `results/split_indices.json`.

## Preprocessing and augmentation

- Pre-trained weights: `MobileNet_V3_Small_Weights.DEFAULT` (ImageNet-1K), from `torchvision`.
- Evaluation (validation, test, and feature extraction): the weights' own transform: resize 256, center crop 224, ImageNet mean/std normalization.
- Training (approaches 1 and 2 only): random resized crop to 224 (scale 0.7 to 1.0), random horizontal flip, then the same normalization. No augmentation on validation or test.
- No class balancing: the dataset is nearly balanced (74 to 80 training images per breed, see `class_counts.csv`).

## Training setup

|                 | Full fine-tuning               | Head training                  | Features + logistic regression / linear SVM               |
|-----------------|--------------------------------|--------------------------------|-----------------------------------------------------------|
| Loss            | cross entropy                  | cross entropy                  | logistic loss / squared hinge loss (scikit-learn)         |
| Optimizer       | Adam                           | Adam                           | scikit-learn solvers (lbfgs / liblinear)                  |
| Learning rate   | 1e-4                           | 1e-3                           | -                                                         |
| Batch size      | 64                             | 64                             | - (64 for feature extraction)                             |
| Epochs          | 10                             | 10                             | -                                                         |
| Scaling         | -                              | -                              | StandardScaler fitted on training features only           |
| Hyperparameter  | -                              | -                              | C in {0.001, 0.01, 0.1, 1}, chosen on validation accuracy |
| Model selection | best validation accuracy epoch | best validation accuracy epoch | best validation accuracy C                                |
| Selected        | epoch 9                        | epoch 10                       | C = 0.01 (both classifiers)                               |

## Environment

Run on Google Colab with a Tesla T4 GPU. Versions of the reported run (also in `results/environment.txt`): Python 3.13.16, PyTorch 2.14.1 (CUDA 13.0), TorchVision 0.29.1, scikit-learn 1.6.1, NumPy 2.4.6, Pandas 2.2.3.

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

Run `evaluate.py` only once, after the other scripts, because it evaluates on the test set.
`extract_features.py` writes `results/features.npz` (about 17 MB), which `evaluate.py` needs. This file is not uploaded to GitHub, so run the scripts in the order above.

Outputs are written to `results/`:

- `results_table.csv`: test accuracy, macro F1, train/validation accuracy, trainable parameters, time
- `confusion_<approach>.png`: confusion matrices with breed names
- `top_confusions.txt`: most frequent confusions
- `examples_<approach>.png`, `pair_<approach>.png`: example predictions
- `*_summary.json`, `feature_extraction_summary.json`: selected epoch / C and hyperparameters
- `checkpoints/`: best network weights; `models/`: scikit-learn models and scaler
- `split_indices.json`, `class_counts.csv`, `environment.txt`

The checkpoints (about 6 MB each) are included for academic reproducibility. They are derived from torchvision's ImageNet-1K pre-trained weights and the Oxford-IIIT Pet training images, so check those licenses before any other use.

## Reproducibility notes

- Seed 42 is set for Python, NumPy and PyTorch (CPU and GPU), for the data split, and for the data loaders.
- Deterministic operations are enabled (`cudnn.deterministic=True`, `cudnn.benchmark=False`, `torch.use_deterministic_algorithms(True, warn_only=True)`). `warn_only=True` means some GPU operations may still be non-deterministic.
- Remaining sources of variation: GPU operations, the GPU type, library versions, and multi-process data loading. Two runs can differ slightly (the neural-network results varied by up to one point of test accuracy between runs; the scikit-learn results were identical). Training times depend on the machine.
- The test set is used once, in `evaluate.py`. It is not used for early stopping, feature selection, threshold selection or hyperparameter tuning.

## Results

Test set results (3,669 images), from `results/results_table.csv`. Train accuracy is measured on unaugmented training images with the selected model. Time is the 10-epoch training time (including validation passes) for the neural networks, and feature extraction (40 s) plus the fit of the chosen classifier for the scikit-learn rows.

| Approach | Trainable params | Time (s) | Train acc | Val acc | Test acc | Test macro F1 |
|---|---|---|---|---|---|---|
| Full fine-tuning | 1,555,781 | 197.6 | 0.963 | 0.864 | 0.800 | 0.797 |
| Classifier-head training | 628,773 | 186.7 | 0.986 | 0.894 | 0.850 | 0.848 |
| Features + logistic regression | 21,349 | 43.8 | 0.988 | 0.876 | 0.857 | 0.855 |
| Features + linear SVM | 21,349 | 52.2 | 0.999 | 0.875 | 0.833 | 0.831 |

Head training, logistic regression and the SVM are within about 2 points of each other, and one point is about 37 test images, so small differences should not be over-interpreted. The full discussion is in the PDF report.

## Limitations

- One run per approach and one learning rate per neural approach, with no further tuning. An earlier exploratory run of the same pipeline gave test accuracies of 0.790 (full fine-tuning), 0.847 (head training), 0.857 (logistic regression) and 0.833 (SVM), so differences of 1 to 2 points are within run-to-run noise. The numbers above come from the run whose files are in `results/`.
- Head training reached its best validation accuracy at the last epoch (10), so longer training might help.
- The validation set has only 736 images, which makes model selection noisy.
- Neural training times include the validation passes. The scikit-learn times exclude the search over C. Timings come from a shared Colab T4 GPU and are approximate.
- Similar breeds are hard to separate, and the labels may be ambiguous. ImageNet already contains many pets, so conclusions may not transfer to very different image domains.

## Use of AI assistance

While preparing this work I used Claude (Anthropic) as a study and coding assistant. Claude explained the concepts, helped me plan the experiments, wrote the first version of the Python scripts in this repository, helped with the Google Colab and GitHub setup, and drafted the text of the report from my results. I ran all the experiments myself on Google Colab and checked the outputs they produced. I take responsibility for the content of this work.

## References

- O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, CVPR 2012.
- A. Howard et al., *Searching for MobileNetV3*, ICCV 2019.
- PyTorch / TorchVision and scikit-learn documentation.
