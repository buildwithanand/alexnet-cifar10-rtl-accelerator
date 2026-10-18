# AlexNet CIFAR-10 Project

This repository contains a compact AlexNet-inspired convolutional neural network for CIFAR-10 classification. The project is implemented in PyTorch and includes training, validation, checkpoint saving, prediction analysis, and output visualization.

The model is designed for the CIFAR-10 dataset (32x32 RGB images) and uses a lightweight CNN structure inspired by AlexNet, but adapted for smaller input size and reduced parameter count.

## Project overview

The code in this repo does the following:

- Builds a small CNN in `cifar10_alexnet/model.py`
- Trains the model on CIFAR-10 with augmentation and validation tracking in `cifar10_alexnet/train.py`
- Saves the best model checkpoint and final weights to `cifar10_alexnet/checkpoints/`
- Runs inference on selected CIFAR-10 image ranges and exports results in `cifar10_alexnet/predict.py`
- Writes training metrics to CSV and produces accuracy/diagnostic plots in `cifar10_alexnet/results/`
- Reports checkpoint tensor dtypes in `cifar10_alexnet/check_weighs.py`

## Repository structure

```text
alexnet_pynq/
├── Readme.md
├── cifar10_alexnet/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── check_weighs.py
│   ├── requirements.txt
│   ├── data/
│   │   └── cifar-10-batches-py/
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_weights.pth
│   └── results/
│       ├── history.csv
│       └── prediction_analysis/
│           └── predictions_0_4999.csv
├── Quantization_FP32_to_INT8/
│   └── (currently empty / reserved for quantization work)
└── .gitignore (if present in your workspace)
```

## Model architecture

The model defined in `cifar10_alexnet/model.py` is a compact CNN based on the AlexNet idea:

- Input: 3x32x32 CIFAR-10 image tensor
- Convolutional feature extractor with ReLU activations
- Max-pooling after convolution blocks
- Fully connected classifier head
- Final output layer with 10 classes

The model returns logits for the 10 CIFAR-10 classes.

## Setup

From the project root, create and activate a virtual environment, then install dependencies:

```bash
cd cifar10_alexnet
python -m venv .venv
```

On Windows:

```powershell
.\.venv\Scripts\activate
pip install -r requirements.txt
```

On Linux/macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Dependencies

The project uses:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `Pillow`

These are listed in `cifar10_alexnet/requirements.txt`.

## Data preparation

The training script expects CIFAR-10 to already be available in the dataset folder.

The expected path is:

```text
cifar10_alexnet/data/cifar-10-batches-py/
```

If the data is not present, download it and extract it into `cifar10_alexnet/data/` before training.

A simple option is:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', download=True)"
```

This will create the CIFAR-10 dataset structure under `./data`.

## Training

Run training from the `cifar10_alexnet` directory:

```bash
python train.py
```

What the training pipeline does:

- Loads CIFAR-10 and splits the training set into train/validation sets
- Applies augmentation to training data only
- Uses a reproducible random seed and deterministic validation/test transforms
- Trains the model with AdamW and cosine learning-rate scheduling
- Uses automatic mixed precision on CUDA if available
- Saves best validation checkpoint and final weights
- Stops early when validation accuracy does not improve enough for several epochs

Key training settings in `train.py` include:

- `BATCH_SIZE = 128`
- `EPOCHS = 50`
- `LEARNING_RATE = 0.0005`
- `VALIDATION_SIZE = 5000`
- `PATIENCE = 8`
- `SEED = 42`

Outputs:

- `checkpoints/best_model.pth`
- `checkpoints/final_weights.pth`
- `results/history.csv`

## Prediction analysis

Use the prediction script to evaluate a selected subset of test images and generate summary plots:

```bash
python predict.py --start 0 --count 5000 --batch-size 128
```

This script:

- Loads the saved model checkpoint
- Evaluates test images in the requested index range
- Writes a CSV with per-image predictions and confidence scores
- Saves plots for:
  - running accuracy across the batch
  - correct vs incorrect predictions
  - per-class accuracy

The default output directory is:

```text
cifar10_alexnet/results/prediction_analysis/
```

Example outputs include files such as:

```text
results/prediction_analysis/predictions_0_4999.csv
results/prediction_analysis/running_accuracy_0_4999.png
results/prediction_analysis/prediction_outcomes_0_4999.png
results/prediction_analysis/class_accuracy_0_4999.png
```

## Inspecting checkpoint precision

To inspect the parameter dtypes in the saved weights:

```bash
python check_weighs.py
```

This reports:

- parameter count
- storage size
- dtypes used in the checkpoint
- whether the weights are FP32, FP16, INT8, or mixed

## Notes

- The project is Windows-friendly and uses `num_workers=2` by default in training.
- Mixed precision is enabled automatically on CUDA devices when available.
- The `Quantization_FP32_to_INT8` folder is currently empty and is intended for future FP32-to-INT8 quantization work.
- The saved model checkpoint includes the model state dict and metadata such as class names and validation accuracy.

## Typical workflow

1. Prepare CIFAR-10 data under `cifar10_alexnet/data/`
2. Run `python train.py`
3. Review `results/history.csv`
4. Run `python predict.py --start 0 --count 5000`
5. Inspect generated CSV and plots
6. Use `python check_weighs.py` to inspect checkpoint precision

This repository is intended as a compact, readable example of a CIFAR-10 training and evaluation pipeline built around a lightweight AlexNet-inspired PyTorch model.
