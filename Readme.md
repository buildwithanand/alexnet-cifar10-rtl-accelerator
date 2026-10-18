# AlexNet CIFAR-10 + INT8 Export Project

This repository contains a compact, PyTorch-based AlexNet-inspired CNN for CIFAR-10 classification, with a separate quantization/export pipeline intended for FPGA/RTL-oriented deployment workflows.

The project has two main goals:

- Train and evaluate a lightweight image classifier on CIFAR-10
- Convert the trained model to INT8 and export quantized weights/metadata for hardware-oriented use

## Project overview

The repository is organized as follows:

- `cifar10_alexnet/model.py` defines the compact CNN architecture
- `cifar10_alexnet/train.py` trains the model and saves checkpoints
- `cifar10_alexnet/predict.py` evaluates a selected image range and writes prediction summaries
- `cifar10_alexnet/check_weighs.py` inspects saved checkpoint precision
- `Quantization_FP32_to_INT8/quantize_INT8.py` performs static INT8 quantization and exports hardware-friendly artifacts

## Repository structure

```text
alexnet_pynq/
├── README.md
├── Readme.md                 # compatibility copy of the project documentation
├── .gitignore
├── cifar10_alexnet/
│   ├── model.py
│   ├── train.py
│   ├── predict.py
│   ├── check_weighs.py
│   ├── requirements.txt
│   ├── checkpoints/
│   │   ├── best_model.pth
│   │   └── final_weights.pth
│   ├── data/
│   │   └── cifar-10-batches-py/
│   └── results/
│       ├── history.csv
│       └── prediction_analysis/
│           ├── predictions_0_4999.csv
│           └── generated plots ...
├── Quantization_FP32_to_INT8/
│   ├── quantize_INT8.py
│   └── outputs/
│       ├── alexnet_int8_model.pth
│       ├── alexnet_int8_state.pth
│       ├── alexnet_int8_weights.npz
│       ├── alexnet_int8_metadata.json
│       └── quantization_report.json
├── RTL_Design/              # hardware-side project directory placeholder
├── state/                   # local workspace/state artifacts
└── .venv/                   # local Python environment (optional)
```

## Model architecture

The network in `cifar10_alexnet/model.py` is a lightweight AlexNet-inspired CNN designed for 32x32 RGB CIFAR-10 images.

It includes:

- 3-channel input images
- A convolutional feature extractor with ReLU activations
- Max-pooling stages to reduce spatial resolution
- A fully connected classifier head
- A 10-class output layer matching CIFAR-10 classes

## Setup

From the repo root, create and activate a Python environment and install dependencies:

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

The required packages are:

- `torch`
- `torchvision`
- `matplotlib`
- `numpy`
- `Pillow`

## Data preparation

The training pipeline expects the CIFAR-10 dataset to be available in:

```text
cifar10_alexnet/data/cifar-10-batches-py/
```

If the dataset is missing, download it and extract it into `cifar10_alexnet/data/` before running training.

Example:

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', download=True)"
```

This creates the expected CIFAR-10 directory structure under the local `./data` folder.

## Training

Run training from inside the `cifar10_alexnet` directory:

```bash
python train.py
```

The training script performs the following tasks:

- Loads CIFAR-10 and splits it into train/validation sets
- Applies augmentation only to the training split
- Uses a fixed random seed for reproducibility
- Trains with AdamW and cosine learning-rate scheduling
- Enables automatic mixed precision on CUDA when available
- Saves the best validation checkpoint and final weights
- Stops early when validation performance plateaus

Key training settings include:

- `BATCH_SIZE = 128`
- `EPOCHS = 50`
- `LEARNING_RATE = 0.0005`
- `VALIDATION_SIZE = 5000`
- `PATIENCE = 8`
- `SEED = 42`

Outputs are saved under:

- `cifar10_alexnet/checkpoints/best_model.pth`
- `cifar10_alexnet/checkpoints/final_weights.pth`
- `cifar10_alexnet/results/history.csv`

## Prediction analysis

Use the prediction script to evaluate a selected subset of test images and generate plots:

```bash
python predict.py --start 0 --count 5000 --batch-size 128
```

This script:

- Loads the saved model checkpoint
- Evaluates the requested image range
- Writes a CSV containing prediction and confidence data
- Produces plots for running accuracy, correct vs incorrect predictions, and per-class accuracy

Generated outputs are written to:

```text
cifar10_alexnet/results/prediction_analysis/
```

Typical files include:

```text
prediction_analysis/predictions_0_4999.csv
prediction_analysis/running_accuracy_0_4999.png
prediction_analysis/prediction_outcomes_0_4999.png
prediction_analysis/class_accuracy_0_4999.png
```

## Checkpoint inspection

Inspect checkpoint precision and tensor dtypes with:

```bash
python check_weighs.py
```

This script reports:

- Parameter count
- Storage size
- Dtypes present in the checkpoint
- Whether the weights are mostly FP32, FP16, INT8, or mixed precision

## INT8 quantization workflow

The repository also includes a post-training static INT8 quantization pipeline for the trained model.

Run it from the project root:

```bash
python Quantization_FP32_to_INT8/quantize_INT8.py
```

The script:

- Loads the trained FP32 checkpoint from `cifar10_alexnet/checkpoints/best_model.pth`
- Calibrates activations on a subset of CIFAR-10 data
- Converts the model to INT8 using PyTorch FX quantization
- Compares FP32 and INT8 accuracy
- Exports a quantized model and metadata intended for hardware deployment

Expected output artifacts:

```text
Quantization_FP32_to_INT8/outputs/
├── alexnet_int8_model.pth
├── alexnet_int8_state.pth
├── alexnet_int8_weights.npz
├── alexnet_int8_metadata.json
├── quantization_report.json
└── extra generated files as needed
```

The exported metadata includes layer-wise INT8 weight arrays, scales, zero points, and quantization metadata that are useful when preparing workloads for FPGA/RTL acceleration.

## Typical workflow

1. Place CIFAR-10 in `cifar10_alexnet/data/`
2. Run `python train.py` from `cifar10_alexnet/`
3. Review `results/history.csv`
4. Run `python predict.py --start 0 --count 5000`
5. Inspect generated plots and predictions
6. Run `python check_weighs.py` to inspect checkpoint precision
7. Run `python Quantization_FP32_to_INT8/quantize_INT8.py` to create the INT8 export

## Notes

- This project is designed to be Windows-friendly and uses `num_workers=2` in the training pipeline by default.
- Mixed precision is enabled automatically when CUDA is available.
- The quantization/export stage is intended to support hardware-oriented deployment preparation and is separate from the training-only workflow.
- The repository includes both a conventional software training path and a quantized model export path.

This project is intended as a compact, readable example of a CIFAR-10 training and evaluation pipeline built around an AlexNet-inspired PyTorch model, with an additional INT8 export flow for deployment-oriented experimentation.
