
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


# ============================================================
# 1. PROJECT PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

PROJECT_DIR = ROOT_DIR / "cifar10_alexnet"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
DATA_DIR = PROJECT_DIR / "data"

# Exact trained checkpoint from your current directory.
CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pth"

OUTPUT_DIR = SCRIPT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Import your existing AlexNet architecture.
sys.path.insert(0, str(PROJECT_DIR))
from model import AlexNetInspired



# ============================================================
# 2. CONFIGURATION
# ============================================================

NUM_CALIBRATION_IMAGES = 2000
CALIBRATION_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256

# This version assumes your train.py uses ToTensor() without
# Normalize(). Match these settings to your original train.py.
USE_NORMALIZATION = False

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


# ============================================================
# 3. LOAD YOUR TRAINED FP32 MODEL
# ============================================================

def load_fp32_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}\n"
            "Check that best_model.pth exists in checkpoints."
        )

    print("Loading checkpoint:", CHECKPOINT_PATH)

    try:
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        checkpoint = torch.load(
            CHECKPOINT_PATH,
            map_location="cpu",
        )

    if isinstance(checkpoint, torch.nn.Module):
        state_dict = checkpoint.state_dict()

    elif isinstance(checkpoint, dict):
        state_dict = None

        # Support common checkpoint formats.
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)

            if isinstance(value, torch.nn.Module):
                state_dict = value.state_dict()
                break

            if isinstance(value, dict):
                state_dict = value
                break

        # Alternatively, the checkpoint may directly contain weights.
        if state_dict is None:
            if checkpoint and all(
                isinstance(value, torch.Tensor)
                for value in checkpoint.values()
            ):
                state_dict = checkpoint
            else:
                raise ValueError(
                    "Could not identify the state_dict in best_model.pth. "
                    "Inspect its keys to determine its checkpoint format."
                )
    else:
        raise TypeError("Unsupported checkpoint format.")

    # Handle common DataParallel / compiled model prefixes.
    cleaned_state_dict = {}

    for key, value in state_dict.items():
        while key.startswith(("module.", "_orig_mod.")):
            if key.startswith("module."):
                key = key[len("module."):]
            elif key.startswith("_orig_mod."):
                key = key[len("_orig_mod."):]

        cleaned_state_dict[key] = value

    model = AlexNetInspired(num_classes=10).cpu()
    model.load_state_dict(cleaned_state_dict, strict=True)
    model.eval()

    print("FP32 model loaded successfully.")
    return model


# ============================================================
# 4. LOAD CIFAR-10 DATA
# ============================================================

def make_transform():
    operations = [transforms.ToTensor()]

    if USE_NORMALIZATION:
        operations.append(
            transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
        )

    return transforms.Compose(operations)


def load_datasets():
    if not (DATA_DIR / "cifar-10-batches-py").is_dir():
        raise FileNotFoundError(
            "CIFAR-10 batch folder not found.\n"
            f"Expected: {DATA_DIR / 'cifar-10-batches-py'}"
        )

    transform = make_transform()

    train_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=True,
        download=False,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root=str(DATA_DIR),
        train=False,
        download=False,
        transform=transform,
    )

    # Fixed random calibration subset for reproducibility.
    count = min(NUM_CALIBRATION_IMAGES, len(train_dataset))

    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(
        len(train_dataset),
        generator=generator,
    )[:count].tolist()

    calibration_loader = DataLoader(
        Subset(train_dataset, indices),
        batch_size=CALIBRATION_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=TEST_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Calibration images:", count)
    print("Test images:", len(test_dataset))

    return calibration_loader, test_loader


# ============================================================
# 5. EVALUATE MODEL ACCURACY
# ============================================================

@torch.inference_mode()
def evaluate(model, loader):
    model.eval()

    correct = 0
    total = 0

    for images, labels in loader:
        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


# ============================================================
# 6. EXPORT QUANTIZED WEIGHTS FOR FPGA DEVELOPMENT
# ============================================================

def export_weights(model):
    arrays = {}
    metadata = {
        "layers": {},
        "activation_qparams_from_graph": [],
    }

    exported_layers = 0

    for name, layer in model.named_modules():
        weight_attr = getattr(layer, "weight", None)

        try:
            weight = (
                weight_attr()
                if callable(weight_attr)
                else weight_attr
            )
        except (TypeError, RuntimeError):
            continue

        if not isinstance(weight, torch.Tensor):
            continue

        if not weight.is_quantized:
            continue

        key = name.replace(".", "__") or "root"

        # Store actual INT8/quantized integer values.
        integer_weight = weight.int_repr().cpu().numpy()
        arrays[f"{key}__weight"] = integer_weight

        info = {
            "shape": list(weight.shape),
            "integer_dtype": str(weight.int_repr().dtype),
            "qscheme": str(weight.qscheme()),
        }

        # Save the corresponding weight quantization parameters.
        if "per_channel" in str(weight.qscheme()):
            info["scales"] = (
                weight.q_per_channel_scales().cpu().tolist()
            )
            info["zero_points"] = (
                weight.q_per_channel_zero_points().cpu().tolist()
            )
            info["quantized_axis"] = weight.q_per_channel_axis()
        else:
            info["scale"] = float(weight.q_scale())
            info["zero_point"] = int(weight.q_zero_point())

        # Bias remains floating-point here; hardware deployment
        # requires conversion to the appropriate accumulator scale.
        bias_attr = getattr(layer, "bias", None)

        try:
            bias = bias_attr() if callable(bias_attr) else bias_attr
        except (TypeError, RuntimeError):
            bias = None

        if bias is not None:
            arrays[f"{key}__bias_fp32"] = (
                bias.detach().cpu().numpy().astype(np.float32)
            )

        # Record output activation quantization parameters where present.
        layer_scale = getattr(layer, "scale", None)
        layer_zero_point = getattr(layer, "zero_point", None)

        if layer_scale is not None and layer_zero_point is not None:
            try:
                info["output_scale"] = float(layer_scale)
                info["output_zero_point"] = int(layer_zero_point)
            except (TypeError, ValueError, RuntimeError):
                pass

        metadata["layers"][name] = info
        exported_layers += 1

        print(
            f"{name:35s} "
            f"dtype={integer_weight.dtype} "
            f"shape={tuple(weight.shape)}"
        )

    if exported_layers == 0:
        raise RuntimeError(
            "No quantized weights were found in the converted model."
        )

    # Record quantize_per_tensor parameters present in the FX graph.
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and "quantize_per_tensor" in str(node.target)
        ):
            args = node.args
            kwargs = node.kwargs

            qparams = {
                "node": node.name,
                "target": str(node.target),
            }

            if "scale" in kwargs:
                qparams["scale"] = kwargs["scale"]
            elif len(args) > 1 and isinstance(args[1], (int, float)):
                qparams["scale"] = args[1]

            if "zero_point" in kwargs:
                qparams["zero_point"] = kwargs["zero_point"]
            elif len(args) > 2 and isinstance(args[2], int):
                qparams["zero_point"] = args[2]

            if "dtype" in kwargs:
                qparams["dtype"] = str(kwargs["dtype"])
            elif len(args) > 3:
                qparams["dtype"] = str(args[3])

            metadata["activation_qparams_from_graph"].append(qparams)

    weights_path = OUTPUT_DIR / "alexnet_int8_weights.npz"
    np.savez_compressed(weights_path, **arrays)

    metadata_path = OUTPUT_DIR / "alexnet_int8_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\nWeight export complete:")
    print(weights_path)
    print(metadata_path)


# ============================================================
# 7. RUN POST-TRAINING STATIC INT8 QUANTIZATION
# ============================================================

def main():
    print("=" * 65)
    print("ALEXNET CIFAR-10 | FP32 TO INT8")
    print("=" * 65)

    print("PyTorch:", torch.__version__)
    print("Project:", PROJECT_DIR)
    print("Output:", OUTPUT_DIR)

    fp32_model = load_fp32_model()

    calibration_loader, test_loader = load_datasets()

    # Baseline test accuracy.
    fp32_accuracy = evaluate(fp32_model, test_loader)
    print(f"\nFP32 test accuracy: {fp32_accuracy:.2f}%")

    # Choose a quantization backend available in this installation.
    supported = torch.backends.quantized.supported_engines

    backend = next(
        (
            candidate
            for candidate in ("x86", "fbgemm", "qnnpack","onednn")
            if candidate in supported
        ),
        None,
    )

    if backend is None:
        raise RuntimeError(
            f"No supported quantization backend found: {supported}"
        )

    torch.backends.quantized.engine = backend
    print("Quantization backend:", backend)

    qconfig_mapping = get_default_qconfig_mapping(backend)
    example_inputs = (torch.randn(1, 3, 32, 32),)

    # Insert observers into a copy of the existing FP32 architecture.
    prepared_model = prepare_fx(
        copy.deepcopy(fp32_model),
        qconfig_mapping,
        example_inputs,
    )
    prepared_model.eval()

    # Calibration: determine activation quantization parameters.
    print("\nCalibrating activation ranges...")

    with torch.inference_mode():
        for images, _ in calibration_loader:
            prepared_model(images)

    # Convert supported operators to quantized operators.
    print("Converting model to quantized operators...")
    int8_model = convert_fx(prepared_model)
    int8_model.eval()

    # Test INT8 model on exactly the same test set.
    int8_accuracy = evaluate(int8_model, test_loader)
    difference = int8_accuracy - fp32_accuracy

    print("\n" + "=" * 65)
    print("QUANTIZATION RESULTS")
    print("=" * 65)
    print(f"FP32 accuracy : {fp32_accuracy:.2f}%")
    print(f"INT8 accuracy : {int8_accuracy:.2f}%")
    print(f"Accuracy delta: {difference:+.2f} percentage points")

    # Save the converted graph model and state dictionary.
    model_path = OUTPUT_DIR / "alexnet_int8_model.pth"
    state_path = OUTPUT_DIR / "alexnet_int8_state.pth"

    torch.save(int8_model, model_path)
    torch.save(int8_model.state_dict(), state_path)

    # Export quantized layer weights and metadata.
    export_weights(int8_model)

    # Keep a record of the experiment.
    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "quantization": "post-training static quantization",
        "backend": backend,
        "calibration_images": min(
            NUM_CALIBRATION_IMAGES, 50000
        ),
        "fp32_test_accuracy_percent": fp32_accuracy,
        "int8_test_accuracy_percent": int8_accuracy,
        "accuracy_delta_percentage_points": difference,
    }

    report_path = OUTPUT_DIR / "quantization_report.json"

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\nFiles saved in:", OUTPUT_DIR)
    print("Quantization completed successfully.")


if __name__ == "__main__":
    main()
