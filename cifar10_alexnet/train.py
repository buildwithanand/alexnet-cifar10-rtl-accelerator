
import csv
import os
import time
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from model import AlexNetInspired


# ---------------- CONFIGURATION ----------------

DATA_DIR = "./data"
CHECKPOINT_DIR = "./checkpoints"
RESULTS_DIR = "./results"

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0005
WEIGHT_DECAY = 5e-4

VALIDATION_SIZE = 5000
PATIENCE = 8
MIN_DELTA = 0.05  # Minimum validation accuracy gain, in percentage points.
SEED = 42

# Windows: start with 2 workers. Increase to 4 if it helps.
NUM_WORKERS = 2

USE_AMP = True  # Automatic mixed precision on CUDA.

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------- REPRODUCIBILITY ----------------

def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)


seed_everything(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
use_amp = USE_AMP and device.type == "cuda"

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

print("=" * 65)
print("ALEXNET-INSPIRED CIFAR-10 TRAINING")
print("=" * 65)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA runtime:", torch.version.cuda)
    print("Mixed precision:", use_amp)
    print("Batch size:", BATCH_SIZE)


# ---------------- TRANSFORMS ----------------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

# Training only: moderate augmentation.
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(p=0.5),

    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.15,
        hue=0.02
    ),

    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),

    transforms.RandomErasing(
        p=0.15,
        scale=(0.02, 0.12),
        ratio=(0.3, 3.3),
        value=0
    )
])

# Validation and test remain deterministic.
eval_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
])


# ---------------- DATASETS ----------------

def main():
    dataset_folder = os.path.join(
        DATA_DIR, "cifar-10-batches-py"
    )

    if not os.path.isdir(dataset_folder):
        raise FileNotFoundError(
            "CIFAR-10 is missing from ./data. "
            "Extract cifar-10-python.tar.gz first."
        )

    augmented_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=train_transform
    )

    evaluation_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=eval_transform
    )

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=eval_transform
    )

    # Keep the same reproducible 45,000 / 5,000 split.
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(
        len(augmented_dataset),
        generator=generator
    ).tolist()

    val_indices = indices[:VALIDATION_SIZE]
    train_indices = indices[VALIDATION_SIZE:]

    train_dataset = Subset(
        augmented_dataset, train_indices
    )
    val_dataset = Subset(
        evaluation_dataset, val_indices
    )

    loader_generator = torch.Generator().manual_seed(SEED)

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "pin_memory": device.type == "cuda",
        "worker_init_fn": seed_worker
    }

    if NUM_WORKERS > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=loader_generator,
        drop_last=False,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        **loader_kwargs
    )

    classes = augmented_dataset.classes

    print("Training images:", len(train_dataset))
    print("Validation images:", len(val_dataset))
    print("Test images:", len(test_dataset))
    print(
        "Augmentation: crop + flip + color jitter "
        "+ random erasing"
    )

    # ---------------- MODEL ----------------

    model = AlexNetInspired(num_classes=10).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=LEARNING_RATE * 0.02
    )

    # Compatible with current PyTorch AMP APIs.
    scaler = torch.amp.GradScaler(
        "cuda", enabled=use_amp
    )

    best_val_accuracy = -1.0
    best_epoch = 0
    epochs_without_improvement = 0

    best_path = os.path.join(
        CHECKPOINT_DIR, "best_model.pth"
    )
    final_path = os.path.join(
        CHECKPOINT_DIR, "final_weights.pth"
    )
    history_path = os.path.join(
        RESULTS_DIR, "history.csv"
    )

    # ---------------- EVALUATION ----------------

    def evaluate(loader):
        model.eval()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.inference_mode():
            for images, labels in loader:
                images = images.to(
                    device, non_blocking=True
                )
                labels = labels.to(
                    device, non_blocking=True
                )

                with torch.amp.autocast(
                    device_type=device.type,
                    enabled=use_amp
                ):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                total_loss += (
                    loss.item() * labels.size(0)
                )
                correct += (
                    outputs.argmax(dim=1) == labels
                ).sum().item()
                total += labels.size(0)

        return total_loss / total, 100.0 * correct / total

    # ---------------- TRAINING ----------------

    with open(history_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_accuracy",
            "val_loss",
            "val_accuracy",
            "seconds"
        ])

    for epoch in range(EPOCHS):
        start_time = time.time()

        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(
                device, non_blocking=True
            )
            labels = labels.to(
                device, non_blocking=True
            )

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(
                device_type=device.type,
                enabled=use_amp
            ):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * labels.size(0)
            correct += (
                outputs.detach().argmax(dim=1) == labels
            ).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_accuracy = 100.0 * correct / total

        val_loss, val_accuracy = evaluate(val_loader)

        # Advance the learning-rate schedule once per epoch.
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy:.2f}% | "
            f"LR: {current_lr:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        with open(history_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1,
                f"{train_loss:.6f}",
                f"{train_accuracy:.3f}",
                f"{val_loss:.6f}",
                f"{val_accuracy:.3f}",
                f"{elapsed:.2f}"
            ])

        # Save the best validation checkpoint.
        if val_accuracy > best_val_accuracy + MIN_DELTA:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            epochs_without_improvement = 0

            torch.save({
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_accuracy": best_val_accuracy,
                "classes": classes
            }, best_path)

            print("  Best validation checkpoint saved.")
        else:
            epochs_without_improvement += 1
            print(
                "  No significant validation improvement: "
                f"{epochs_without_improvement}/{PATIENCE}"
            )

        if epochs_without_improvement >= PATIENCE:
            print(
                f"\nEarly stopping after "
                f"{PATIENCE} epochs without a significant "
                "validation accuracy improvement."
            )
            break

    # ---------------- FINAL TEST EVALUATION ----------------

    checkpoint = torch.load(
        best_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    torch.save(model.state_dict(), final_path)

    test_loss, test_accuracy = evaluate(test_loader)

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE")
    print("=" * 65)
    print("Best epoch:", best_epoch)
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Final test accuracy: {test_accuracy:.2f}%")
    print(f"Final test loss: {test_loss:.4f}")
    print("Best checkpoint:", best_path)
    print("Inference weights:", final_path)
    print("Training history:", history_path)


if __name__ == "__main__":
    main()
