import argparse
import csv
import time
from pathlib import Path

import torch
import matplotlib
matplotlib.use("Agg")  # Save plots without requiring a GUI
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

from model import AlexNetInspired


DATA_DIR = "./data"
CHECKPOINT = "./checkpoints/best_model.pth"
RESULTS_DIR = Path("./results/prediction_analysis")

MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)


def main():
    parser = argparse.ArgumentParser(
        description="Run CIFAR-10 predictions over a selected index range and plot results."
    )
    parser.add_argument("--start", type=int, default=0,
                        help="First test-dataset index (inclusive). Default: 0")
    parser.add_argument("--count", type=int, default=5000,
                        help="Number of images to evaluate. Default: 5000")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Inference batch size. Default: 128")
    args = parser.parse_args()

    if args.start < 0 or args.count <= 0 or args.batch_size <= 0:
        parser.error("--start must be >= 0; --count and --batch-size must be > 0")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    test_dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,  # Keep the run offline
        transform=transform,
    )

    if args.start >= len(test_dataset):
        parser.error(f"--start must be less than {len(test_dataset)}")

    end = min(args.start + args.count, len(test_dataset))
    actual_count = end - args.start

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=True,
    )

    model = AlexNetInspired(num_classes=10).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    classes = checkpoint.get("classes", test_dataset.classes)

    # A subset lets us evaluate exactly the selected index range.
    subset = torch.utils.data.Subset(
        test_dataset, list(range(args.start, end))
    )
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Reliable on Windows
        pin_memory=(device.type == "cuda"),
    )

    print("=" * 78)
    print("CIFAR-10 PREDICTION ANALYSIS")
    print("=" * 78)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Image indices: {args.start} through {end - 1}")
    print(f"Images to evaluate: {actual_count}")
    print(f"Batch size: {args.batch_size}")
    print("Running inference...")

    rows = []
    total_correct = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)
    running_correct = []
    processed = 0

    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time.perf_counter()

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predictions = probabilities.max(dim=1)

            predictions_cpu = predictions.cpu().tolist()
            labels_cpu = labels.cpu().tolist()
            confidences_cpu = confidences.cpu().tolist()

            for offset, (pred, true, confidence) in enumerate(
                zip(predictions_cpu, labels_cpu, confidences_cpu)
            ):
                image_index = args.start + processed
                is_correct = pred == true
                total_correct += int(is_correct)
                class_total[true] += 1
                class_correct[true] += int(is_correct)
                running_correct.append(total_correct / (processed + 1) * 100)

                rows.append({
                    "index": image_index,
                    "actual_class": classes[true],
                    "predicted_class": classes[pred],
                    "confidence_percent": round(confidence * 100, 4),
                    "correct": is_correct,
                })
                processed += 1

            if processed % 512 == 0 or processed == actual_count:
                print(f"Processed {processed}/{actual_count} images...")

    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start_time

    csv_path = results_dir / f"predictions_{args.start}_{end - 1}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "index", "actual_class", "predicted_class",
                "confidence_percent", "correct"
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Graph 1: running accuracy across the selected range.
    indices = [row["index"] for row in rows]
    plt.figure(figsize=(12, 5))
    plt.plot(indices, running_correct, linewidth=1.5)
    plt.xlabel("CIFAR-10 test image index")
    plt.ylabel("Running accuracy (%)")
    plt.title(f"Running Accuracy (indices {args.start}–{end - 1})")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    accuracy_path = results_dir / f"running_accuracy_{args.start}_{end - 1}.png"
    plt.savefig(accuracy_path, dpi=160)
    plt.close()

    # Graph 2: correct vs incorrect.
    incorrect = actual_count - total_correct
    plt.figure(figsize=(7, 5))
    bars = plt.bar(
        ["Correct", "Incorrect"],
        [total_correct, incorrect],
    )
    plt.ylabel("Number of images")
    plt.title(f"Prediction Outcomes (accuracy: {100 * total_correct / actual_count:.2f}%)")
    plt.bar_label(bars)
    plt.tight_layout()
    outcome_path = results_dir / f"prediction_outcomes_{args.start}_{end - 1}.png"
    plt.savefig(outcome_path, dpi=160)
    plt.close()

    # Graph 3: accuracy for each class represented in the selected range.
    class_names = []
    class_accuracies = []
    for class_id, name in enumerate(classes):
        if class_total[class_id] > 0:
            class_names.append(name)
            class_accuracies.append(
                100 * class_correct[class_id] / class_total[class_id]
            )

    plt.figure(figsize=(11, 5))
    bars = plt.bar(class_names, class_accuracies)
    plt.ylabel("Accuracy (%)")
    plt.title("Per-Class Accuracy in Selected Image Range")
    plt.ylim(0, 100)
    plt.xticks(rotation=30, ha="right")
    plt.bar_label(bars, fmt="%.1f%%", padding=2)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    class_path = results_dir / f"class_accuracy_{args.start}_{end - 1}.png"
    plt.savefig(class_path, dpi=160)
    plt.close()

    accuracy = 100 * total_correct / actual_count
    print("=" * 78)
    print("PREDICTION SUMMARY")
    print("=" * 78)
    print(f"Correct: {total_correct}/{actual_count}")
    print(f"Incorrect: {incorrect}/{actual_count}")
    print(f"Accuracy on selected images: {accuracy:.2f}%")
    print(f"Total inference time: {elapsed:.2f} seconds")
    print(f"Average time per image: {elapsed / actual_count * 1000:.3f} ms")
    print("\nSaved files:")
    print(f"  CSV: {csv_path}")
    print(f"  Running accuracy graph: {accuracy_path}")
    print(f"  Correct/incorrect graph: {outcome_path}")
    print(f"  Per-class accuracy graph: {class_path}")
    print("=" * 78)


if __name__ == "__main__":
    main()
