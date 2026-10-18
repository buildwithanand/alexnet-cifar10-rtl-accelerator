
import os
import torch

WEIGHTS_PATH = "./checkpoints/final_weights.pth"

if not os.path.exists(WEIGHTS_PATH):
    raise FileNotFoundError(f"Weights not found: {WEIGHTS_PATH}")

# Your final_weights.pth contains a state_dict.
weights = torch.load(
    WEIGHTS_PATH,
    map_location="cpu",
    weights_only=True
)

print("=" * 55)
print("MODEL WEIGHT PRECISION REPORT")
print("=" * 55)

total_params = 0
total_bytes = 0
dtype_counts = {}

for name, tensor in weights.items():
    if not isinstance(tensor, torch.Tensor):
        continue

    count = tensor.numel()
    size_bytes = tensor.numel() * tensor.element_size()
    dtype = str(tensor.dtype)

    total_params += count
    total_bytes += size_bytes
    dtype_counts[dtype] = dtype_counts.get(dtype, 0) + count

    print(
        f"{name:<40} "
        f"{dtype:<15} "
        f"{count:>10,} parameters"
    )

print("-" * 55)
print("Total parameters:", f"{total_params:,}")
print("Total tensor storage:", f"{total_bytes / 1024:.2f} KB")
print("\nParameter counts by dtype:")

for dtype, count in dtype_counts.items():
    print(f"{dtype}: {count:,}")

print("\nConclusion:")

if dtype_counts and all(
    dtype == "torch.float32" for dtype in dtype_counts
):
    print("All saved parameters are FP32.")
elif dtype_counts and all(
    dtype == "torch.float16" for dtype in dtype_counts
):
    print("All saved parameters are FP16.")
elif dtype_counts and all(
    dtype == "torch.int8" for dtype in dtype_counts
):
    print("All saved parameters are INT8.")
else:
    print("The checkpoint contains mixed or other data types.")
