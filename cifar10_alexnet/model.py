
import torch
import torch.nn as nn


class AlexNetInspired(nn.Module):
    """Small AlexNet-inspired CNN for CIFAR-10 (32x32 RGB images)."""

    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


if __name__ == "__main__":
    model = AlexNetInspired()
    x = torch.randn(1, 3, 32, 32)
    y = model(x)

    print("Input shape:", x.shape)
    print("Output shape:", y.shape)
    print("Parameters:", sum(p.numel() for p in model.parameters()))
