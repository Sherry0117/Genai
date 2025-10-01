import torch
import torch.nn as nn


class AssignmentCNN(nn.Module):
    """
    Convolutional Neural Network matching the exact specification:
    1) Conv2D(3->16, kernel=3, stride=1, padding=1)
    2) ReLU
    3) MaxPool2D(kernel=2, stride=2)
    4) Conv2D(16->32, kernel=3, stride=1, padding=1)
    5) ReLU
    6) MaxPool2D(kernel=2, stride=2)
    7) Flatten
    8) Linear(32*16*16 -> 100)
    9) ReLU
    10) Linear(100 -> 10)

    Assumes input: RGB image tensor of shape (N, 3, 64, 64).
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU(inplace=False)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU(inplace=False)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # After two 2x2 pools, spatial dims: 64 -> 32 -> 16
        flattened_dim = 32 * 16 * 16
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(flattened_dim, 100)
        self.relu3 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


if __name__ == "__main__":
    # Quick test: random batch of 4 RGB 64x64 images
    model = AssignmentCNN(num_classes=10)
    dummy = torch.randn(4, 3, 64, 64)
    out = model(dummy)
    print("Output shape:", out.shape)


