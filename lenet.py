"""
Standard LeNet-5 for MNIST (1x28x28 -> 10 classes).

Architecture:
  Conv(1->6, 5x5)  -> tanh -> AvgPool(2x2)   # 6x12x12
  Conv(6->16, 5x5) -> tanh -> AvgPool(2x2)   # 16x4x4
  Flatten -> 256
  FC(256->120)     -> tanh   <-- watermark neurons monitored here
  FC(120->84)      -> tanh
  FC(84->10)
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)    # -> 6x24x24
        self.pool1 = nn.AvgPool2d(2, 2)                # -> 6x12x12
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)   # -> 16x8x8
        self.pool2 = nn.AvgPool2d(2, 2)                # -> 16x4x4
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def conv_features(self, x):
        """Flattened output of the conv stack — input to fc1. Shape: [B, 256]."""
        x = torch.tanh(self.conv1(x))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        return x.view(x.size(0), -1)

    def fc1_preact(self, x):
        """FC1 pre-activation (before tanh). Shape: [B, 120].
        This is what the hardware comparator monitors."""
        return self.fc1(self.conv_features(x))
