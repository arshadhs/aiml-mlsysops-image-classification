# Define the CNN architecture

import torch
import torch.nn as nn

class CIFAR10CNN(nn.Module):
    """
    CNN for CIFAR-10 image classification

    Input: [batch_size, 3, 32, 32]
    Output: [batch_size, 10]
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()

        # Convolutional feature extraction
        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels = 3,
                out_channels = 32,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels = 32,
                out_channels = 64,
                kernel_size = 3,
                padding = 1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        # Classification Layer
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x