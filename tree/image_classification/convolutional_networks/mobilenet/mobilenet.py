import sys
from pathlib import Path

import torch 
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT))

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=in_channels,
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)

    block = MobileNetBlock(
        in_channels=3,
        out_channels=16,
        stride=2,
    )

    y = block(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)