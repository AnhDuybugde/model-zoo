import sys
from pathlib import Path

import torch 
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT))

from tree.fundamentals.convolution.depthwise_conv import DepthwiseConv2DBlock
from tree.fundamentals.convolution.pointwise_conv import PointwiseConv2DBlock

class MobileNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(MobileNetBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride

        # Depthwise convolution
        self.depthwise_conv = DepthwiseConv2DBlock(in_channels=in_channels, kernel_size=3, stride=stride, padding=1)

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU()

        # Pointwise convolution
        self.pointwise_conv = PointwiseConv2DBlock(in_channels=in_channels, out_channels=out_channels)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise_conv(x)
        x = self.bn2(x)
        x = self.relu2(x)
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