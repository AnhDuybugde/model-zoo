import torch
import torch.nn as nn

""""
Vì sao gọi là “inverted”?

Residual block thông thường của ResNet:
channel lớn
→ nén nhỏ
→ xử lý
→ mở lớn lại

MobileNetV2 làm ngược:
channel nhỏ
→ mở lớn
→ xử lý
→ nén nhỏ lại

=> nên gọi là inverted residual.
"""
class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()

        hidden_channels = in_channels * expand_ratio

        # Chỉ cộng residual khi shape input và output giống nhau
        self.use_residual = (stride == 1 and in_channels == out_channels)

        layers = []

        # Bước 1: Expansion bằng pointwise convolution
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_channels, kernel_size=1))

        # Bước 2: Depthwise convolution
        layers.append(ConvBNReLU(hidden_channels, hidden_channels, kernel_size=3, stride=stride, groups=hidden_channels))

        # Bước 3: Projection bằng pointwise convolution
        # Không dùng ReLU sau projection
        layers.append(nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False))

        layers.append(nn.BatchNorm2d(out_channels))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        output = self.block(x)

        if self.use_residual:
            output = output + x

        return output
