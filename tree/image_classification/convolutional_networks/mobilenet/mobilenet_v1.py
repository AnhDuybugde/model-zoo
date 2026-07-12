import sys
from pathlib import Path
from typing import Any

import torch 
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[4]
sys.path.append(str(ROOT))

from mobilenet import MobileNetBlock

class MobileNetV1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.first_conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.features = nn.Sequential(
            MobileNetBlock(32, 64, stride=1),

            MobileNetBlock(64, 128, stride=2),
            MobileNetBlock(128, 128, stride=1),

            MobileNetBlock(128, 256, stride=2),
            MobileNetBlock(256, 256, stride=1),

            MobileNetBlock(256, 512, stride=2),

            MobileNetBlock(512, 512, stride=1),
            MobileNetBlock(512, 512, stride=1),
            MobileNetBlock(512, 512, stride=1),
            MobileNetBlock(512, 512, stride=1),
            MobileNetBlock(512, 512, stride=1),

            MobileNetBlock(512, 1024, stride=2),
            MobileNetBlock(1024, 1024, stride=1),
        )

        """"
        Ý nghĩa của Global Average Pool là:
        - Không quan tâm vật nằm ở đâu trong ảnh.
        - Chỉ quan tâm feature đó có xuất hiện mạnh hay không.
        - Giảm cực nhiều tham số so với Flatten toàn bộ feature map
        """
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Linear(1024, num_classes)
    
    def forward (self, x):
        x = self.first_conv(x)
        x = self.features(x)

        x = self.pool(x)  # B, 1024, H, W -> B, 1024, 1, 1
        x = torch.flatten(x, 1)  # B, 1024, 1, 1 -> B, 1024

        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)

    model = MobileNetV1(num_classes=1000)

    y = model(x)

    print("Input shape :", x.shape)
    print("Output shape:", y.shape)