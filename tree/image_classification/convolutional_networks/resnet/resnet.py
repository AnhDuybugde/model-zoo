import torch
import torch.nn as nn

from basic_block import BasicBlock
from bottleneck_block import Bottleneck


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super().__init__()

        self.in_channels = 64

        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self.make_layer(block, out_channels=64, blocks=layers[0], stride=1)
        self.layer2 = self.make_layer(block, out_channels=128, blocks=layers[1], stride=2)
        self.layer3 = self.make_layer(block, out_channels=256, blocks=layers[2], stride=2)
        self.layer4 = self.make_layer(block, out_channels=512, blocks=layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

    def make_layer(self, block, out_channels, blocks, stride):
        layers = []

        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels, stride=1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes)


def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)


def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes)


def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes)


if __name__ == "__main__":
    x = torch.randn(2, 3, 224, 224)

    models = {
        "ResNet18": resnet18(),
        "ResNet34": resnet34(),
        "ResNet50": resnet50()
    }

    for name, model in models.items():
        y = model(x)
        parameters = sum(parameter.numel() for parameter in model.parameters())

        print(name)
        print("Input :", x.shape)
        print("Output:", y.shape)
        print("Parameters:", parameters)
        print()

        print("\nParameter details:")
        for name, parameter in model.named_parameters():
            print(f"{name:40} {list(parameter.shape)} -> {parameter.numel():,}")