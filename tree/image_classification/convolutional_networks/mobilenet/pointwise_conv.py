import torch
import torch.nn as nn

# moi kernel se nhin toan bo channel cua input
# vi du input co 3 channel, output co 16 channel, kernel size = 1x1 => output channel = so luong kernel = 16
# nen weight cua pointwise conv co shape = (out_channels, in_channels, 1, 1) = (16, 3, 1, 1)

# Layer	Weight Shape
# Conv2D	[out_channels, in_channels, 3, 3]
# Depthwise	[in_channels, 1, 3, 3]
# Pointwise	[out_channels, in_channels, 1, 1]

class PointwiseConv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(PointwiseConv2DBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels))

    def forward(self, x):
        # Input
        batch_size, in_channels, height, width = x.shape

        # Output tensor
        output = torch.zeros((batch_size, self.out_channels, height, width), device=x.device)

        # Perform pointwise convolution operation
        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    output[b, oc] += x[b, ic] * self.weight[oc, ic]
                output[b, oc] += self.bias[oc]

        return output
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)

    model = PointwiseConv2DBlock(in_channels=3, out_channels=16)

    y = model(x)

    print('Input shape:', x.shape)
    print('Output shape:', y.shape) 

    print("\nWeight shape:", model.weight.shape)
    print("Bias shape  :", model.bias.shape)

    print("\nKernel của Output Channel 0:")
    print(model.weight[0])

    print("\nKernel R (Output 0 - Input 0):")
    print(model.weight[0, 0])

    print("\nBias của Output Channel 0:")
    print(model.bias[0])

    print("\nPixel đầu tiên của ảnh đầu tiên (3 channel):")
    print(x[0, :, 0, 0])

    print("\nOutput feature map đầu tiên:")
    print(y[0, 0])