import torch 
import torch.nn as nn

# moi kernel chi di qua 1 channel, nen output channel = imput channel = so luong kernel
class DepthwiseConv2DBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride, padding):
        super(DepthwiseConv2DBlock, self).__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size 
        self.stride = stride
        self.padding = padding

        # khac voi conv2d co shape = (out_channels, in_channels, kernel_size, kernel_size), 
        # weight cua depthwise conv co shape = (in_channels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(torch.randn(in_channels, 1, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.randn(in_channels))

    def forward(self, x):
        if self.padding > 0:
            x = nn.functional.pad(x, (self.padding, self.padding, self.padding, self.padding))

        # Input
        batch_size, in_channels, height, width = x.shape

        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        # Output tensor
        output = torch.zeros((batch_size, in_channels, out_height, out_width), device=x.device)

        # Perform depthwise convolution operation
        for b in range(batch_size):
            for ic in range(in_channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        w_start = j * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size

                        output[b, ic, i, j] = torch.sum(x[b, ic, h_start:h_end, w_start:w_end] * self.weight[ic])
                output[b, ic] += self.bias[ic]
        
        return output
    

if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)

    model = DepthwiseConv2DBlock(in_channels=3, kernel_size=3, stride=2, padding=1)

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