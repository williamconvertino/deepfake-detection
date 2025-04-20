import torch
import torch.nn as nn
import torch.nn.functional as F

"""
This is an implementation of the xception CNN model, which can be found here: https://arxiv.org/pdf/1610.02357

This is for the ECE661 final project. 
Jake Wolfram 4/20/25
"""



# Depthwise Separable Convolution
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                                   stride=stride, padding=kernel_size//2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return self.bn(x)

# Block with residual connection
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, reps, stride=1, grow_first=True):
        super().__init__()
        self.skip = None
        if out_channels != in_channels or stride != 1:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_channels)

        layers = []
        filters = in_channels
        if grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_channels, out_channels, 3))
            filters = out_channels

        for _ in range(reps - 1):
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(filters, filters, 3))

        if not grow_first:
            layers.append(nn.ReLU(inplace=True))
            layers.append(SeparableConv2d(in_channels, out_channels, 3))

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride=stride, padding=1))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.skip is not None:
            x = self.skip_bn(self.skip(x))
        return out + x if self.skip is not None else out

# Xception Model
class Xception(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            XceptionBlock(64, 128, reps=2, stride=2, grow_first=True),
            XceptionBlock(128, 256, reps=2, stride=2, grow_first=True),
            XceptionBlock(256, 728, reps=2, stride=2, grow_first=True)
        )

        self.middle = nn.Sequential(*[
            XceptionBlock(728, 728, reps=3, stride=1, grow_first=True) for _ in range(8)
        ])

        self.exit = nn.Sequential(
            XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False),
            SeparableConv2d(1024, 1536, 3),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, 3),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        x = self.exit(x)
        x = torch.flatten(x, 1)
        return self.fc(x)