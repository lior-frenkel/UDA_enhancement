"""UNet building blocks for speech enhancement.

Contains: DoubleConv, Down, Up, OutConv modules used in the encoder-decoder UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block: (Conv2d -> BN -> ReLU) × 2."""

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3, padding=1, use_prelu=False, dropout=False, dropout_fact=0.2):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        if use_prelu:
            activation = nn.PReLU
        else:
            activation = nn.ReLU

        layers = [
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            activation(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout_fact))

        self.double_conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv."""

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, use_prelu=False, dropout=False, dropout_fact=0.2):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, kernel_size=kernel_size, padding=padding,
                       use_prelu=use_prelu, dropout=dropout, dropout_fact=dropout_fact)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""

    def __init__(self, in_channels, out_channels, bilinear=True, kernel_size=3, padding=1, use_prelu=False, dropout=False, dropout_fact=0.2):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2,
                                   kernel_size=kernel_size, padding=padding,
                                   use_prelu=use_prelu, dropout=dropout, dropout_fact=dropout_fact)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels,
                                   kernel_size=kernel_size, padding=padding,
                                   use_prelu=use_prelu, dropout=dropout, dropout_fact=dropout_fact)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad to match skip-connection dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """1x1 convolution for output projection."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
