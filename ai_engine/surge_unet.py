# surge_unet.py


import torch.nn as nn

import logging, torch


class DoubleConv(nn.Module):

    """Std dual convolutional block used in U-Net archs. Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):

        return self.conv(x)


class SurgeUNet(nn.Module):

    """Physics-surrogate CNN. Maps atmospheric forces & terrain into a 2D hydrodynamic surge prediction."""

    def __init__(self, in_channels = 3, out_channels = 1):
        super().__init__()

        # Encoder (Downsampling: Extracting macro-physics context)
        self.down1 = DoubleConv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2) # 256 -> 128
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2) # 128 -> 64
        self.down3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2) # 64 -> 32

        # Bottleneck (Deepest learned rep of storm)
        self.bottleneck = DoubleConv(256, 512)
        
        # Decoder (Upsampling: Reconstructing the high-res flood map)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size = 2, stride = 2) # 32 -> 64
        self.up_conv1 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size = 2, stride = 2) # 64 -> 128
        self.up_conv2 = DoubleConv(256, 128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size = 2, stride = 2) # 128 -> 256
        self.up_conv3 = DoubleConv(128, 64)

        # Final output layer (Maps 64 features down to 1 surge depth channel)
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size = 1)

    def forward(self, x):
        # Encoder pass
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        d3 = self.down3(p2)
        p3 = self.pool3(d3)
        # Bottleneck
        bn = self.bottleneck(p3)
        # Decoder pass with skip conns. Skip conns (d3, d2, d1) feed high-res boundary data (like coastlines) directly into the upsampler so the edges remain sharp.
        u1 = self.up1(bn)
        u1 = torch.cat([u1, d3], dim = 1)
        u1 = self.up_conv1(u1)
        u2 = self.up2(u1)
        u2 = torch.cat([u2, d2], dim = 1)
        u2 = self.up_conv2(u2)
        u3 = self.up3(u2)
        u3 = torch.cat([u3, d1], dim = 1)
        u3 = self.up_conv3(u3)
        out = torch.sigmoid(self.final_conv(u3)) # Sigmoid activation compresses output b/w 0 & 1 (matching the min-max scaling)

        return out


logging.basicConfig(level = logging.INFO, format = "%(asctime)s - [%(levelname)s] - %(message)s")


logging.info("Init-ing SurgeUNet Surrogate Model...")

model = SurgeUNet(in_channels = 3, out_channels = 1)

dummy_input = torch.randn(1, 3, 256, 256) # Dummy batch mimicking dataset output: [batch_size, channels, height, width]

logging.info(f"Passing input tensor of shape: {dummy_input.shape}")

# Forward pass
prediction = model(dummy_input)

logging.info(f"Model output tensor shape: {prediction.shape}")