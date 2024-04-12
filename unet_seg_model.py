import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor
from monai.data import Dataset
from monai.transforms import LoadImage, ScaleIntensityRange, RandCropByPosNegLabel, Rotate
#from monai.networks.nets import UNet
from monai.losses import DiceLoss
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, channels, strides):
        super(UNet, self).__init__()
        self.model = UNetModel(in_channels, out_channels, channels, strides)

    def forward(self, x):
        return self.model(x)

class UNetModel(nn.Module):
    def __init__(self, in_channels, out_channels, channels, strides):
        super(UNetModel, self).__init__()
        self.encoder = Encoder(in_channels, channels, strides)
        self.decoder = Decoder(out_channels, channels[::-1])

    def forward(self, x):
        # Forward pass through the encoder
        x, skip_connections = self.encoder(x)
        print("Size of x after encoder:", x.size())
        # Forward pass through the decoder
        x = self.decoder(x, skip_connections)
        return x

class Encoder(nn.Module):
    def __init__(self, in_channels, channels, strides):
        super(Encoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(channels)):
            self.blocks.append(EncoderBlock(in_channels, channels[i], strides[i]))
            in_channels = channels[i]

    def forward(self, x):
        skip_connections = []
        for block in self.blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        return x, skip_connections

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(EncoderBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.conv1(x))
        out = self.relu(self.conv2(out))
        return out, out

class Decoder(nn.Module):
    def __init__(self, out_channels, channels):
        super(Decoder, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(len(channels) - 1):
            self.blocks.append(DecoderBlock(channels[i], channels[i+1]))
        self.final_conv = nn.Conv3d(channels[-1], out_channels, kernel_size=3)

    def forward(self, x, skip_connections):
        for i, block in enumerate(self.blocks):
            x = block(x, skip_connections[-(i+2)])
        x = self.final_conv(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=3, stride=2)
        self.conv2 = nn.Conv3d(out_channels + out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, skip):
        print("Shape of x before upsampling:", x.shape)
        x = self.relu(self.conv1(x))
        print("Shape of x after upsampling:", x.shape)
        print("Shape of skip connection:", skip.shape)
        x = torch.cat([x, skip], dim=1)
        print("Shape of x after concatenation:", x.shape)
        x = self.relu(self.conv2(x))
        print("Shape of x after convolution:", x.shape)
        return x
