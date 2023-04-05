import os
import cv2
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm
import torchvision.transforms as tt

class CSPBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1),
            nn.Upsample(scale_factor=2)
        )

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
        )
    def forward(self, x):
        return self.conv2(x) + self.block(self.conv1(x))

class Generator(nn.Module):
    def __init__(self, lat_dim):
        super(Generator, self).__init__()

        channels = [512, 256, 128, 64]

        self.linear = nn.Linear(lat_dim, 1024*4*4)
        self.csp_up = nn.Sequential(*[CSPBlock(ch) for ch in channels])

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x1 = self.linear(x).view(-1, 1024, 4, 4)
        x2 = self.csp_up(x1)
        return self.deconv(x2)


class Discriminator(nn.Module):
    def __init__(self, ):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
        )

    def forward(self, input):
        return self.main(input)
