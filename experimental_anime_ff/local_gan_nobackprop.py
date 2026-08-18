# =====================================================================
# PURE PYTORCH ISOLATED BLOCK-WISE GENERATOR (NO FULL BACKPROP)
# =====================================================================
import os
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
from safetensors.torch import save_file, load_file

# =====================================================================
# ISOLATED LOCAL GENERATOR BLOCK
# =====================================================================
class IsolatedGeneratorBlock(nn.Module):
    def __init__(self, in_channels, out_resolution, is_first_block=False, device='cuda'):
        super().__init__()
        self.out_resolution = out_resolution
        self.is_first_block = is_first_block

        # Local Generator
        if is_first_block:
            self.G = nn.Sequential(
                nn.ConvTranspose2d(in_channels, 128, kernel_size=4, stride=1, padding=0), # 1x1 -> 4x4
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 4x4 -> 8x8
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1), # 8x8 -> 16x16
                nn.Tanh()
            ).to(device)
        else:
            self.G = nn.Sequential(
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1), # Upscale 2x
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
            ).to(device)

        # Local Spatial Discriminator
        self.D = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        ).to(device)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.criterion = nn.BCEWithLogitsLoss()

    def train_block(self, x_in, x_real_target):
        # 1. Train Local Critic
        self.opt_D.zero_grad()

        d_real = self.D(x_real_target)
        loss_d_real = self.criterion(d_real, torch.ones_like(d_real))

        # Explicit detach prevents gradients from leaking to earlier layers
        x_fake = self.G(x_in.detach())
        d_fake = self.D(x_fake.detach())
        loss_d_fake = self.criterion(d_fake, torch.zeros_like(d_fake))

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.opt_D.step()

        # 2. Train Local Generator
        self.opt_G.zero_grad()
        d_fake_for_g = self.D(x_fake)
        loss_g = self.criterion(d_fake_for_g, torch.ones_like(d_fake_for_g))
        loss_g.backward()
        self.opt_G.step()

        return loss_g.item(), loss_d.item(), x_fake

# =====================================================================
# FULL NETWORK TOPOLOGY
# =====================================================================
class NoBackpropGenerator(nn.Module):
    def __init__(self, latent_dim=128, device='cuda'):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device
        self.block1 = IsolatedGeneratorBlock(in_channels=latent_dim, out_resolution=16, is_first_block=True, device=device)
        self.block2 = IsolatedGeneratorBlock(in_channels=3, out_resolution=32, is_first_block=False, device=device)
        self.block3 = IsolatedGeneratorBlock(in_channels=3, out_resolution=64, is_first_block=False, device=device)

    def train_step(self, x_real_64):
        # Dynamically scale targets for localized training
        x_real_16 = F.interpolate(x_real_64, size=(16, 16), mode='bilinear', align_corners=False)
        x_real_32 = F.interpolate(x_real_64, size=(32, 32), mode='bilinear', align_corners=False)

        # Base mathematical prior
        z = torch.randn(x_real_64.size(0), self.latent_dim, 1, 1, dtype=torch.float32, device=self.device)

        # Sequentially train detached blocks
        g_loss1, d_loss1, x_fake_16 = self.block1.train_block(z, x_real_16)
        g_loss2, d_loss2, x_fake_32 = self.block2.train_block(x_fake_16, x_real_32)
        g_loss3, d_loss3, x_fake_64 = self.block3.train_block(x_fake_32, x_real_64)

        total_g = g_loss1 + g_loss2 + g_loss3
        total_d = d_loss1 + d_loss2 + d_loss3

        return total_g, total_d, x_fake_64

    def generate(self, n_samples):
        with torch.no_grad():
            z = torch.randn(n_samples, self.latent_dim, 1, 1, dtype=torch.float32, device=self.device)
            x1 = self.block1.G(z)
            x2 = self.block2.G(x1)
            x3 = self.block3.G(x2)
            return x3
