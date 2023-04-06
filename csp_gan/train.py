import os
import wandb
import torch
import numpy as np
import torch.nn as nn
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torchvision.utils import save_image
from torchvision.utils import make_grid
from PIL import Image
import torchvision.utils as vutils

from config import config as cfg
from model import Generator, Discriminator
from losses import R1, R2

def train(
        dataloader, 
        discriminator, 
        generator, 
        optimizerD, 
        optimizerG,
        criterion, 
        r1_reg, 
        device,
        wandb_run=False
):

    if wandb_run:
        wandb.init(
            project="CSPGAN",
            config=cfg
        )
    G_losses = []
    D_losses = []
    fixed_noise = torch.randn(cfg['batch_size'], cfg['latent_size'], device=device)

    for epoch in range(cfg['epochs']):
        for i, data in enumerate(dataloader, 0):

            discriminator.zero_grad()

            real = data[0].to(device)
            b_size = real.size(0)
            real.requires_grad = True

            output = discriminator(real)

            label = torch.full((b_size,), 1., dtype=torch.float, device=device)
            errD_real = criterion(output.view(-1), label) + r1_reg(output, real, cfg['r1_coef'])
            D_x = torch.sigmoid(output).view(-1).mean().item()
            errD_real.backward()

            noise = torch.randn(b_size, cfg['latent_size'], device=device)
            fake = generator(noise)
            label.fill_(0.)
            output = discriminator(fake.detach())
            errD_fake = criterion(output.view(-1), label)
            D_G_z1 = torch.sigmoid(output).view(-1).mean().item()
            errD = errD_real.detach() + errD_fake
            errD.backward()

            optimizerD.step()

            generator.zero_grad()
            label.fill_(1.)
            output = discriminator(fake)
            errG = criterion(output.view(-1), label)
            errG.backward()
            D_G_z2 = torch.sigmoid(output).view(-1).mean().item()
            optimizerG.step()

            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, cfg['epochs'], i, len(dataloader),
                        errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))    
            if wandb_run:
                wandb.log({"generator_loss": errG.item(), "discriminator_loss": errD.item()})

            G_losses.append(errG.item())
            D_losses.append(errD.item())
        
        torch.save({
            'discriminator_state_dict': discriminator.state_dict(),
            'generator_state_dict': generator.state_dict(),
            'optimizerD_state_dict': optimizerD.state_dict(),
            'optimizerG_state_dict': optimizerG.state_dict()},
            'csp_gan.pth.tar')

        with torch.no_grad():
            fake = generator(fixed_noise).detach().cpu()
        if wandb_run:
            wandb.log({"examples": [wandb.Image(im) for im in fake[:10]]})

    if wandb_run:
        wandb.finish()
    return {'G_losses': G_losses, 'D_losses': D_losses}

def main():

    dataset = ImageFolder(
        root='celeba',
        transform=tt.Compose([
            tt.Resize(cfg['image_size']),
            tt.CenterCrop(cfg['image_size']),
            tt.ToTensor(),
            tt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )

    dataloader = DataLoader(dataset, batch_size=cfg['batch_size'], shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    r1_reg = R1().to(device)
    generator = Generator(cfg['latent_size']).to(device)
    discriminator = Discriminator().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], 0.999))
    optimizerG = torch.optim.Adam(generator.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], 0.999))

    losses = train(        
        dataloader, 
        discriminator, 
        generator, 
        optimizerD, 
        optimizerG,
        criterion, 
        r1_reg, 
        device,
        wandb_run=True
    )


if __name__ == "__main__":
    main()
