import argparse
import os
from utils import get_args

import torch

from networks import Discriminator, Generator
from train import train_model


def compute_discriminator_loss(
    discrim_real, discrim_fake, discrim_interp, interp, lamb
):
    """
    TODO 1.3.1: Implement GAN loss for discriminator.
    Do not use discrim_interp, gitinterp, lamb. They are placeholders for Q1.5.
    """
    
    # loss = -torch.log(discrim_real) - torch.log(1 - discrim_fake)
    criterion = torch.nn.BCEWithLogitsLoss()
    real_loss = criterion(discrim_real, torch.ones_like(discrim_real)) 
    fake_loss = criterion(discrim_fake, torch.zeros_like(discrim_fake)) 
    loss = real_loss + fake_loss
    # real_loss = nn.BCEWithLogitsLoss(discrim_real, torch.ones_like(discrim_real))
    # fake_loss = nn.BCEWithLogitsLoss(discrim_fake, torch.zeros_like(discrim_fake))
    # return real_loss + fake_loss
    # print(loss)
    return loss
    

def compute_generator_loss(discrim_fake):
    """
    TODO 1.3.1: Implement GAN loss for generator.
    """
    criterion = torch.nn.BCEWithLogitsLoss()
    loss = criterion(discrim_fake, torch.ones_like(discrim_fake))
    # return F.binary_cross_entropy_with_logits(discrim_fake, torch.ones_like(discrim_fake))
    # loss = -torch.log(discrim_fake)
    return loss


if __name__ == "__main__":
    args = get_args()
    gen = Generator().cuda()
    disc = Discriminator().cuda()
    prefix = "data_gan/"
    os.makedirs(prefix, exist_ok=True)

    # TODO 1.3.2: Run this line of code.
    train_model(
        gen,
        disc,
        num_iterations=int(3e4),
        batch_size=256,
        prefix=prefix,
        gen_loss_fn=compute_generator_loss,
        disc_loss_fn=compute_discriminator_loss,
        log_period=1000,
        amp_enabled=not args.disable_amp,
    )
