import argparse

import numpy as np
import torch
from cleanfid import fid
from matplotlib import pyplot as plt
import torchvision


def save_plot(x, y, xlabel, ylabel, title, filename):
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.savefig(filename + ".png")


@torch.no_grad()
def get_fid(gen, dataset_name, dataset_resolution, z_dimension, batch_size, num_gen):
    gen_fn = lambda z: (gen.forward_given_samples(z) / 2 + 0.5) * 255
    score = fid.compute_fid(
        gen=gen_fn,
        dataset_name=dataset_name,
        dataset_res=dataset_resolution,
        num_gen=num_gen,
        z_dim=z_dimension,
        batch_size=batch_size,
        verbose=True,
        dataset_split="custom",
    )
    return score


@torch.no_grad()
def interpolate_latent_space(gen, path):
    # TODO 1.2: Generate and save out latent space interpolations.
    # Generate 100 samples of 128-dim vectors
    # Do so by linearly interpolating for 10 steps across each of the first two dimensions between -1 and 1.
    # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # Forward the samples through the generator.
    # Save out an image holding all 100 samples.
    # Use torchvision.utils.save_image to save out the visualization.

    # Adeesh Starts
    # z = torch.zeros((100, 128))
    # z[:, :2] = torch.linspace(-1, 1, 100)[:, None]
    # # Keep the rest of the z vector for the samples to be some fixed value (e.g. 0).
    # # Forward the samples through the generator.
    # samples = gen.forward_given_samples(z.cuda())
    # #Save out an image holding all 100 samples.
    # #torchvision.utils.save_image(out, "interpolate_latent_space.png")
    # torchvision.utils.save_image(samples, path, nrow=10)
    # Adeesh Ends

    min_val = -1
    max_val = 1
    num_steps = 10
    samples = torch.zeros(100, 128).cuda()
    dim_1 = np.linspace(min_val, max_val, num_steps)
    dim_2 = np.linspace(min_val, max_val, num_steps)
    xx, yy = np.meshgrid(dim_1, dim_2, indexing='ij')
    interp_points = np.column_stack([xx.ravel(), yy.ravel()])
    samples[:, :2] = torch.from_numpy(interp_points).cuda()

    generated_samples = gen.forward_given_samples(samples)
    # generated_samples = (generated_samples / 2 + 0.5) * 255
    torchvision.utils.save_image(generated_samples, path, nrow=10)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--disable_amp", action="store_true")
    args = parser.parse_args()
    return args
