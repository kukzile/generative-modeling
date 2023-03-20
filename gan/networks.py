import torch
import torch.nn as nn


class UpSampleConv2D(torch.jit.ScriptModule):
    # TODO 1.1: Implement nearest neighbor upsampling + conv layer

    def __init__(
        self,
        input_channels,
        kernel_size=3,
        n_filters=128,
        upscale_factor=2,
        padding=0,
    ):
        super(UpSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.upscale_factor = upscale_factor
        self.padding = padding
        self.conv_layer = nn.Conv2d(self.input_channels, self.n_filters, self.kernel_size, padding=self.padding)
        self.pixel_shuffle = nn.PixelShuffle(self.upscale_factor)



    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement nearest neighbor upsampling
        # 1. Repeat x channel wise upscale_factor^2 times
        # 2. Use pixel shuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelShuffle.html#torch.nn.PixelShuffle)
        # to form a (batch x channel x height*upscale_factor x width*upscale_factor) output
        # 3. Apply convolution and return output
        
        # Upscale x
        x = x.repeat(1, int(self.upscale_factor ** 2), 1, 1)
        x = self.pixel_shuffle(x)
        return self.conv_layer(x)

class DownSampleConv2D(torch.jit.ScriptModule, nn.Module):
    # TODO 1.1: Implement spatial mean pooling + conv layer

    def __init__(
        self, input_channels, kernel_size=3, n_filters=128, downscale_ratio=2, padding=0
    ):
        super(DownSampleConv2D, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.downscale_ratio = downscale_ratio
        self.padding = padding
        self.conv_layer = nn.Conv2d(self.input_channels, self.n_filters, self.kernel_size, padding=self.padding)
        self.pixel_unshuffle = nn.PixelUnshuffle(self.downscale_ratio)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Implement spatial mean pooling
        # 1. Use pixel unshuffle (https://pytorch.org/docs/master/generated/torch.nn.PixelUnshuffle.html#tor`ch.nn.PixelUnshuffle)
        # to form a (batch x channel * downscale_factor^2 x height x width) output
        # 2. Then split channel wise into (downscale_factor^2xbatch x channel x height x width) images
        # 3. Average across dimension 0, apply convolution and return output
        
        unshuffle_x = self.pixel_unshuffle(x)
        split_tensors = torch.split(unshuffle_x, int(unshuffle_x.shape[1] / self.downscale_ratio**2), dim=1)
        output_tensor = torch.stack(split_tensors, dim=0)
        output_tensor = torch.mean(output_tensor, dim=0)
        conv_x = self.conv_layer(output_tensor)
        return conv_x


class ResBlockUp(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Upsampler.
    """
    ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(in_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(n_filters, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockUp, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.fwd = nn.Sequential(nn.BatchNorm2d(self.input_channels, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True), 
        nn.ReLU(), 
        nn.Conv2d(self.input_channels, self.n_filters, kernel_size = self.kernel_size, padding=(1,1), bias=False), 
        nn.BatchNorm2d(self.n_filters, eps = 1e-05, momentum = 0.1, affine = True, track_running_stats = True), 
        nn.ReLU(), 
        UpSampleConv2D(self.n_filters, self.kernel_size, self.n_filters, padding = 1))
        self.upsample = UpSampleConv2D(self.input_channels, 1, self.n_filters)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through the layers and implement a residual connection.
        # Make sure to upsample the residual before adding it to the layer output.
        fwd = self.fwd(x)
        upsample = self.upsample(x)
        return fwd + upsample

    

class ResBlockDown(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block Downsampler.
    """
    ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
                (conv): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(input_channels, n_filters, kernel_size=(1, 1), stride=(1, 1))
        )
    )
    """
    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlockDown, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.fwd = nn.Sequential(
        nn.ReLU(),
        nn.Conv2d(self.input_channels, self.n_filters, kernel_size = (self.kernel_size, self.kernel_size), padding=(1,1)),
        nn.ReLU(),
        DownSampleConv2D(input_channels = self.n_filters, n_filters = self.n_filters, kernel_size = self.kernel_size, padding = 1)
        )
        self.downsample = nn.Sequential(DownSampleConv2D(input_channels = self.input_channels, n_filters = self.n_filters, kernel_size = 1))

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward through self.layers and implement a residual connection.
        # Make sure to downsample the residual before adding it to the layer output.
        fwd = self.fwd(x)
        downsample = self.downsample(x)
        return fwd + downsample



class ResBlock(torch.jit.ScriptModule):
    # TODO 1.1: Impement Residual Block as described below.
    """
    ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(in_channels, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(n_filters, n_filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
    )
    """

    def __init__(self, input_channels, kernel_size=3, n_filters=128):
        super(ResBlock, self).__init__()
        # TODO 1.1: Setup the network layers
        self.input_channels = input_channels
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.resblock = nn.Sequential(nn.ReLU(),
        nn.Conv2d(self.input_channels, self.n_filters, kernel_size = (self.kernel_size, self.kernel_size), padding=(1,1)),
        nn.ReLU(),
        nn.Conv2d(self.n_filters, self.n_filters, kernel_size = (self.kernel_size, self.kernel_size), padding=(1,1)))

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the conv layers. Don't forget the residual connection!
        residual = self.resblock(x)
        return x + residual


class Generator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Generator. Follow the architecture described below:
    """
    Generator(
    (dense): Linear(in_features=128, out_features=2048, bias=True)
    (layers): Sequential(
        (0): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlockUp(
        (layers): Sequential(
            (0): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (1): ReLU()
            (2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (4): ReLU()
            (5): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (upsample_residual): UpSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (3): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (4): ReLU()
        (5): Conv2d(128, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (6): Tanh()
    )
    )
    """

    def __init__(self, starting_image_size=4):
        super(Generator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.starting_image_size = starting_image_size
        self.dense = nn.Linear(128, 2048, bias=True)
        self.layers = nn.Sequential(ResBlockUp(128, 3, 128),
        ResBlockUp(128, 3, 128),
        ResBlockUp(128, 3, 128),
        nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Conv2d(128, 3, kernel_size = 3, padding=(1,1)),
        nn.Tanh())

    @torch.jit.script_method
    def forward_given_samples(self, z):
        # TODO 1.1: forward the generator assuming a set of samples z have been passed in.
        # Don't forget to re-shape the output of the dense layer into an image with the appropriate size!
        dense = self.dense(z)
        dense = dense.view(-1, 128, self.starting_image_size, self.starting_image_size)
        return self.layers(dense)

    @torch.jit.script_method
    def forward(self, n_samples: int = 1024):
        # TODO 1.1: Generate n_samples latents and forward through the network.
        z = torch.randn(n_samples, 128).cuda()
        return self.forward_given_samples(z)



class Discriminator(torch.jit.ScriptModule):
    # TODO 1.1: Impement Discriminator. Follow the architecture described below:
    """
    Discriminator(
    (layers): Sequential(
        (0): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(3, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(3, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (1): ResBlockDown(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            )
        )
        (downsample_residual): DownSampleConv2D(
            (conv): Conv2d(128, 128, kernel_size=(1, 1), stride=(1, 1))
        )
        )
        (2): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (3): ResBlock(
        (layers): Sequential(
            (0): ReLU()
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
            (2): ReLU()
            (3): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        )
        )
        (4): ReLU()
    )
    (dense): Linear(in_features=128, out_features=1, bias=True)
    )
    """

    def __init__(self):
        super(Discriminator, self).__init__()
        # TODO 1.1: Setup the network layers
        self.layers = nn.Sequential(ResBlockDown(3, 3, 128),
        ResBlockDown(128, 3, 128),
        ResBlock(128, 3, 128),
        ResBlock(128, 3, 128),
        nn.ReLU())
        self.dense = nn.Linear(128, 1, bias=True)

    @torch.jit.script_method
    def forward(self, x):
        # TODO 1.1: Forward the discriminator assuming a batch of images have been passed in.
        # Make sure to sum across the image dimensions after passing x through self.layers.
        forward_pass = self.layers(x)
        forward_pass = forward_pass.sum(dim=(2,3))
        return self.dense(forward_pass)




