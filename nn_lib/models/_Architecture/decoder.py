import torch

import torch.nn as nn



class Decoder_LgeFormated(nn.Module):
    def __init__(
            self,
            in_shape=None,
            in_channel=None,
            lat_dims=None,
            kernel=None,
            scale_shape=None
        ):

        super().__init__()
        self.in_shape = in_shape
        self.in_channel = in_channel
        self.lat_dims = lat_dims

        self.f = nn.Sequential(
            nn.Linear(lat_dims, int(in_channel*(in_shape[1]/8)*(in_shape[2]/8))),
            nn.Unflatten(1,(in_channel,int(in_shape[1]/8),int(in_shape[2]/8))),

            nn.Upsample(scale_factor=scale_shape, mode='nearest'),
            nn.Conv2d(in_channel, in_channel, kernel_size=kernel, padding="same"),
            nn.Conv2d(in_channel, 64, kernel_size=kernel, padding="same"),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=scale_shape, mode='nearest'),
            nn.Conv2d(64, 64, kernel_size=kernel, padding="same"),
            nn.Conv2d(64, 32, kernel_size=kernel, padding="same"),
            nn.LeakyReLU(),

            nn.Upsample(scale_factor=scale_shape, mode='nearest'),
            nn.Conv2d(32, 32, kernel_size=kernel, padding="same"),
            nn.Conv2d(32, 1, kernel_size=kernel, padding="same"),

            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.f(x)
    
    