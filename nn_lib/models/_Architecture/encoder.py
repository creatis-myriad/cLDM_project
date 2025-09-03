import torch.nn as nn



class Encoder_LgeFormated(nn.Module):
    def __init__(
            self,
            in_shape=None,
            in_channel=None,
            lat_dims=None,
            kernel=None,
            pooling_shape=None
        ):

        super().__init__()
        self.in_shape = in_shape
        self.in_channel = in_channel
        self.lat_dims = lat_dims

        self.f = nn.Sequential(
            nn.Conv2d(in_channel, 32, kernel_size=kernel, padding="same"),
            nn.Conv2d(32, 32, kernel_size=kernel, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling_shape, stride=pooling_shape),

            nn.Conv2d(32, 64, kernel_size=kernel, padding="same"),
            nn.Conv2d(64, 64, kernel_size=kernel, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling_shape, stride=pooling_shape),

            nn.Conv2d(64, 128, kernel_size=kernel, padding="same"),
            nn.Conv2d(128, 128, kernel_size=kernel, padding="same"),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=pooling_shape, stride=pooling_shape),

            nn.Flatten(start_dim=1,end_dim=-1),
            nn.Linear(int(128*(in_shape[1]/8)*(in_shape[2]/8)), lat_dims),
        )

    def forward(self, x):
        return self.f(x)



