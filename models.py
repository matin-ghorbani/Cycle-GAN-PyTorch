import torch
from torch import nn


# region Discriminator


class DisBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, padding_mode='reflect'),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(.2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(
                in_channels,
                features[0],
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect'
            ),
            nn.LeakyReLU(.2)
        )

        layers = nn.ModuleList()
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(DisBlock(
                in_channels=in_channels,
                out_channels=feature,
                stride=1 if feature == features[-1] else 2
            ))
            in_channels = feature

        layers.append(nn.Conv2d(
            in_channels,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
            padding_mode='reflect'
        ))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        return torch.sigmoid(self.model(x))


# endregion


# region Generator

class GenBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        """
        Main block of generator for down sampling and up sampling
        :param down: whether it is down sampled or not
        """
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, padding_mode='reflect', **kwargs)
            if down else
            nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),

            # nn.Identity: Is just gonna pass it trow
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            GenBlock(channels, channels, kernel_size=3, padding=1),
            GenBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, img_channels, num_features=64, num_res_blocks=9):
        super().__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(img_channels, num_features, kernel_size=7, stride=1, padding=3, padding_mode='reflect'),
            nn.ReLU(inplace=True)
        )
        self.down_blocks = nn.ModuleList([
            GenBlock(num_features, num_features * 2, kernel_size=3, stride=2, padding=1),
            GenBlock(num_features * 2, num_features * 4, kernel_size=3, stride=2, padding=1),
        ])

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features * 4) for _ in range(num_res_blocks)]
        )

        self.up_blocks = nn.ModuleList([
            GenBlock(
                num_features * 4,
                num_features * 2,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            ),
            GenBlock(
                num_features * 2,
                num_features,
                down=False,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1
            )
        ])

        self.last = nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=3, padding_mode='reflect')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        for down_layer in self.down_blocks:
            x = down_layer(x)
        x = self.residual_blocks(x)
        for up_layer in self.up_blocks:
            x = up_layer(x)

        return torch.tanh(self.last(x))


# endregion


def discriminator_test():
    x = torch.randn((1, 3, 256, 256))
    model = Discriminator()
    output = model(x)
    print(f'{output.shape = }')


def generator_test():
    img_channels = 3
    img_size = 256

    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, 9)
    output = gen(x)
    print(f'{output.shape = }')


if __name__ == '__main__':
    # discriminator_test()
    generator_test()
