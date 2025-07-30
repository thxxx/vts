import math
from typing import List

import numpy as np
import torch
from einops import rearrange, reduce, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from vocos.dac.layers import Snake1d
from vocos.dac.model import ResidualUnit

__all__ = [
    "dacvae_1024dim_small",  # 21.5fps
    "dacvae_1024dim_mid",  # 42 fps
    "dacvae_1024dim_large",  # 93 fps
]


class EncoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            ResidualUnit(in_dim, dilation=1),
            ResidualUnit(in_dim, dilation=3),
            ResidualUnit(in_dim, dilation=9),
            Snake1d(in_dim),
            nn.Conv1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
        )
        self.ratio = out_dim // in_dim
        self.group_size = in_dim * stride // out_dim

    def shortcut(self, x):
        x = reduce(
            x, "b c (l r g) -> b (c r) l", "mean", r=self.ratio, g=self.group_size
        )
        return x

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        strides: list = [2, 4, 8, 8],
        latent_dim: int = 1024,
    ):
        super().__init__()
        layers = []

        channels = base_channels
        self.init_conv = nn.Conv1d(in_channels, channels, kernel_size=7, padding=3)

        for stride in strides:
            in_dim = channels
            channels *= 2

            layers.append(EncoderBlock(in_dim, channels, stride=stride))

        self.latent_channels = channels
        layers.append(Snake1d(channels))
        self.last_layer = nn.Conv1d(channels, latent_dim, kernel_size=3, padding=1)

        self.last_channel = channels
        self.latent_dim = latent_dim
        self.shortcut_group_dim = self.last_channel // self.latent_dim
        self.net = nn.Sequential(*layers)

        self.fc_mean = nn.Linear(latent_dim, latent_dim)
        self.fc_log_var = nn.Linear(latent_dim, latent_dim)

    def head_shortcut(self, x):
        x = reduce(x, "b (d s) l -> b d l", "mean", s=self.shortcut_group_dim)
        return x

    def forward(self, x):
        x = self.init_conv(x)
        x = self.net(x)
        x = self.last_layer(x) + self.head_shortcut(x)
        x = rearrange(x, "b c l -> b l c")

        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return rearrange(mean, "b l c -> b c l"), rearrange(log_var, "b l c -> b c l")


class DecoderBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            Snake1d(in_dim),
            nn.ConvTranspose1d(
                in_dim,
                out_dim,
                kernel_size=2 * stride,
                stride=stride,
                padding=math.ceil(stride / 2),
            ),
            ResidualUnit(out_dim, dilation=1),
            ResidualUnit(out_dim, dilation=3),
            ResidualUnit(out_dim, dilation=9),
        )

        self.ratio = in_dim // out_dim
        self.group_size = out_dim * stride // in_dim

    def shortcut(self, x):
        x = repeat(x, "b (c r) l -> b c (l r g)", r=self.ratio, g=self.group_size)
        return x

    def forward(self, x):
        return self.block(x) + self.shortcut(x)


class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        base_channels: int = 1536,
        strides: list = [8, 8, 4, 2],
        out_channels: int = 1,
    ):
        super().__init__()
        layers = []
        channels = base_channels
        self.latent_dim = latent_dim
        self.base_channels = base_channels
        self.repeats = base_channels // latent_dim
        assert self.base_channels % self.latent_dim == 0

        self.init_conv = nn.Conv1d(latent_dim, channels, kernel_size=7, padding=3)

        for stride in strides:
            in_dim = channels
            out_dim = channels // 2
            layers.append(DecoderBlock(in_dim, out_dim, stride=stride))
            channels = out_dim

        layers.append(Snake1d(channels))
        self.last_conv = nn.Conv1d(channels, out_channels, kernel_size=7, padding=3)

        self.net = nn.Sequential(*layers)

    def init_shortcut(self, x):
        x = repeat(x, "b c l -> b (c r) l", r=self.repeats)
        return x

    def forward(self, x):
        x = self.init_conv(x) + self.init_shortcut(x)
        x = self.net(x)
        x = self.last_conv(x)
        return x


class DACVAE(nn.Module):
    def __init__(
        self,
        encoder_dim: int = 64,
        encoder_rates: List[int] = [2, 4, 8, 8],
        latent_dim: int | None = None,
        decoder_dim: int = 1536,
        decoder_rates: List[int] = [8, 8, 4, 2],
        sample_rate: int = 48000,
    ):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        self.sample_rate = sample_rate

        if latent_dim is None:
            self.latent_dim = encoder_dim * (2 ** len(encoder_rates))
        else:
            self.latent_dim = latent_dim

        self.hop_length = np.prod(encoder_rates)

        self.encoder = Encoder(
            in_channels=1,
            base_channels=encoder_dim,
            strides=encoder_rates,
            latent_dim=self.latent_dim,
        )

        self.decoder = Decoder(
            latent_dim=self.latent_dim,
            base_channels=decoder_dim,
            strides=decoder_rates,
            out_channels=1,
        )

    def preprocess(self, audio_data, sample_rate):
        if sample_rate is None:
            sample_rate = self.sample_rate
        assert sample_rate == self.sample_rate

        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.hop_length) * self.hop_length - length
        audio_data = F.pad(audio_data, (0, right_pad))

        return audio_data

    def encode(self, audio_data: Tensor, sample_rate: int | None = None):
        audio_data = self.preprocess(audio_data, sample_rate)
        mean, scale = self.encoder(audio_data)
        stdev = F.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        z = torch.randn_like(mean) * stdev + mean
        kl_loss = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, mean, scale, kl_loss

    def decode(self, z: Tensor):
        decoded_audio = self.decoder(z)
        return decoded_audio

    def forward(self, audio_data: Tensor, sample_rate: int | None = None):
        length = audio_data.shape[-1]
        audio_data = self.preprocess(audio_data, sample_rate)

        mean, scale = self.encoder(audio_data)
        stdev = F.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        z = torch.randn_like(mean) * stdev + mean

        kl_loss = (mean * mean + var - logvar - 1).sum(1).mean()

        decoded_audio = self.decoder(z)
        return decoded_audio[..., :length], z, kl_loss


def dacvae_1024dim_small():
    return DACVAE(
        encoder_dim=64,
        encoder_rates=[2, 4, 4, 8, 8],
        latent_dim=1024,
        decoder_dim=2048,
        decoder_rates=[8, 8, 4, 4, 2],
        sample_rate=48000,
    )


def dacvae_1024dim_mid():
    return DACVAE(
        encoder_dim=64,
        encoder_rates=[2, 2, 4, 8, 8],
        latent_dim=1024,
        decoder_dim=2048,
        decoder_rates=[8, 8, 4, 2, 2],
        sample_rate=48000,
    )


def dacvae_1024dim_large():
    return DACVAE(
        encoder_dim=64,
        encoder_rates=[2, 4, 8, 8],
        latent_dim=1024,
        decoder_dim=2048,
        decoder_rates=[8, 8, 4, 2],
        sample_rate=48000,
    )
