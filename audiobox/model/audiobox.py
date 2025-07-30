import math

import torch
from einops import einsum, rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from model.transformer import TransformerDecoder
from utils.typing import EncMaskTensor, EncTensor, TimeTensor


class ConvPositionEmbed(nn.Module):
    def __init__(self, dim: int, kernel_size: int):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel size must be odd for ConvPositionEmbed"
        self.conv1 = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.conv2 = nn.Conv1d(
            dim, dim, kernel_size, padding=kernel_size // 2, groups=16
        )
        self.gelu = nn.GELU()

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        origin = x
        mask = rearrange(~mask, "b n -> b () n")
        x = rearrange(x, "b n c -> b c n")
        x = x.masked_fill(mask, 0.0)
        x = self.conv1(x)
        x = self.gelu(x)
        x = x.masked_fill(mask, 0.0)
        x = self.conv2(x)
        x = self.gelu(x)
        x = x.masked_fill(mask, 0.0)
        x = rearrange(x, "b c n -> b n c")
        return x + origin


class TimeEncoding(nn.Module):
    """used by @crowsonkb"""

    def __init__(self, dim: int):
        super().__init__()
        assert dim % 2 == 0, "dimension must be divisible by 2"
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x: Tensor) -> Tensor:
        freqs = 2 * math.pi * einsum(x, self.weights, "b, d -> b d")
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return rearrange(fouriered, "b d -> b () d")


class AudioBox(nn.Module):
    def __init__(
        self,
        audio_dim: int,
        phoneme_dim: int,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        kernel_size: int,
    ):
        super().__init__()

        # self.bottleneck = nn.Sequential(
        #     nn.Linear(dim, dim//4),
        #     nn.ReLU(),
        #     nn.Linear(dim//4, dim)
        # )
        # self.to_input = nn.Linear(dim * 2, dim)

        self.combine = nn.Linear(audio_dim * 2, dim)
        self.conv_embed = ConvPositionEmbed(dim=dim, kernel_size=kernel_size)
        self.phoneme_linear = nn.Linear(phoneme_dim, dim)
        self.time_emb = TimeEncoding(dim)

        self.transformer = TransformerDecoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )

        self.to_pred = nn.Linear(dim, audio_dim, bias=False)

    def cfg(
        self,
        w: EncTensor,
        context: EncTensor,
        mask: EncMaskTensor,
        times: TimeTensor,
        phoneme_emb: EncTensor,
        phoneme_mask: EncMaskTensor,
        alpha=0.0,
    ) -> EncTensor:
        w = repeat(w, "b ... -> (r b) ...", r=2)
        context = repeat(context, "b ... -> (r b) ...", r=2)
        mask = repeat(mask, "b ... -> (r b) ...", r=2)
        times = repeat(times, "b ... -> (r b) ...", r=2)
        
        phoneme_emb = torch.cat((phoneme_emb, torch.zeros_like(phoneme_emb)), dim=0)
        phoneme_mask = repeat(phoneme_mask, "b ... -> (r b) ...", r=2)
        
        logits, null_logits = self(
            w=w,
            context=context,
            audio_mask=mask,
            times=times,
            phoneme_emb=phoneme_emb,
            phoneme_mask=phoneme_mask,
        ).chunk(2, dim=0)
        return logits + alpha * (logits - null_logits)
    
    def cfg_negative(
        self,
        w: EncTensor,
        context: EncTensor,
        mask: EncMaskTensor,
        times: TimeTensor,
        phoneme_emb: EncTensor,
        phoneme_mask: EncMaskTensor,
        negative_phoneme_emb: EncTensor,
        negative_phoneme_mask: EncMaskTensor,
        alpha=0.0,
        nalpha=1.0
    ) -> EncTensor:
        r_number = 2
        
        w = repeat(w, "b ... -> (r b) ...", r=r_number)
        context = repeat(context, "b ... -> (r b) ...", r=r_number)
        mask = repeat(mask, "b ... -> (r b) ...", r=r_number)
        times = repeat(times, "b ... -> (r b) ...", r=r_number)
        
        phoneme_emb = torch.cat((phoneme_emb, torch.zeros_like(phoneme_emb)), dim=0)
        phoneme_mask = repeat(phoneme_mask, "b ... -> (r b) ...", r=r_number)
        
        negative_phoneme_emb = torch.cat((negative_phoneme_emb, torch.zeros_like(negative_phoneme_emb)), dim=0)
        negative_phoneme_mask = repeat(negative_phoneme_mask, "b ... -> (r b) ...", r=r_number)
        
        logits, null_logits = self(
            w=w,
            context=context,
            audio_mask=mask,
            times=times,
            phoneme_emb=phoneme_emb,
            phoneme_mask=phoneme_mask,
        ).chunk(r_number, dim=0)
        
        negative_logits, negative_null_logits = self(
            w=w,
            context=context,
            audio_mask=mask,
            times=times,
            phoneme_emb=negative_phoneme_emb,
            phoneme_mask=negative_phoneme_mask,
        )
        
        return logits + alpha * (logits - null_logits) - nalpha * (negative_logits - negative_null_logits)

    def forward(
        self,
        w: EncTensor,
        context: EncTensor,
        audio_mask: EncMaskTensor,
        times: TimeTensor,
        phoneme_emb: EncTensor,
        phoneme_mask: EncMaskTensor,
    ) -> EncTensor:
        embed = torch.cat((w, context), dim=-1)

        combined = self.combine(embed)
        w = self.conv_embed(combined, audio_mask)

        time_emb = self.time_emb(times)
        phoneme_emb = self.phoneme_linear(phoneme_emb)
        phoneme_emb = torch.cat((time_emb, phoneme_emb), dim=1)
        phoneme_mask = F.pad(phoneme_mask, (1, 0), value=1)

        w = self.transformer(w, mask=audio_mask, key=phoneme_emb, key_mask=phoneme_mask)

        w = self.to_pred(w)
        return w
