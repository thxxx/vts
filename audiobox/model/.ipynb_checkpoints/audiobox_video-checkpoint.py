import math

import torch
from einops import einsum, rearrange, repeat
from torch import Tensor, nn
from torch.nn import functional as F

from model.transformer import TransformerDecoder
from model.conv import ConvNeXtV2Block
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

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, n: int, device='cpu'):
        position = torch.arange(n, dtype=torch.float32, device=device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.dim, 2, device=device).float() * (-math.log(10000.0) / self.dim))
        pe = torch.zeros(n, self.dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (N, dim)

class AudioBox(nn.Module):
    def __init__(
        self,
        audio_dim: int,
        text_dim: int,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        kernel_size: int,
    ):
        super().__init__()
        self.text_dim = text_dim

        self.combine = nn.Linear(audio_dim * 2, dim)
        self.conv_embed = ConvPositionEmbed(dim=dim, kernel_size=kernel_size)
        self.phoneme_linear = nn.Linear(text_dim, dim)
        self.time_emb = TimeEncoding(dim)

        self.transformer = TransformerDecoder(
            dim=dim,
            depth=depth,
            heads=heads,
            ff_dropout=ff_dropout,
            attn_dropout=attn_dropout,
        )

        self.to_pred = nn.Linear(dim, audio_dim, bias=False)

        self.video_latent_dim = 1280
        # self.frame_to_one = nn.Conv3d(
        #     in_channels=self.video_latent_dim,
        #     out_channels=self.video_latent_dim,
        #     kernel_size=(1, 4, 4),
        #     stride=(1, 4, 4),
        #     padding=0,
        # )

        conv_mult = 2
        conv_layers = 2
        self.convnexts = nn.Sequential(
            *[ConvNeXtV2Block(self.video_latent_dim, self.video_latent_dim * conv_mult) for _ in range(conv_layers)]
        )
        
        self.video_to_cross_dim = nn.Sequential(
            # nn.Linear(self.video_latent_dim, self.video_latent_dim*4),
            # nn.SiLU(),
            nn.Linear(self.video_latent_dim, dim),
        )
        nn.init.zeros_(self.video_to_cross_dim[-1].weight)
        nn.init.zeros_(self.video_to_cross_dim[-1].bias)

        conv_mult = 2
        conv_layers = 2
        self.convnextssync = nn.Sequential( # 이게 non-linear하니까 ㅇㅋ
            *[ConvNeXtV2Block(768, 768 * conv_mult) for _ in range(conv_layers)]
        )
        # self.to_ln = nn.Linear(768 + dim, dim)
        # nn.init.zeros_(self.to_ln.weight)
        # nn.init.zeros_(self.to_ln.bias)

        self.to_ln = nn.Linear(768 + dim, dim)

        # 전체 weight 0으로 초기화
        nn.init.zeros_(self.to_ln.weight)
        nn.init.zeros_(self.to_ln.bias)
        
        # 뒤쪽 dim 부분을 identity로 설정
        with torch.no_grad():
            self.to_ln.weight[:, 768:] = torch.eye(dim)

        self.video_pos_emb = SinusoidalPosEmb(dim)
    
    def cfg(
        self,
        w: EncTensor,
        context: EncTensor,
        mask: EncMaskTensor,
        times: TimeTensor,
        text_emb: EncTensor,
        text_mask: EncMaskTensor,
        video_latent: EncTensor,
        sync_latent: EncTensor,
        alpha=0.0,
    ) -> EncTensor:
        w = repeat(w, "b ... -> (r b) ...", r=2)
        # video_latent = repeat(video_latent, "b ... -> (r b) ...", r=2)
        context = repeat(context, "b ... -> (r b) ...", r=2)
        mask = repeat(mask, "b ... -> (r b) ...", r=2)
        times = repeat(times, "b ... -> (r b) ...", r=2)
        
        text_emb = torch.cat((text_emb, torch.zeros_like(text_emb)), dim=0)
        text_mask = repeat(text_mask, "b ... -> (r b) ...", r=2)
        
        video_latent = torch.cat((video_latent, torch.zeros_like(video_latent)), dim=0)
        sync_latent = torch.cat((sync_latent, torch.zeros_like(sync_latent)), dim=0)

        logits, null_logits = self(
            w=w,
            context=context,
            audio_mask=mask,
            times=times,
            text_emb=text_emb,
            text_mask=text_mask,
            video_latent=video_latent,
            sync_latent=sync_latent
        ).chunk(2, dim=0)
        
        return logits + alpha * (logits - null_logits)
    
    def forward(
        self,
        w: EncTensor,
        context: EncTensor,
        audio_mask: EncMaskTensor,
        times: TimeTensor,
        text_emb: EncTensor,
        text_mask: EncMaskTensor,
        video_latent: EncTensor,
        sync_latent: EncTensor
    ) -> EncTensor:
        fvl = self.convnexts(video_latent) # B N D
        fvl = self.video_to_cross_dim(fvl)
        
        pe = self.video_pos_emb(n=video_latent.shape[1], device=video_latent.device)  # (N, dim)
        pe = pe.unsqueeze(0).expand(video_latent.size(0), -1, -1)  # (B, N, dim)
        fvl = fvl + pe  # (B, N, dim)
        
        embed = torch.cat((w, context), dim=-1)
        combined = self.combine(embed)

        w = self.conv_embed(combined, audio_mask)

        sync_context = self.convnextssync(sync_latent)
        sync_context = torch.cat((sync_context, w), dim=-1)
        w = self.to_ln(sync_context)
        # w = w + sync_context

        time_emb = self.time_emb(times)
        text_emb = self.phoneme_linear(text_emb)

        cross_emb = torch.cat((time_emb, text_emb, fvl), dim=1)
        text_mask = F.pad(text_mask, (1, fvl.shape[1]), value=1) # 왼쪽에 1개(for time), 오른쪽에 3fps * seconds개
        # text_mask = F.pad(text_mask, (0, ), value=1)

        w = self.transformer(w, mask=audio_mask, key=cross_emb, key_mask=text_mask)

        w = self.to_pred(w)
        return w
