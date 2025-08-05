from typing import cast

import torch
from einops import einsum, rearrange
from torch import Tensor, nn
from torch.nn import functional as F

from model.norm import RMSNorm
from utils.typing import EncMaskTensor


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.to_out = nn.Linear(dim, dim)
        slope = -torch.arange(1, self.heads + 1) * 8 / self.heads
        slope = slope.exp2()
        self.register_buffer("slope", slope, persistent=False)

    def get_alibi_bias(self, length: int, device: torch.device):
        arange = torch.arange(length, device=device, dtype=self.slope.dtype)
        rel = rearrange(arange, "q -> q ()") - rearrange(arange, "k -> () k")
        bias = einsum(rel, self.slope, "q k, h -> h q k")
        return -bias.abs()

    def forward(self, x: Tensor, key_mask: EncMaskTensor):
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)

        q = rearrange(q, "b q (h d) -> b h q d", h=self.heads)
        k = rearrange(k, "b k (h d) -> b h k d", h=self.heads)
        v = rearrange(v, "b k (h d) -> b h k d", h=self.heads)

        attn_mask = rearrange(key_mask, "b k -> b () () k")
        attn_mask = torch.where(attn_mask, 0, -torch.inf)
        alibi_bias = self.get_alibi_bias(q.shape[2], q.device)
        attn_mask = attn_mask + alibi_bias
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = rearrange(out, "b h q d -> b q (h d)")
        return self.to_out(out)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.to_out = nn.Linear(dim, dim)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        key_mask: EncMaskTensor,
    ):
        q = self.to_q(query)
        k, v = self.to_kv(key).chunk(2, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.heads)

        attn_mask = rearrange(key_mask, "b k -> b () () k")
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )

        out = rearrange(out, "b h q d -> b q (h d)")
        return self.to_out(out)


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        skip: bool,
    ):
        super().__init__()
        self.skip = skip
        self.skip_combiner = nn.Linear(dim * 2, dim) if skip else nn.Identity()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dropout=attn_dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
        )


class TransformerDecoderBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
        skip: bool,
    ):
        super().__init__()
        self.skip = skip
        self.skip_combiner = nn.Linear(dim * 2, dim) if skip else nn.Identity()
        self.attn_norm = RMSNorm(dim)
        self.attn = SelfAttention(dim, heads=heads, dropout=attn_dropout)
        self.cross_attn_norm = RMSNorm(dim)
        self.cross_attn = CrossAttention(dim, heads=heads, dropout=attn_dropout)
        self.ff_norm = RMSNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.ReLU(),
            nn.Dropout(ff_dropout),
            nn.Linear(dim * 4, dim),
        )


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()

        self.layers = cast(
            list[TransformerEncoderBlock],
            nn.ModuleList(
                TransformerEncoderBlock(
                    dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    skip=ind + 1 > (depth // 2),
                )
                for ind in range(depth)
            ),
        )

        self.final_norm = RMSNorm(dim)

    def forward(self, x: Tensor, mask: EncMaskTensor) -> Tensor:
        skip_connects = []
        for layer in self.layers:
            if layer.skip:
                skip_connect = skip_connects.pop()
                x = torch.cat((x, skip_connect), dim=-1)
            else:
                skip_connects.append(x)

            x = layer.skip_combiner(x)

            attn_input = layer.attn_norm(x)
            x = layer.attn(attn_input, key_padding_mask=mask) + x

            ff_input = layer.ff_norm(x)
            x = layer.ff(ff_input) + x

        return self.final_norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        attn_dropout: float,
        ff_dropout: float,
    ):
        super().__init__()

        self.layers = cast(
            list[TransformerDecoderBlock],
            nn.ModuleList(
                TransformerDecoderBlock(
                    dim,
                    heads=heads,
                    attn_dropout=attn_dropout,
                    ff_dropout=ff_dropout,
                    skip=ind + 1 > (depth // 2),
                )
                for ind in range(depth)
            ),
        )

        self.final_norm = RMSNorm(dim)

    def forward(
        self, x: Tensor, key: Tensor, mask: EncMaskTensor, key_mask: EncMaskTensor
    ) -> Tensor:
        skip_connects = []
        for layer in self.layers:
            if layer.skip:
                skip_connect = skip_connects.pop()
                x = torch.cat((x, skip_connect), dim=-1)
            else:
                skip_connects.append(x)

            x = layer.skip_combiner(x)

            attn_input = layer.attn_norm(x)
            x = layer.attn(attn_input, key_mask=mask) + x

            attn_input = layer.cross_attn_norm(x)
            x = layer.cross_attn(attn_input, key=key, key_mask=key_mask) + x

            ff_input = layer.ff_norm(x)
            x = layer.ff(ff_input) + x

        return self.final_norm(x)
