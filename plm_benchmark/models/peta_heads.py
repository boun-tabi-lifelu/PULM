
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PetaHeadConfig:
    hidden_size: int
    num_labels: int


class MaskedConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding=padding,
        )

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


class Attention1dPooling(nn.Module):
    def __init__(self, config: PetaHeadConfig):
        super().__init__()
        self.layer = MaskedConv1d(config.hidden_size, 1, 1)

    def forward(self, x, input_mask=None):
        batch_size = x.shape[0]
        attn = self.layer(x, input_mask=input_mask)
        attn = attn.view(batch_size, -1)
        if input_mask is not None:
            if input_mask.dim() == 3:
                mask = input_mask.view(batch_size, -1).bool()
            else:
                mask = input_mask.bool()
            attn = attn.masked_fill_(~mask, float("-inf"))
        attn = F.softmax(attn, dim=-1).view(batch_size, -1, 1)
        return (attn * x).sum(dim=1)


class Attention1dPoolingProjection(nn.Module):
    def __init__(self, config: PetaHeadConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.relu = nn.ReLU()
        self.final = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.relu(self.linear(x))
        return self.final(x)


class Attention1dPoolingHead(nn.Module):
    """Single-sequence path (non-PPI)."""

    def __init__(self, config: PetaHeadConfig):
        super().__init__()
        self.attention1d = Attention1dPooling(config)
        self.attention1d_projection = Attention1dPoolingProjection(config)

    def forward(self, x, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        pooled = self.attention1d(x, input_mask=mask)
        return self.attention1d_projection(pooled)


class Attention1dPPIHead(nn.Module):
    """PPI path: pool each protein, sum pair embeddings, then project."""

    def __init__(self, config: PetaHeadConfig):
        super().__init__()
        self.pooling = Attention1dPooling(config)
        self.projection = Attention1dPoolingProjection(config)

    def forward(self, x, attention_mask):
        mask = attention_mask.unsqueeze(-1)
        pooled = self.pooling(x, input_mask=mask)
        batch_pairs = pooled.size(0) // 2
        pair_repr = pooled.view(batch_pairs, 2, -1).sum(dim=1)
        return self.projection(pair_repr)
