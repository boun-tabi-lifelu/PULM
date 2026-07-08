"""From-scratch (randomly initialised) encoder for the tokenizer baseline.

Isolates the effect of the *tokenizer* from pretraining. It exposes the same
interface as a HuggingFace encoder (``.last_hidden_state`` and
``.config.hidden_size``) so it plugs into the shared attention1d head unchanged.

Capacity is a dial (``num_layers``):

* ``num_layers=0`` — a **bag-of-tokens** model: token embedding + positional
  embedding + LayerNorm, then straight to pooling. No token ever sees another
  token, so it measures the signal the vocabulary carries on its own. A crippled
  but clean floor.
* ``num_layers>=1`` — adds that many Transformer blocks (self-attention + FFN +
  LayerNorm), giving the model **context** (token↔token interaction). A fairer
  "small non-pretrained model" baseline. ``num_heads`` sets attention heads per
  block and must divide ``hidden_size``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class ScratchConfig:
    hidden_size: int
    model_type: str = "scratch"


class ScratchEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 320,
        *,
        num_layers: int = 0,
        num_heads: int = 8,
        max_position: int = 4096,
        pad_token_id: int | None = None,
        dropout: float = 0.1,
        ff_mult: int = 4,
    ):
        super().__init__()
        if num_layers > 0 and hidden_size % num_heads != 0:
            raise ValueError(
                f"scratch hidden_size ({hidden_size}) must be divisible by "
                f"--scratch-heads ({num_heads})."
            )
        self.config = ScratchConfig(hidden_size=hidden_size)
        self.token = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position = nn.Embedding(max_position, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        if num_layers > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=ff_mult * hidden_size,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,  # pre-LN: more stable to train from scratch
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        else:
            self.encoder = None

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> BaseModelOutput:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token(input_ids) + self.position(pos_ids)
        hidden = self.dropout(self.norm(hidden))
        if self.encoder is not None:
            # src_key_padding_mask: True marks padding positions to ignore.
            pad_mask = ~attention_mask.bool() if attention_mask is not None else None
            hidden = self.encoder(hidden, src_key_padding_mask=pad_mask)
        return BaseModelOutput(last_hidden_state=hidden)
