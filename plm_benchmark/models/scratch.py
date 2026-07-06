"""From-scratch (randomly initialised) embedding encoder.

Isolates the effect of the *tokenizer* from pretraining: an ``nn.Embedding``
over the tokenizer vocabulary plus a learned positional embedding, exposing the
same interface as a HuggingFace encoder (``.last_hidden_state`` and
``.config.hidden_size``) so it plugs into the shared attention1d head and the
unified training loop unchanged.
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
        max_position: int = 4096,
        pad_token_id: int | None = None,
    ):
        super().__init__()
        self.config = ScratchConfig(hidden_size=hidden_size)
        self.token = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_token_id)
        self.position = nn.Embedding(max_position, hidden_size)

    def forward(self, input_ids=None, attention_mask=None, **kwargs) -> BaseModelOutput:
        seq_len = input_ids.size(1)
        pos_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        hidden = self.token(input_ids) + self.position(pos_ids)
        return BaseModelOutput(last_hidden_state=hidden)
