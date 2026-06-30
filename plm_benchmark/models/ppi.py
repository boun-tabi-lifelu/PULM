
from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import SequenceClassifierOutput


def mean_pool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    summed = (hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class PPIPairClassifier(nn.Module):
    """Pair two consecutive rows in a batch as one PPI example (2N rows -> N pairs)."""

    def __init__(self, encoder: AutoModel, num_labels: int, hidden_size: int):
        super().__init__()
        self.encoder = encoder
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        **kwargs,
    ) -> SequenceClassifierOutput:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = mean_pool(outputs.last_hidden_state, attention_mask)
        batch_pairs = pooled.size(0) // 2
        pair_repr = pooled.view(batch_pairs, 2, -1).sum(dim=1)
        logits = self.classifier(pair_repr)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


def load_ppi_model(
    checkpoint: str,
    num_labels: int,
    *,
    method: str = "full_ft",
    lora_r: int = 4,
) -> tuple[PPIPairClassifier, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    encoder = AutoModel.from_pretrained(checkpoint, add_pooling_layer=False)

    if method == "lora":
        config = LoraConfig(
            r=lora_r, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
        )
        encoder = inject_adapter_in_model(config, encoder)

    hidden = encoder.config.hidden_size
    model = PPIPairClassifier(encoder, num_labels, hidden)

    if method == "lora":
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model, tokenizer


def tokenize_ppi_pairs(
    tokenizer,
    seq_a: list[str],
    seq_b: list[str],
    *,
    max_length: int = 1024,
) -> dict:
    """Interleave A/B so row 2i and 2i+1 form pair i (PETA batch layout)."""
    flat: list[str] = []
    for a, b in zip(seq_a, seq_b):
        flat.append(a)
        flat.append(b)
    return tokenizer(flat, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
