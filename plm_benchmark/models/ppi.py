
from __future__ import annotations


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
