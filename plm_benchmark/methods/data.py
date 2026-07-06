"""Datasets and dynamic-padding collators shared by all downstream tasks."""

from __future__ import annotations

import torch
from datasets import Dataset

from plm_benchmark.models.ppi import tokenize_ppi_pairs
from plm_benchmark.tasks import TaskSpec


def label_dtype(spec: TaskSpec) -> torch.dtype:
    if spec.task_type in ("classification", "ppi"):
        return torch.long
    return torch.float  # regression, multilabel


def single_dataset(df, spec: TaskSpec) -> Dataset:
    labels = list(df["label"])
    if spec.task_type in ("classification", "ppi"):
        labels = [int(x) for x in labels]
    elif spec.task_type == "regression":
        labels = [float(x) for x in labels]
    seqs = list(df["sequence"])
    return Dataset.from_dict(
        {"sequence": seqs, "labels": labels, "length": [len(s) for s in seqs]}
    )


def ppi_dataset(df) -> Dataset:
    seq_a = list(df["sequence_a"])
    seq_b = list(df["sequence_b"])
    return Dataset.from_dict(
        {
            "sequence_a": seq_a,
            "sequence_b": seq_b,
            "labels": list(df["label"].astype(int)),
            "length": [len(a) + len(b) for a, b in zip(seq_a, seq_b)],
        }
    )


def collate_single(batch, tokenizer, max_length: int, spec: TaskSpec):
    seqs = [x["sequence"] for x in batch]
    labels = torch.tensor([x["labels"] for x in batch], dtype=label_dtype(spec))
    tok = tokenizer(seqs, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
    tok["labels"] = labels
    return tok


def collate_ppi(batch, tokenizer, max_length: int):
    seq_a = [x["sequence_a"] for x in batch]
    seq_b = [x["sequence_b"] for x in batch]
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    tok = tokenize_ppi_pairs(tokenizer, seq_a, seq_b, max_length=max_length)
    tok["labels"] = labels
    return tok
