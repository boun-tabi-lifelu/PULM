"""ESM / PULM model loading and embedding."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, inject_adapter_in_model
from tqdm import tqdm
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from plm_benchmark.config import MAX_SEQ_LENGTH


def set_gpu(gpu_index: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index - 1)


def load_encoder(checkpoint: str, device: torch.device | None = None) -> tuple[AutoModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModel.from_pretrained(checkpoint)
    if device is not None:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def load_classifier(
    checkpoint: str,
    num_labels: int,
    *,
    method: str = "full_ft",
    lora_r: int = 4,
    problem_type: str | None = None,
) -> tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint, num_labels=num_labels, ignore_mismatched_sizes=True
    )
    if problem_type:
        model.config.problem_type = problem_type

    if method == "lora":
        config = LoraConfig(
            r=lora_r, lora_alpha=1, bias="all", target_modules=["query", "key", "value", "dense"]
        )
        model = inject_adapter_in_model(config, model)
        for _, p in model.classifier.named_parameters():
            p.requires_grad = True

    return model, tokenizer


def tokenize_dataset(tokenizer, sequences: list, labels: list, max_length: int = MAX_SEQ_LENGTH) -> Dataset:
    tok = tokenizer(sequences, max_length=max_length, padding=True, truncation=True)
    return Dataset.from_dict(tok).add_column("labels", labels)


@torch.no_grad()
def mean_pool_embeddings(
    checkpoint: str,
    df: pd.DataFrame,
    gpu: int,
    *,
    max_length: int = MAX_SEQ_LENGTH,
    batch_size: int = 4,
    desc: str = "embed",
) -> pd.DataFrame:
    set_gpu(gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_encoder(checkpoint, device)

    print(f"[{desc}] {len(df)} sequences, batch={batch_size}, max_len={max_length}", flush=True)
    rows = []
    n = len(df)
    for start in tqdm(range(0, n, batch_size), desc=desc, mininterval=5.0):
        seqs = df["sequence"].iloc[start : start + batch_size].tolist()
        inputs = tokenizer(seqs, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        hidden = model(**inputs).last_hidden_state
        pooled = torch.mean(hidden, dim=1).cpu().numpy()
        rows.extend(pooled[j] for j in range(pooled.shape[0]))

    del model, tokenizer
    torch.cuda.empty_cache()

    emb = pd.DataFrame(np.vstack(rows))
    emb["sequence"] = df["sequence"].values
    emb["label"] = df["label"].values
    return emb
