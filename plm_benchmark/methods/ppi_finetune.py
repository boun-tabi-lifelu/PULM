
from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from plm_benchmark.config import CHECKPOINT_POLICY, LR_FULL_FT, LR_LORA, MAX_SEQ_LENGTH, OUTPUTS_DIR
from plm_benchmark.models.esm import set_gpu
from plm_benchmark.models.ppi import load_ppi_model, tokenize_ppi_pairs
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_ppi_splits


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _ppi_dataset(df) -> Dataset:
    return Dataset.from_dict(
        {
            "sequence_a": list(df["sequence_a"]),
            "sequence_b": list(df["sequence_b"]),
            "labels": list(df["label"].astype(int)),
        }
    )


def _collate_ppi(batch, tokenizer, max_length: int):
    seq_a = [x["sequence_a"] for x in batch]
    seq_b = [x["sequence_b"] for x in batch]
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    tok = tokenize_ppi_pairs(tokenizer, seq_a, seq_b, max_length=max_length)
    tok["labels"] = labels
    return tok


def _metrics_fn():
    def compute(eval_pred):
        preds, labels = eval_pred
        return {"accuracy": accuracy_score(labels, np.argmax(preds, axis=1))}

    return compute


@torch.no_grad()
def _test_ppi(model, tokenizer, test_df, spec: TaskSpec, batch_size: int = 8) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    labels_all, preds_all = [], []

    n = len(test_df)
    for start in tqdm(range(0, n, batch_size), desc="test"):
        chunk = test_df.iloc[start : start + batch_size]
        batch = [
            {"sequence_a": r.sequence_a, "sequence_b": r.sequence_b, "labels": int(r.label)}
            for r in chunk.itertuples()
        ]
        tok = _collate_ppi(batch, tokenizer, MAX_SEQ_LENGTH)
        tok = {k: v.to(device) if hasattr(v, "to") else v for k, v in tok.items()}
        logits = model(**tok).logits
        preds_all.extend(logits.argmax(-1).cpu().tolist())
        labels_all.extend(tok["labels"].cpu().tolist())

    return float(accuracy_score(labels_all, preds_all))


def run_ppi_finetune(
    spec: TaskSpec,
    checkpoint: str,
    model_name: str,
    train_df,
    valid_df,
    test_df,
    *,
    method: str,
    gpu: int,
    epochs: int,
    batch: int,
    accum: int,
    lr: float | None,
    seed: int,
    fp16: bool,
    val_batch: int = 8,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict:
    if method not in ("full_ft", "lora"):
        raise ValueError(f"PPI supports full_ft or lora, got {method}")

    lr = lr or (LR_FULL_FT if method == "full_ft" else LR_LORA)
    set_gpu(gpu)
    _set_seeds(seed)

    train_df, valid_df, test_df = preprocess_ppi_splits(train_df, valid_df, test_df)

    run_dir = OUTPUTS_DIR / model_name / spec.name.lower() / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {spec.name} (PPI) | {method} | seed={seed} | {checkpoint} ===", flush=True)
    print(
        f"Pairs train/valid: {len(train_df)}/{len(valid_df)} | epochs={epochs} lr={lr}",
        flush=True,
    )

    model, tokenizer = load_ppi_model(checkpoint, spec.num_labels, method=method)
    train_ds = _ppi_dataset(train_df)
    valid_ds = _ppi_dataset(valid_df)

    collator = lambda batch: _collate_ppi(batch, tokenizer, max_length)

    import inspect

    ta_kwargs = dict(
        output_dir=str(run_dir / "hf_cache"),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
        greater_is_better=True,
        save_total_limit=1,
        learning_rate=lr,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        fp16=fp16,
        report_to="none",
        remove_unused_columns=False,
    )
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")

    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=_metrics_fn(),
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    weights = run_dir / "finetuned_weights.pth"
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, weights)

    test = _test_ppi(model, tokenizer, test_df, spec, batch_size=val_batch)
    val = max((x["eval_accuracy"] for x in trainer.state.log_history if "eval_accuracy" in x), default=None)

    print(f"Test accuracy: {test:.4f} | best val: {val}", flush=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": spec.name,
        "model": model_name,
        "checkpoint": checkpoint,
        "method": method,
        "metric": "accuracy",
        "test_score": round(test, 6),
        "val_score": round(val, 6) if val is not None else "",
        "epochs": epochs,
        "lr": lr,
        "batch": batch,
        "seed": seed,
        "checkpoint_policy": CHECKPOINT_POLICY,
        "run_dir": str(run_dir.relative_to(OUTPUTS_DIR.parent)),
    }
    append_experiment(row)

    del model, tokenizer
    torch.cuda.empty_cache()
    return row
