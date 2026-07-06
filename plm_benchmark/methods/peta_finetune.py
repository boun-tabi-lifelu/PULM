
from __future__ import annotations

import inspect
import random
from datetime import datetime, timezone

import numpy as np
import torch
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from plm_benchmark.config import CHECKPOINT_POLICY, MAX_SEQ_LENGTH, OUTPUTS_DIR
from plm_benchmark.methods.finetune import _metrics_fn, _score_predictions
from plm_benchmark.models.esm import set_gpu
from plm_benchmark.models.peta_classifier import load_peta_model
from plm_benchmark.models.ppi import tokenize_ppi_pairs
from plm_benchmark.peta_protocol import PETA_MAX_EPOCHS, peta_settings
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_ppi_splits, preprocess_sequences


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _problem_type(spec: TaskSpec) -> str:
    if spec.task_type == "ppi":
        return "classification"
    return spec.task_type


def _label_dtype(spec: TaskSpec) -> torch.dtype:
    if spec.task_type in ("classification", "ppi"):
        return torch.long
    if spec.task_type == "multilabel":
        return torch.float
    return torch.float


def _single_dataset(df, spec: TaskSpec) -> Dataset:
    labels = list(df["label"])
    if spec.task_type == "classification" or spec.task_type == "ppi":
        labels = [int(x) for x in labels]
    elif spec.task_type == "regression":
        labels = [float(x) for x in labels]
    return Dataset.from_dict({"sequence": list(df["sequence"]), "labels": labels})


def _ppi_dataset(df) -> Dataset:
    return Dataset.from_dict(
        {
            "sequence_a": list(df["sequence_a"]),
            "sequence_b": list(df["sequence_b"]),
            "labels": list(df["label"].astype(int)),
        }
    )


def _collate_single(batch, tokenizer, max_length: int, spec: TaskSpec):
    seqs = [x["sequence"] for x in batch]
    dtype = _label_dtype(spec)
    labels = torch.tensor([x["labels"] for x in batch], dtype=dtype)
    tok = tokenizer(
        seqs, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    tok["labels"] = labels
    return tok


def _collate_ppi(batch, tokenizer, max_length: int):
    seq_a = [x["sequence_a"] for x in batch]
    seq_b = [x["sequence_b"] for x in batch]
    labels = torch.tensor([x["labels"] for x in batch], dtype=torch.long)
    tok = tokenize_ppi_pairs(tokenizer, seq_a, seq_b, max_length=max_length)
    tok["labels"] = labels
    return tok


@torch.no_grad()
def _test_score(model, tokenizer, test_df, spec: TaskSpec, *, batch_size: int, max_length: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if spec.task_type == "ppi":
        labels_all, preds_all = [], []
        n = len(test_df)
        for start in tqdm(range(0, n, batch_size), desc="test"):
            chunk = test_df.iloc[start : start + batch_size]
            batch = [
                {"sequence_a": r.sequence_a, "sequence_b": r.sequence_b, "labels": int(r.label)}
                for r in chunk.itertuples()
            ]
            tok = _collate_ppi(batch, tokenizer, max_length)
            tok = {k: v.to(device) if hasattr(v, "to") else v for k, v in tok.items()}
            logits = model(**tok).logits
            preds_all.append(logits.cpu().numpy())
            labels_all.append(tok["labels"].cpu().numpy())
        return _score_predictions(
            np.concatenate(preds_all, axis=0), np.concatenate(labels_all, axis=0), spec
        )

    from plm_benchmark.models.esm import tokenize_dataset

    ds = tokenize_dataset(tokenizer, list(test_df["sequence"]), list(test_df["label"]))
    ds = ds.with_format("torch", device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, labels = [], []
    for batch in tqdm(loader, desc="test"):
        logits = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        ).logits
        preds.append(logits.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
    return _score_predictions(np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), spec)


def _best_val(history: list[dict], metric_key: str, greater_is_better: bool) -> float | None:
    vals = [x[metric_key] for x in history if metric_key in x]
    if not vals:
        return None
    return max(vals) if greater_is_better else min(vals)


def run_peta_finetune(
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
    patience: int | None = None,
    val_batch: int = 16,
    max_length: int = MAX_SEQ_LENGTH,
) -> dict:
    settings = peta_settings(spec, method)
    lr = lr if lr is not None else settings.lr
    epochs = epochs or settings.max_epochs
    stop_patience = settings.patience if patience is None else patience
    is_ppi = spec.task_type == "ppi"

    set_gpu(gpu)
    _set_seeds(seed)

    if is_ppi:
        train_df, valid_df, test_df = preprocess_ppi_splits(train_df, valid_df, test_df)
        train_ds = _ppi_dataset(train_df)
        valid_ds = _ppi_dataset(valid_df)
    else:
        train_df = preprocess_sequences(train_df, task_type=spec.task_type)
        valid_df = preprocess_sequences(valid_df, task_type=spec.task_type)
        test_df = preprocess_sequences(test_df, task_type=spec.task_type)
        train_ds = _single_dataset(train_df, spec)
        valid_ds = _single_dataset(valid_df, spec)

    run_dir = OUTPUTS_DIR / model_name / spec.name.lower() / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    mode = "all" if settings.train_encoder else "head"
    metric_for_best = f"eval_{spec.metric}"

    print(
        f"\n=== {spec.name} (PETA) | {method} (finetune={mode}) | seed={seed} | {checkpoint} ===",
        flush=True,
    )
    print(
        f"Train/valid: {len(train_df)}/{len(valid_df)} | epochs={epochs} lr={lr} "
        f"wd={settings.weight_decay} patience={stop_patience} "
        f"| pooling=attention1d | metric={spec.metric}",
        flush=True,
    )

    model, tokenizer = load_peta_model(
        checkpoint,
        spec.num_labels,
        is_ppi=is_ppi,
        problem_type=_problem_type(spec),
        train_encoder=settings.train_encoder,
    )

    if is_ppi:
        collator = lambda batch: _collate_ppi(batch, tokenizer, max_length)  # noqa: E731
    else:
        collator = lambda batch: _collate_single(batch, tokenizer, max_length, spec)  # noqa: E731

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not fp16
    ta_kwargs = dict(
        output_dir=str(run_dir / "hf_cache"),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=settings.greater_is_better,
        save_total_limit=1,
        learning_rate=lr,
        weight_decay=settings.weight_decay,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        fp16=fp16 and not use_bf16,
        bf16=use_bf16,
        report_to="none",
        remove_unused_columns=False,
    )
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")

    callbacks = []
    if stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=stop_patience))

    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=_metrics_fn(spec),
        callbacks=callbacks,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    weights = run_dir / "finetuned_weights.pth"
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, weights)

    test = _test_score(model, tokenizer, test_df, spec, batch_size=val_batch, max_length=max_length)
    val = _best_val(trainer.state.log_history, metric_for_best, settings.greater_is_better)

    print(f"Test {spec.metric}: {test:.4f} | best val: {val}", flush=True)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": spec.name,
        "model": model_name,
        "checkpoint": checkpoint,
        "method": method,
        "metric": spec.metric,
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
