
from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from evaluate import load
from scipy import stats
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from plm_benchmark.config import CHECKPOINT_POLICY, LR_FULL_FT, LR_LORA, OUTPUTS_DIR
from plm_benchmark.models.esm import load_classifier, set_gpu, tokenize_dataset
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_sequences


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _metrics_fn(spec: TaskSpec):
    def compute(eval_pred):
        preds, labels = eval_pred
        if spec.task_type == "classification":
            m = load("accuracy")
            preds = np.argmax(preds, axis=1)
        else:
            m = load("spearmanr")
            preds = np.squeeze(preds).astype(float)
            labels = np.asarray(labels, dtype=float)
        return m.compute(predictions=preds, references=labels)

    return compute


@torch.no_grad()
def _test_score(model, tokenizer, test_df, spec: TaskSpec, batch_size: int = 16) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ds = tokenize_dataset(tokenizer, list(test_df["sequence"]), list(test_df["label"]))
    ds = ds.with_format("torch", device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, labels = [], test_df["label"].tolist()

    for batch in tqdm(loader, desc="test"):
        logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        if spec.task_type == "classification":
            preds.extend(logits.argmax(-1).cpu().tolist())
        else:
            preds.extend(logits.squeeze(-1).float().cpu().tolist())

    if spec.task_type == "classification":
        return float(accuracy_score(labels, preds))
    return float(stats.spearmanr(preds, labels).correlation)


def run_finetune(
    spec: TaskSpec,
    checkpoint: str,
    model_name: str,
    train_df,
    valid_df,
    test_df,
    *,
    method: str,  # full_ft | lora
    gpu: int,
    epochs: int,
    batch: int,
    accum: int,
    lr: float | None,
    seed: int,
    fp16: bool,
    val_batch: int = 16,
) -> dict:
    if method not in ("full_ft", "lora"):
        raise ValueError(f"finetune method must be full_ft or lora, got {method}")

    lr = lr or (LR_FULL_FT if method == "full_ft" else LR_LORA)
    set_gpu(gpu)
    _set_seeds(seed)

    train_df = preprocess_sequences(train_df)
    valid_df = preprocess_sequences(valid_df)
    test_df = preprocess_sequences(test_df)

    run_dir = OUTPUTS_DIR / model_name / spec.name.lower() / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_for_best = f"eval_{spec.metric}"

    print(f"\n=== {spec.name} | {method} | seed={seed} | {checkpoint} ===", flush=True)
    print(
        f"Train/valid: {len(train_df)}/{len(valid_df)} | epochs={epochs} lr={lr} "
        f"| checkpoint={CHECKPOINT_POLICY}",
        flush=True,
    )

    model, tokenizer = load_classifier(checkpoint, spec.num_labels, method=method)
    train_ds = tokenize_dataset(tokenizer, list(train_df["sequence"]), list(train_df["label"]))
    valid_ds = tokenize_dataset(tokenizer, list(valid_df["sequence"]), list(valid_df["label"]))

    import inspect

    ta_kwargs = dict(
        output_dir=str(run_dir / "hf_cache"),
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
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
    )
    if "evaluation_strategy" in inspect.signature(TrainingArguments.__init__).parameters:
        ta_kwargs["evaluation_strategy"] = ta_kwargs.pop("eval_strategy")

    trainer_kwargs = dict(
        model=model,
        args=TrainingArguments(**ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        compute_metrics=_metrics_fn(spec),
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train()

    weights = run_dir / "finetuned_weights.pth"
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, weights)

    test = _test_score(model, tokenizer, test_df, spec)
    key = metric_for_best
    val = max((x[key] for x in trainer.state.log_history if key in x), default=None)

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
