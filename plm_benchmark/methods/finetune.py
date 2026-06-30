
from __future__ import annotations

import random
from datetime import datetime, timezone

import numpy as np
import torch
from evaluate import load
from scipy import stats
from sklearn.metrics import accuracy_score, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Trainer, TrainingArguments, set_seed

from plm_benchmark.config import CHECKPOINT_POLICY, LR_FULL_FT, LR_LORA, OUTPUTS_DIR
from plm_benchmark.models.esm import load_classifier, set_gpu, tokenize_dataset
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_sequences


def _problem_type(spec: TaskSpec) -> str | None:
    if spec.task_type == "multilabel":
        return "multi_label_classification"
    if spec.task_type == "regression":
        return "regression"
    if spec.task_type == "classification":
        return "single_label_classification"
    return None


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _score_predictions(preds: np.ndarray, labels: np.ndarray, spec: TaskSpec) -> float:
    if spec.task_type in ("classification", "ppi"):
        return float(accuracy_score(labels, np.argmax(preds, axis=1)))
    if spec.task_type == "multilabel":
        probs = 1 / (1 + np.exp(-preds))
        pred_bin = (probs > 0.5).astype(int)
        labels_bin = np.asarray(labels, dtype=int)
        return float((pred_bin == labels_bin).all(axis=1).mean())
    if spec.metric == "mse":
        return float(mean_squared_error(labels, np.squeeze(preds).astype(float)))
    return float(stats.spearmanr(np.squeeze(preds).astype(float), np.asarray(labels, dtype=float)).correlation)


def _metrics_fn(spec: TaskSpec):
    def compute(eval_pred):
        preds, labels = eval_pred
        score = _score_predictions(preds, labels, spec)
        return {spec.metric: score}

    return compute


@torch.no_grad()
def _test_score(model, tokenizer, test_df, spec: TaskSpec, batch_size: int = 16) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    ds = tokenize_dataset(tokenizer, list(test_df["sequence"]), list(test_df["label"]))
    ds = ds.with_format("torch", device=device)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    preds, labels = [], []

    for batch in tqdm(loader, desc="test"):
        logits = model(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device)).logits
        preds.append(logits.cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(preds, axis=0)
    labels = np.concatenate(labels, axis=0)
    return _score_predictions(preds, labels, spec)


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

    train_df = preprocess_sequences(train_df, task_type=spec.task_type)
    valid_df = preprocess_sequences(valid_df, task_type=spec.task_type)
    test_df = preprocess_sequences(test_df, task_type=spec.task_type)

    run_dir = OUTPUTS_DIR / model_name / spec.name.lower() / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metric_for_best = f"eval_{spec.metric}"

    print(f"\n=== {spec.name} | {method} | seed={seed} | {checkpoint} ===", flush=True)
    print(
        f"Train/valid: {len(train_df)}/{len(valid_df)} | epochs={epochs} lr={lr} "
        f"| metric={spec.metric} | checkpoint={CHECKPOINT_POLICY}",
        flush=True,
    )

    model, tokenizer = load_classifier(
        checkpoint, spec.num_labels, method=method, problem_type=_problem_type(spec)
    )
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
        greater_is_better=spec.greater_is_better,
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
    history = [x[key] for x in trainer.state.log_history if key in x]
    val = max(history) if history and spec.greater_is_better else (min(history) if history else None)

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
