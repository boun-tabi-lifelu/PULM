"""Unified PyTorch downstream pipeline for every Rost + PETA task.

One HuggingFace-Trainer loop (attention1d pooling + linear head) serves all
tasks and all modes (full_ft, embed_head=frozen encoder, lora, and the
from-scratch tokenizer baseline). Recipe (lr / epochs / weight-decay / patience)
comes from the task spec, so Rost and PETA differ only by configuration.
"""

from __future__ import annotations

import dataclasses
import inspect
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments, set_seed

from plm_benchmark.config import (
    CHECKPOINT_POLICY,
    EARLY_STOPPING_PATIENCE,
    LR_FULL_FT,
    LR_HEAD,
    LR_LORA,
    LR_SCRATCH,
    MAX_EPOCHS,
    MAX_SEQ_LENGTH,
    OUTPUTS_DIR,
    WEIGHT_DECAY,
)
from plm_benchmark.methods.data import collate_ppi, collate_single, ppi_dataset, single_dataset
from plm_benchmark.methods.metrics import metrics_fn, score_predictions
from plm_benchmark.models import set_gpu
from plm_benchmark.models.peta_classifier import build_model
from plm_benchmark.results import append_experiment
from plm_benchmark.tasks import TaskSpec, preprocess_ppi_splits, preprocess_sequences
from plm_benchmark.tokenizers import resolve_tokenizer

METHODS = {"full_ft", "full_ft_peta20", "lora", "embed_head"}


@dataclass
class WandbConfig:
    project: str | None = None
    run_name: str | None = None
    entity: str | None = None
    group: str | None = None
    mode: str = "online"
    run_id: str | None = None
    resume: str = "allow"


def make_training_args(kwargs: dict) -> TrainingArguments:
    """Build TrainingArguments across transformers versions.

    Field names drift between releases (e.g. the boolean ``group_by_length``
    became ``train_sampling_strategy="group_by_length"`` in v5, and
    ``evaluation_strategy`` became ``eval_strategy``). Map known renames, then
    drop anything the installed version doesn't accept.
    """
    valid = {f.name for f in dataclasses.fields(TrainingArguments) if f.init}
    kwargs = dict(kwargs)

    # group_by_length (bool, <=v4)  <->  train_sampling_strategy (str, >=v5)
    if "group_by_length" in kwargs and "group_by_length" not in valid:
        grouped = kwargs.pop("group_by_length")
        if "train_sampling_strategy" in valid:
            kwargs["train_sampling_strategy"] = "group_by_length" if grouped else "random"

    # eval_strategy (>=4.41)  <->  evaluation_strategy (older)
    if "eval_strategy" in kwargs and "eval_strategy" not in valid and "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    dropped = [k for k in list(kwargs) if k not in valid]
    for k in dropped:
        print(f"  (dropping TrainingArguments kwarg unsupported by this transformers version: {k})", flush=True)
        kwargs.pop(k)
    return TrainingArguments(**kwargs)


def _set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)


def _resolve_epochs(method: str, epochs: int | None) -> int:
    if epochs:
        return epochs
    if method == "full_ft_peta20":
        return 20
    return MAX_EPOCHS


def _resolve_lr(model_cfg, method: str, lr: float | None) -> float:
    """LR by training regime, not by task source."""
    if lr is not None:
        return lr
    if model_cfg.backend == "scratch":
        return LR_SCRATCH  # random init: no pretrained weights to preserve
    if method == "embed_head":
        return LR_HEAD  # frozen encoder, train head only
    if method == "lora":
        return LR_LORA
    return LR_FULL_FT  # full_ft / full_ft_peta20 on a pretrained encoder


def _setup_wandb(cfg: WandbConfig | None, default_run_name: str, resume_from_checkpoint) -> tuple[list[str], str | None]:
    """Wire env vars for a single run. Returns (report_to, run_name)."""
    if cfg is None or not cfg.project:
        return ["none"], None

    os.environ["WANDB_PROJECT"] = cfg.project
    os.environ["WANDB_MODE"] = cfg.mode
    if cfg.entity:
        os.environ["WANDB_ENTITY"] = cfg.entity
    if cfg.group:
        os.environ["WANDB_RUN_GROUP"] = cfg.group

    # Continue THE SAME run only when an id is given (resume/merge). Otherwise a
    # fresh run per (task, seed) — clear any stale id so the loop never collides.
    if cfg.run_id:
        os.environ["WANDB_RUN_ID"] = cfg.run_id
        os.environ["WANDB_RESUME"] = cfg.resume
    else:
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.pop("WANDB_RESUME", None)
        if resume_from_checkpoint:
            print(
                "WARNING: resuming from checkpoint without --wandb_run_id; wandb will start a NEW "
                "run. Pass --wandb_run_id <id> to append to the original run.",
                flush=True,
            )
    return ["wandb"], (cfg.run_name or default_run_name)


def _finish_wandb(report_to: list[str]) -> None:
    if "wandb" in report_to:
        try:
            import wandb

            if wandb.run is not None:
                wandb.finish()
        except Exception:  # pragma: no cover - wandb optional
            pass


@torch.no_grad()
def _test_score(model, tokenizer, test_df, spec, collator, *, batch_size: int) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if spec.task_type == "ppi":
        ds = ppi_dataset(test_df)
    else:
        ds = single_dataset(test_df, spec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)

    preds, labels = [], []
    for batch in tqdm(loader, desc="test"):
        moved = {k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()}
        logits = model(**moved).logits
        preds.append(logits.cpu().numpy())
        labels.append(moved["labels"].cpu().numpy())
    return score_predictions(np.concatenate(preds, axis=0), np.concatenate(labels, axis=0), spec)


def _best_val(history, metric_key: str, greater_is_better: bool):
    vals = [x[metric_key] for x in history if metric_key in x]
    if not vals:
        return None
    return max(vals) if greater_is_better else min(vals)


def run_downstream(
    spec: TaskSpec,
    model_cfg,
    *,
    method: str,
    split: str | None,
    gpu: int,
    epochs: int | None,
    batch: int,
    accum: int,
    lr: float | None,
    seed: int,
    fp16: bool,
    max_length: int = MAX_SEQ_LENGTH,
    val_batch: int = 64,
    eval_every: int = 1,
    patience: int | None = None,
    tokenizer_spec: str | None = None,
    scratch_dim: int = 320,
    scratch_layers: int = 0,
    scratch_heads: int = 8,
    num_workers: int = 4,
    wandb_cfg: WandbConfig | None = None,
    resume_from_checkpoint: str | None = None,
    train_df=None,
    valid_df=None,
    test_df=None,
) -> dict:
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}. Choose from {sorted(METHODS)}.")
    if method == "lora" and model_cfg.backend == "scratch":
        raise ValueError("lora is not applicable to the scratch baseline (no attention modules).")

    freeze = method == "embed_head"
    use_lora = method == "lora"
    is_ppi = spec.task_type == "ppi"

    epochs = _resolve_epochs(method, epochs)
    lr = _resolve_lr(model_cfg, method, lr)
    weight_decay = WEIGHT_DECAY
    stop_patience = EARLY_STOPPING_PATIENCE if patience is None else patience
    metric_for_best = f"eval_{spec.metric}"

    set_gpu(gpu)
    _set_seeds(seed)

    if is_ppi:
        train_df, valid_df, test_df = preprocess_ppi_splits(train_df, valid_df, test_df)
        train_ds, valid_ds = ppi_dataset(train_df), ppi_dataset(valid_df)
    else:
        train_df = preprocess_sequences(train_df, task_type=spec.task_type)
        valid_df = preprocess_sequences(valid_df, task_type=spec.task_type)
        test_df = preprocess_sequences(test_df, task_type=spec.task_type)
        train_ds, valid_ds = single_dataset(train_df, spec), single_dataset(valid_df, spec)

    split_tag = split or "default"
    run_dir = OUTPUTS_DIR / model_cfg.name / spec.name.lower() / split_tag / method / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    default_run_name = f"{model_cfg.name}/{spec.name}/{split_tag}/{method}/seed{seed}"
    report_to, run_name = _setup_wandb(wandb_cfg, default_run_name, resume_from_checkpoint)

    mode = "frozen-encoder" if freeze else ("lora" if use_lora else "full")
    print(
        f"\n=== {spec.name} | {method} ({mode}) | split={split_tag} | seed={seed} "
        f"| {model_cfg.name} [{model_cfg.backend}] ===",
        flush=True,
    )
    print(
        f"Train/valid: {len(train_df)}/{len(valid_df)} | epochs={epochs} lr={lr} wd={weight_decay} "
        f"patience={stop_patience} eval_every={eval_every}ep | head=attention1d | metric={spec.metric}",
        flush=True,
    )

    tokenizer_obj = resolve_tokenizer(tokenizer_spec, max_length=max_length) if tokenizer_spec else None
    model, tokenizer = build_model(
        model_cfg,
        spec,
        tokenizer=tokenizer_obj,
        freeze=freeze,
        use_lora=use_lora,
        scratch_dim=scratch_dim,
        scratch_layers=scratch_layers,
        scratch_heads=scratch_heads,
    )
    if is_ppi:
        collator = partial(collate_ppi, tokenizer=tokenizer, max_length=max_length)
    else:
        collator = partial(collate_single, tokenizer=tokenizer, max_length=max_length, spec=spec)

    # Eval/checkpoint cadence: every epoch by default (unchanged behaviour). When
    # eval_every > 1, switch to a steps schedule so we eval + checkpoint every
    # eval_every epochs (fewer large-val forward passes + fewer checkpoint writes).
    if eval_every > 1:
        steps_per_epoch = max(1, math.ceil(len(train_ds) / (batch * accum)))
        total_steps = steps_per_epoch * epochs
        # Ensure at least one eval happens, else load_best_model_at_end has nothing.
        cadence_steps = min(eval_every * steps_per_epoch, total_steps)
        cadence = dict(
            eval_strategy="steps",
            save_strategy="steps",
            logging_strategy="steps",
            eval_steps=cadence_steps,
            save_steps=cadence_steps,
            logging_steps=cadence_steps,
        )
    else:
        cadence = dict(eval_strategy="epoch", save_strategy="epoch", logging_strategy="epoch")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not fp16
    ta_kwargs = dict(
        output_dir=str(run_dir / "hf_cache"),
        **cadence,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=spec.greater_is_better,
        save_total_limit=1,
        learning_rate=lr,
        weight_decay=weight_decay,
        per_device_train_batch_size=batch,
        per_device_eval_batch_size=val_batch,
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,
        seed=seed,
        fp16=fp16 and not use_bf16,
        bf16=use_bf16,
        group_by_length=True,
        length_column_name="length",
        dataloader_num_workers=num_workers,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=report_to,
        run_name=run_name,
    )

    callbacks = []
    if stop_patience and stop_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=stop_patience))

    trainer_kwargs = dict(
        model=model,
        args=make_training_args(ta_kwargs),
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=collator,
        compute_metrics=metrics_fn(spec),
        callbacks=callbacks,
    )
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    weights = run_dir / "finetuned_weights.pth"
    torch.save({n: p for n, p in model.named_parameters() if p.requires_grad}, weights)

    test = _test_score(model, tokenizer, test_df, spec, collator, batch_size=val_batch)
    val = _best_val(trainer.state.log_history, metric_for_best, spec.greater_is_better)
    print(f"Test {spec.metric}: {test:.4f} | best val: {val}", flush=True)

    if "wandb" in report_to:
        trainer.log({f"test/{spec.metric}": test})
    _finish_wandb(report_to)

    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": spec.name,
        "model": model_cfg.name,
        "checkpoint": model_cfg.checkpoint,
        "method": method,
        "split": split_tag,
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

    del model, tokenizer, trainer
    torch.cuda.empty_cache()
    return row
