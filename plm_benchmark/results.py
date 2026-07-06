"""Experiment logging and comparison tables."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from plm_benchmark.config import CHECKPOINT_POLICY, COMPARE_CSV, LOG_CSV

EXPERIMENT_FIELDS = [
    "timestamp",
    "task",
    "model",
    "checkpoint",
    "method",
    "metric",
    "test_score",
    "val_score",
    "epochs",
    "lr",
    "batch",
    "seed",
    "checkpoint_policy",
    "run_dir",
    "error",
]


def append_experiment(row: dict, path: Path = LOG_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    normalized = {k: row.get(k, "") for k in EXPERIMENT_FIELDS}
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(normalized)


def load_experiments(path: Path = LOG_CSV, *, keep_all: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    ok = df["test_score"].notna() & (df["test_score"].astype(str) != "")
    df = df[ok]
    if keep_all:
        return df.sort_values("timestamp")
    subset = ["task", "model", "method", "seed"]
    if "seed" not in df.columns:
        subset = ["task", "model", "method"]
    return df.sort_values("timestamp").drop_duplicates(subset=subset, keep="last")


def build_comparison(log_path: Path = LOG_CSV, out_path: Path = COMPARE_CSV) -> pd.DataFrame:
    df = load_experiments(log_path, keep_all=True)
    if df.empty:
        raise FileNotFoundError(f"No experiments in {log_path}")

    if "checkpoint_policy" in df.columns:
        best = df[df["checkpoint_policy"] == CHECKPOINT_POLICY]
        if not best.empty:
            df = best

    df = df.copy()
    df["test_score"] = pd.to_numeric(df["test_score"], errors="coerce")
    df["val_score"] = pd.to_numeric(df["val_score"], errors="coerce")

    rows = []
    for (task, model, method), sub in df.groupby(["task", "model", "method"], sort=True):
        metric = sub["metric"].iloc[0]
        scores = sub["test_score"].dropna()
        vals = sub["val_score"].dropna()
        rows.append(
            {
                "task": task,
                "model": model,
                "method": method,
                "metric": metric,
                "n_seeds": len(scores),
                "test_mean": round(scores.mean(), 6) if len(scores) else "",
                "test_std": round(scores.std(ddof=0), 6) if len(scores) > 1 else 0.0,
                "val_mean": round(vals.mean(), 6) if len(vals) else "",
            }
        )

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
