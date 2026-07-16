"""Experiment logging and comparison tables."""

from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import fcntl  # POSIX advisory locking (Linux/macOS)
except ImportError:  # pragma: no cover - non-POSIX
    fcntl = None

from plm_benchmark.config import CHECKPOINT_POLICY, COMPARE_CSV, LOG_CSV

EXPERIMENT_FIELDS = [
    "start_datetime",
    "end_datetime",
    "duration_sec",
    "task",
    "model",
    "tokenizer",
    "vocab_size",
    "method",
    "split",
    "metric",
    "test_score",
    "val_score",
    "epochs",
    "lr",
    "batch",
    "seed",
    "full_name",
    "scratch_dim",
    "scratch_layers",
    "scratch_heads",
    "git_commit",
    "checkpoint_policy",
    "checkpoint",
    "run_dir",
    "error",
]


def append_experiment(row: dict, path: Path = LOG_CSV) -> None:
    """Append one row. Safe under concurrent writers via an exclusive file lock,
    so parallel runs (many processes on one GPU) don't interleave/corrupt rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    normalized = {k: row.get(k, "") for k in EXPERIMENT_FIELDS}
    with path.open("a", newline="") as f:
        if fcntl is not None:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            write_header = f.tell() == 0  # decided while holding the lock
            w = csv.DictWriter(f, fieldnames=EXPERIMENT_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow(normalized)
            f.flush()
            os.fsync(f.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def load_experiments(path: Path = LOG_CSV, *, keep_all: bool = False) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if df.empty:
        return df
    ok = df["test_score"].notna() & (df["test_score"].astype(str) != "")
    df = df[ok]
    sort_col = next((c for c in ("end_datetime", "start_datetime", "timestamp") if c in df.columns), None)
    if sort_col:
        df = df.sort_values(sort_col)
    if keep_all:
        return df
    subset = [c for c in ["task", "model", "tokenizer", "method", "split", "seed"] if c in df.columns]
    return df.drop_duplicates(subset=subset, keep="last")


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

    group_cols = [c for c in ["task", "model", "tokenizer", "method", "split"] if c in df.columns]
    rows = []
    for keys, sub in df.groupby(group_cols, sort=True):
        record = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        metric = sub["metric"].iloc[0]
        scores = sub["test_score"].dropna()
        vals = sub["val_score"].dropna()
        # constant within a (model, tokenizer) group; carried through for readability
        if "vocab_size" in sub.columns:
            record["vocab_size"] = sub["vocab_size"].iloc[0]
        rows.append(
            {
                **record,
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
