
from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from plm_benchmark.config import COMPARE_CSV, LOG_CSV


def append_experiment(row: dict, path: Path = LOG_CSV) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_experiments(path: Path = LOG_CSV) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df = df[df["test_score"].notna() & (df["test_score"].astype(str) != "")]
    return df.sort_values("timestamp").drop_duplicates(
        subset=["task", "model", "method"], keep="last"
    )


def build_comparison(log_path: Path = LOG_CSV, out_path: Path = COMPARE_CSV) -> pd.DataFrame:
    df = load_experiments(log_path)
    if df.empty:
        raise FileNotFoundError(f"No experiments in {log_path}")

    rows = []
    for task in df["task"].unique():
        sub = df[df["task"] == task]
        metric = sub["metric"].iloc[0]
        row = {"task": task, "metric": metric}
        for method in ("full_ft", "lora", "embed_head"):
            m = sub[sub["method"] == method]
            if not m.empty:
                row[f"{method}_test"] = m["test_score"].iloc[0]
                row[f"{method}_val"] = m["val_score"].iloc[0]
        if "full_ft_test" in row and "embed_head_test" in row:
            row["ft_minus_embed"] = row["full_ft_test"] - row["embed_head_test"]
        rows.append(row)

    out = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    return out
