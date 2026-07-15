
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plm_benchmark.config import DATA_DIR, RARE_AA

PER_PROTEIN_TASKS = ["GB1", "AAV", "GFP", "Meltome", "Stab", "SubLoc"]


@dataclass(frozen=True)
class TaskSpec:
    """Task identity only. The training recipe (lr, epochs, patience, weight
    decay) is uniform across all tasks — see config.py and methods/train.py."""

    name: str
    folder: str
    seq_col: str
    label_col: str
    task_type: str  # regression | classification | multilabel | ppi
    num_labels: int
    metric: str  # spearmanr | accuracy | mse
    greater_is_better: bool = True
    data_source: str = "rost"  # rost | peta
    peta_key: str | None = None
    default_split: str | None = None


TASKS: dict[str, TaskSpec] = {
    "GB1": TaskSpec("GB1", "GB1", "primary", "gb1_score", "regression", 1, "spearmanr"),
    "AAV": TaskSpec("AAV", "AAV", "primary", "aav_score", "regression", 1, "spearmanr"),
    "GFP": TaskSpec("GFP", "GFP", "primary", "fluor_score", "regression", 1, "spearmanr"),
    "Meltome": TaskSpec("Meltome", "Meltome", "primary", "thermo_score", "regression", 1, "spearmanr"),
    "Stab": TaskSpec("Stab", "Stab", "primary", "stability_score", "regression", 1, "spearmanr"),
    "SubLoc": TaskSpec("SubLoc", "SubLoc", "Sequence", "loc_num", "classification", 10, "accuracy"),
}


def _merge_peta_tasks() -> None:
    from plm_benchmark.peta_tasks import PETA_TASKS

    TASKS.update(PETA_TASKS)


_merge_peta_tasks()

PETA_TASK_NAMES: list[str] = [n for n in TASKS if n.startswith("peta_")]


def all_task_names() -> list[str]:
    return PER_PROTEIN_TASKS + PETA_TASK_NAMES


def resolve_tasks(task_arg: str) -> list[str]:
    arg = task_arg.lower()
    if arg == "all":
        return PER_PROTEIN_TASKS.copy()
    if arg == "peta_all":
        return PETA_TASK_NAMES.copy()
    if arg == "everything":
        return all_task_names()

    names = [t.strip() for t in task_arg.split(",")]
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(
            f"Unknown task(s): {unknown}. "
            f"Rost: {PER_PROTEIN_TASKS}. PETA: use peta_* or peta_all. Run: python -m plm_benchmark.cli list-tasks"
        )
    return names


ALL_SPLITS = "all"  # --split-method sentinel: run every curated split of the task(s)


def run_splits_for(name: str) -> list[str | None]:
    """Curated splits to enumerate for a task (default first). Single-split tasks -> [split]."""
    from plm_benchmark.peta_data import PETA_RUN_SPLITS

    spec = TASKS[name]
    if spec.peta_key and spec.peta_key in PETA_RUN_SPLITS:
        return list(PETA_RUN_SPLITS[spec.peta_key])
    return [spec.default_split]  # may be None (Rost / single-split PETA)


def _validate_split(name: str, split: str) -> None:
    from plm_benchmark.peta_data import PETA_SPLIT_OPTIONS

    spec = TASKS[name]
    if spec.peta_key and spec.peta_key in PETA_SPLIT_OPTIONS:
        options = PETA_SPLIT_OPTIONS[spec.peta_key]
        if split not in options:
            raise ValueError(f"Unknown split {split!r} for {name}. Options: {options}")


def resolve_task_splits(task_arg: str, split_method: str | None = None) -> list[tuple[str, str | None]]:
    """Expand a task selection into (task, split) pairs.

    - group (all/peta_all/everything) -> every task x its curated run-splits (split_method ignored)
    - --split-method all              -> named task(s) x their curated run-splits
    - --split-method X                -> named task(s) at split X (validated)
    - no --split-method               -> named task(s) at default split only (backward compatible)
    """
    is_group = task_arg.lower() in ("all", "peta_all", "everything")
    names = resolve_tasks(task_arg)

    if is_group and split_method and split_method != ALL_SPLITS:
        print(f"WARNING: --split-method {split_method!r} ignored for group '{task_arg}'; "
              "running each task's curated splits.", flush=True)

    pairs: list[tuple[str, str | None]] = []
    for name in names:
        if is_group or split_method == ALL_SPLITS:
            splits = run_splits_for(name)
        elif split_method:
            _validate_split(name, split_method)
            splits = [split_method]
        else:
            splits = [TASKS[name].default_split]
        for s in splits:
            if (name, s) not in pairs:
                pairs.append((name, s))
    return pairs


def preprocess_sequences(df: pd.DataFrame, *, task_type: str = "regression") -> pd.DataFrame:
    if task_type == "ppi":
        return df.copy()
    out = df.copy()
    out["sequence"] = out["sequence"].astype(str).str.replace("|".join(RARE_AA), "X", regex=True)
    return out


def preprocess_ppi_splits(train_df, valid_df, test_df):
    """Rare-AA cleanup on both chains of each protein pair."""

    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col in ("sequence_a", "sequence_b"):
            out[col] = out[col].astype(str).str.replace("|".join(RARE_AA), "X", regex=True)
        return out

    return _clean(train_df), _clean(valid_df), _clean(test_df)


def load_splits(
    spec: TaskSpec,
    data_dir: Path | None = None,
    *,
    split_method: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if spec.data_source == "peta":
        from plm_benchmark.peta_data import load_peta_splits

        split = split_method or spec.default_split
        return load_peta_splits(spec.peta_key, split_method=split, data_dir=data_dir)

    root = (data_dir or DATA_DIR) / spec.folder
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing {root}. Setup data:\n"
            f"  clone https://github.com/RSchmirler/data-repo_plm-finetune-eval"
        )

    def _load(split: str) -> pd.DataFrame:
        df = pd.read_pickle(root / f"{split}.pkl")
        out = pd.DataFrame({"sequence": df[spec.seq_col].astype(str), "label": df[spec.label_col]})
        if spec.task_type == "classification":
            out["label"] = out["label"].astype(int)
        else:
            out["label"] = out["label"].astype(float)
        return out.reset_index(drop=True)

    return _load("train"), _load("valid"), _load("test")
