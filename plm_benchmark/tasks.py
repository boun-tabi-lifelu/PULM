
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from plm_benchmark.config import DATA_DIR, RARE_AA

PER_PROTEIN_TASKS = ["GB1", "AAV", "GFP", "Meltome", "Stab", "SubLoc"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    folder: str
    seq_col: str
    label_col: str
    task_type: str  # regression | classification | multilabel | ppi
    num_labels: int
    finetune_epochs: int
    embed_head_epochs: int
    metric: str  # spearmanr | accuracy | mse
    greater_is_better: bool = True
    data_source: str = "rost"  # rost | peta
    peta_key: str | None = None
    default_split: str | None = None


TASKS: dict[str, TaskSpec] = {
    "GB1": TaskSpec("GB1", "GB1", "primary", "gb1_score", "regression", 1, 20, 240, "spearmanr"),
    "AAV": TaskSpec("AAV", "AAV", "primary", "aav_score", "regression", 1, 10, 120, "spearmanr"),
    "GFP": TaskSpec("GFP", "GFP", "primary", "fluor_score", "regression", 1, 20, 240, "spearmanr"),
    "Meltome": TaskSpec("Meltome", "Meltome", "primary", "thermo_score", "regression", 1, 10, 120, "spearmanr"),
    "Stab": TaskSpec("Stab", "Stab", "primary", "stability_score", "regression", 1, 10, 120, "spearmanr"),
    "SubLoc": TaskSpec("SubLoc", "SubLoc", "Sequence", "loc_num", "classification", 10, 10, 120, "accuracy"),
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
            f"Rost: {PER_PROTEIN_TASKS}. PETA: use peta_* or peta_all. Run: python run.py list-tasks"
        )
    return names


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
