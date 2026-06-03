
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
    task_type: str  # regression | classification
    num_labels: int
    finetune_epochs: int
    embed_head_epochs: int
    metric: str  # spearmanr | accuracy


TASKS: dict[str, TaskSpec] = {
    "GB1": TaskSpec("GB1", "GB1", "primary", "gb1_score", "regression", 1, 20, 240, "spearmanr"),
    "AAV": TaskSpec("AAV", "AAV", "primary", "aav_score", "regression", 1, 10, 120, "spearmanr"),
    "GFP": TaskSpec("GFP", "GFP", "primary", "fluor_score", "regression", 1, 20, 240, "spearmanr"),
    "Meltome": TaskSpec("Meltome", "Meltome", "primary", "thermo_score", "regression", 1, 10, 120, "spearmanr"),
    "Stab": TaskSpec("Stab", "Stab", "primary", "stability_score", "regression", 1, 10, 120, "spearmanr"),
    "SubLoc": TaskSpec("SubLoc", "SubLoc", "Sequence", "loc_num", "classification", 10, 10, 120, "accuracy"),
}


def resolve_tasks(task_arg: str) -> list[str]:
    if task_arg.lower() == "all":
        return PER_PROTEIN_TASKS.copy()
    names = [t.strip() for t in task_arg.split(",")]
    unknown = [t for t in names if t not in TASKS]
    if unknown:
        raise ValueError(f"Unknown task(s): {unknown}. Choose from {PER_PROTEIN_TASKS}")
    return names


def preprocess_sequences(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sequence"] = out["sequence"].str.replace("|".join(RARE_AA), "X", regex=True)
    return out


def load_splits(spec: TaskSpec, data_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = (data_dir or DATA_DIR) / spec.folder
    if not root.is_dir():
        raise FileNotFoundError(
            f"Missing {root}. Setup data:\n"
            f"  unzip -q 'training data.zip'"
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
