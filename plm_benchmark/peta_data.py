"""Load PETA benchmark splits (JSON) with the same preprocessing as ProteinPretraining.

Data: download benchmark_datasets.zip from the PETA repo and unzip to ft_datasets/
https://github.com/mingchen-li/ProteinPretraining
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from plm_benchmark.config import PETA_DATA_DIR


def _read_json_split(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing PETA file: {path}")
    with path.open() as f:
        return json.load(f)


def _rows_to_df(rows: list[dict], *, seq_key: str = "sequence", label_key: str = "label") -> pd.DataFrame:
    sequences, labels = [], []
    for row in rows:
        seq = row[seq_key]
        lab = row[label_key]
        sequences.append(seq)
        labels.append(lab)
    return pd.DataFrame({"sequence": sequences, "label": labels})


def _flatten_scalar_labels(df: pd.DataFrame) -> pd.DataFrame:
    """PETA often stores labels as [value] lists."""

    def _flat(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            if len(x) == 1:
                return x[0]
            return x
        return x

    out = df.copy()
    out["label"] = out["label"].map(_flat)
    return out


def _add_prefix_m(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["sequence"] = "M" + out["sequence"].astype(str)
    return out


def _z_score_labels(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    vals = pd.concat([train["label"], valid["label"], test["label"]]).astype(float)
    mean, std = float(vals.mean()), float(vals.std())
    if std == 0:
        std = 1.0

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label"] = (out["label"].astype(float) - mean) / std
        return out

    return _norm(train), _norm(valid), _norm(test)


def _min_max_labels(train: pd.DataFrame, valid: pd.DataFrame, test: pd.DataFrame):
    vals = pd.concat([train["label"], valid["label"], test["label"]]).astype(float)
    lo, hi = float(vals.min()), float(vals.max())
    span = hi - lo or 1.0

    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["label"] = (out["label"].astype(float) - lo) / span
        return out

    return _norm(train), _norm(valid), _norm(test)


def _tape_fluorescence(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "tape" / "fluorescence"
    train = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "fluorescence_train.json"), seq_key="primary", label_key="log_fluorescence")
    )
    valid = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "fluorescence_valid.json"), seq_key="primary", label_key="log_fluorescence")
    )
    test = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "fluorescence_test.json"), seq_key="primary", label_key="log_fluorescence")
    )
    train, valid, test = map(_add_prefix_m, (train, valid, test))
    return _z_score_labels(train, valid, test)


def _tape_stability(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "tape" / "stability"
    train = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "stability_train.json"), seq_key="primary", label_key="stability_score")
    )
    valid = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "stability_valid.json"), seq_key="primary", label_key="stability_score")
    )
    test = _flatten_scalar_labels(
        _rows_to_df(_read_json_split(base / "stability_test.json"), seq_key="primary", label_key="stability_score")
    )
    train, valid, test = map(_add_prefix_m, (train, valid, test))
    return _z_score_labels(train, valid, test)


def _flip_split(root: Path, dataset: str, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "flip" / dataset / split_method
    train = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "train.json")))
    valid = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "valid.json")))
    test = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "test.json")))
    return train, valid, test


def _flip_meltome(root: Path, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, valid, test = _flip_split(root, "meltome", split_method)
    return _min_max_labels(train, valid, test)


def _flip_gb1(root: Path, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _flip_split(root, "gb1", split_method)


def _flip_aav(root: Path, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, valid, test = _flip_split(root, "aav", split_method)
    return _z_score_labels(train, valid, test)


def _remote_homology(root: Path, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "deepsf"
    train_rows = _read_json_split(base / "train.json")
    valid_rows = _read_json_split(base / "valid.json")
    test_rows = _read_json_split(base / f"test_{split_method}.json")
    train = _rows_to_df(train_rows, seq_key="primary", label_key="fold_label")
    valid = _rows_to_df(valid_rows, seq_key="primary", label_key="fold_label")
    test = _rows_to_df(test_rows, seq_key="primary", label_key="fold_label")
    train, valid, test = map(_add_prefix_m, (train, valid, test))
    for df in (train, valid, test):
        df["label"] = df["label"].astype(int)
    return train, valid, test


def _sol_pair(root: Path, subpath: str, *, z_score: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / subpath
    train = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "train.json")))
    valid = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "val.json")))
    test = _flatten_scalar_labels(_rows_to_df(_read_json_split(base / "test.json")))
    if z_score:
        return _z_score_labels(train, valid, test)
    return train, valid, test


def _deeploc_single(root: Path, subpath: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return _sol_pair(root, subpath, z_score=False)


def _deeploc2(root: Path, split_method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "deeploc-2" / "location"
    train = _rows_to_df(_read_json_split(base / "train.json"))
    valid = _rows_to_df(_read_json_split(base / "val.json"))
    test = _rows_to_df(_read_json_split(base / f"{split_method}.json"))
    return _multilabel_df(train), _multilabel_df(valid), _multilabel_df(test)


def _deeploc_signal(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "deeploc-2" / "signal"
    train = _rows_to_df(_read_json_split(base / "train.json"))
    valid = _rows_to_df(_read_json_split(base / "val.json"))
    test = _rows_to_df(_read_json_split(base / "test.json"))
    return _multilabel_df(train), _multilabel_df(valid), _multilabel_df(test)


def _multilabel_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["label"] = out["label"].map(lambda x: [float(v) for v in x])
    return out


def _ppi(root: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / "PPI" / dataset

    def _parse(rows: list[dict]) -> pd.DataFrame:
        seq_a, seq_b, labels = [], [], []
        for row in rows:
            seq = row["sequence"]
            if not isinstance(seq, (list, tuple)) or len(seq) != 2:
                raise ValueError(f"PPI example must have sequence=[protA, protB], got {type(seq)}")
            seq_a.append(str(seq[0]))
            seq_b.append(str(seq[1]))
            labels.append(int(row["label"]))
        return pd.DataFrame({"sequence_a": seq_a, "sequence_b": seq_b, "label": labels})

    train = _parse(_read_json_split(base / "train.json"))
    valid = _parse(_read_json_split(base / "val.json"))
    test = _parse(_read_json_split(base / "test.json"))
    return train, valid, test


PETA_SPLIT_OPTIONS: dict[str, list[str]] = {
    "gb1": ["sampled", "one_vs_rest", "two_vs_rest", "three_vs_rest", "low_vs_high"],
    "aav": ["des_mut", "mut_des", "sampled", "low_vs_high", "one_vs_many", "two_vs_many", "seven_vs_many"],
    "meltome": ["human", "human_cell", "mixed_split"],
    "remote_homology": ["fold_holdout", "family_holdout", "superfamily_holdout"],
    "deeploc_2": ["test", "hpa_test"],
}

PETA_DEFAULT_SPLIT: dict[str, str] = {
    "gb1": "one_vs_rest",
    "aav": "seven_vs_many",
    "meltome": "human",
    "remote_homology": "family_holdout",
    "deeploc_2": "test",
}


def load_peta_splits(
    peta_key: str,
    *,
    split_method: str | None = None,
    data_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir or PETA_DATA_DIR)
    if not root.is_dir():
        raise FileNotFoundError(
            f"PETA data not found at {root}\n"
            "  Run: ./scripts/setup_peta_data.sh\n"
            "  Or set PETA_DATA_DIR to the ft_datasets folder"
        )

    key = peta_key
    split = split_method

    if key == "fluorescence":
        return _tape_fluorescence(root)
    if key == "stability":
        return _tape_stability(root)
    if key == "gb1":
        split = split or PETA_DEFAULT_SPLIT["gb1"]
        return _flip_gb1(root, split)
    if key == "aav":
        split = split or PETA_DEFAULT_SPLIT["aav"]
        return _flip_aav(root, split)
    if key == "meltome":
        split = split or PETA_DEFAULT_SPLIT["meltome"]
        return _flip_meltome(root, split)
    if key == "remote_homology":
        split = split or PETA_DEFAULT_SPLIT["remote_homology"]
        return _remote_homology(root, split)
    if key == "deepsol":
        train, valid, test = _sol_pair(root, "sol/deepsol")
        for df in (train, valid, test):
            df["label"] = df["label"].astype(int)
        return train, valid, test
    if key == "esol":
        return _sol_pair(root, "sol/esol")
    if key == "solmut_blat":
        return _sol_pair(root, "sol/soluprotmutdb/Beta-lactamase_TEM", z_score=True)
    if key == "solmut_cs":
        return _sol_pair(root, "sol/soluprotmutdb/chalcone_synthase", z_score=True)
    if key == "solmut_lgk":
        return _sol_pair(root, "sol/soluprotmutdb/Levoglucosan_kinase", z_score=True)
    if key == "deeploc_1":
        train, valid, test = _deeploc_single(root, "deeploc-1/location")
        return _multilabel_df(train), _multilabel_df(valid), _multilabel_df(test)
    if key == "deeploc_binary":
        train, valid, test = _deeploc_single(root, "deeploc-1/binary")
        for df in (train, valid, test):
            df["label"] = df["label"].astype(int)
        return train, valid, test
    if key == "deeploc_2":
        split = split or PETA_DEFAULT_SPLIT["deeploc_2"]
        return _deeploc2(root, split)
    if key == "deeploc_signal":
        return _deeploc_signal(root)
    if key in ("ppi_yeast", "ppi_shs27k", "ppi_sun"):
        sub = key.replace("ppi_", "")
        return _ppi(root, sub)

    raise ValueError(f"Unknown PETA dataset key: {peta_key}")
