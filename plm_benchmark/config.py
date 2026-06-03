"""Paths and model registry."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training data"
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "embeddings"
LOG_CSV = OUTPUTS_DIR / "experiments.csv"
COMPARE_CSV = OUTPUTS_DIR / "comparison.csv"
SOURCE_DATA_MAIN = ROOT / "source_data_main"
SOURCE_DATA_SOM = ROOT / "source_data_SOM"

RARE_AA = ["O", "B", "U", "Z", "J"]
MAX_SEQ_LENGTH = 1024

# Default learning rates per method (paper Table S10 / S12)
LR_FULL_FT = 2e-5
LR_LORA = 3e-4
LR_EMBED_HEAD = 1e-4


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: str
    backend: str = "esm"


MODELS: dict[str, ModelConfig] = {
    "esm2_8m": ModelConfig("esm2_8m", "facebook/esm2_t6_8M_UR50D", "esm"),
    "esm2_35m": ModelConfig("esm2_35m", "facebook/esm2_t12_35M_UR50D", "esm"),
    "esm2_150m": ModelConfig("esm2_150m", "facebook/esm2_t30_150M_UR50D", "esm"),
}

# Embedding cache filenames ({split}_{suffix}.pkl), matching paper notebooks
EMBED_CACHE_SUFFIX: dict[str, str] = {
    "esm2_8m": "ESM2_8M",
    "esm2_35m": "ESM2_35M",
    "esm2_150m": "ESM2_150M",
}

DEFAULT_MODEL = "esm2_8m"
