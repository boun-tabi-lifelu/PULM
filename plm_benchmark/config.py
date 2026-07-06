"""Paths and model registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "training data"


def _default_peta_data_dir() -> Path:
    candidates = [
        ROOT / "training data" / "PETA" / "ft_datasets",
        ROOT / "ft_datasets",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    return candidates[-1]


PETA_DATA_DIR = Path(os.environ.get("PETA_DATA_DIR", str(_default_peta_data_dir())))
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "embeddings"
LOG_CSV = OUTPUTS_DIR / "experiments.csv"
COMPARE_CSV = OUTPUTS_DIR / "comparison.csv"

PULM_MODELS_ROOT = Path(os.environ.get("PULM_MODELS_ROOT", "/cta/share/users/PULM/models"))

RARE_AA = ["O", "B", "U", "Z", "J"]
MAX_SEQ_LENGTH = 1024

LR_FULL_FT = 2e-5
LR_LORA = 3e-4
LR_EMBED_HEAD = 1e-4

FINETUNE_SEEDS: tuple[int, ...] = (42, 43, 44)
CHECKPOINT_POLICY = "best_val"


@dataclass(frozen=True)
class ModelConfig:
    name: str
    checkpoint: str
    backend: str = "esm"
    source: str = "hub"  # hub | pulm


HUB_MODELS: dict[str, ModelConfig] = {
    "esm2_8m": ModelConfig("esm2_8m", "facebook/esm2_t6_8M_UR50D", "esm", "hub"),
    "esm2_35m": ModelConfig("esm2_35m", "facebook/esm2_t12_35M_UR50D", "esm", "hub"),
    "esm2_150m": ModelConfig("esm2_150m", "facebook/esm2_t30_150M_UR50D", "esm", "hub"),
}

EMBED_CACHE_SUFFIX: dict[str, str] = {
    "esm2_8m": "ESM2_8M",
    "esm2_35m": "ESM2_35M",
    "esm2_150m": "ESM2_150M",
}

DEFAULT_MODEL = "esm2_8m"


def _pulm_slug(rel: Path) -> str:
    return str(rel).replace("/", "__")


@lru_cache(maxsize=1)
def discover_pulm_models(root: str | None = None) -> dict[str, ModelConfig]:
    base = Path(root) if root else PULM_MODELS_ROOT
    if not base.is_dir():
        return {}

    found: dict[str, ModelConfig] = {}
    for weights in sorted(base.rglob("final/model.safetensors")):
        model_dir = weights.parent
        rel = model_dir.relative_to(base)
        name = _pulm_slug(rel)
        found[name] = ModelConfig(name=name, checkpoint=str(model_dir), backend="esm", source="pulm")
    return found


def get_models() -> dict[str, ModelConfig]:
    return {**HUB_MODELS, **discover_pulm_models()}


def resolve_model(name: str | None = None, checkpoint: str | None = None) -> ModelConfig:
    models = get_models()

    # scratch_<model>: randomly-initialised baseline that borrows <model>'s tokenizer.
    if name and name.startswith("scratch_"):
        base = resolve_model(name[len("scratch_") :], checkpoint)
        return ModelConfig(name=name, checkpoint=base.checkpoint, backend="scratch", source="scratch")

    if checkpoint:
        ckpt = str(Path(checkpoint).resolve())
        if name and name in models:
            base = models[name]
            return ModelConfig(base.name, ckpt, base.backend, base.source)
        slug = Path(ckpt).name
        if ckpt.endswith("/final") or ckpt.endswith("\\final"):
            slug = Path(ckpt).parent.name
        return ModelConfig(
            slug,
            ckpt,
            "esm",
            "pulm" if PULM_MODELS_ROOT in Path(ckpt).parents else "local",
        )

    if not name or name not in models:
        available = ", ".join(sorted(models)[:8])
        raise ValueError(f"Unknown model '{name}'. Examples: {available} ... (run: python run.py list-models)")
    return models[name]
