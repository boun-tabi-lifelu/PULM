"""Paths and model registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT = PKG_DIR.parent
# Recommended layout: plm_benchmark/data/{training data, ft_datasets}
DATA_ROOT = PKG_DIR / "data"


def _first_dir(candidates: list[Path], default: Path) -> Path:
    for path in candidates:
        if path.is_dir():
            return path
    return default


def _default_rost_data_dir() -> Path:
    return _first_dir(
        [DATA_ROOT / "training data", ROOT / "training data"],
        DATA_ROOT / "training data",
    )


def _default_peta_data_dir() -> Path:
    return _first_dir(
        [
            DATA_ROOT / "ft_datasets",
            DATA_ROOT / "training data" / "PETA" / "ft_datasets",
            ROOT / "training data" / "PETA" / "ft_datasets",
            ROOT / "ft_datasets",
        ],
        DATA_ROOT / "ft_datasets",
    )


DATA_DIR = Path(os.environ["ROST_DATA_DIR"]) if os.environ.get("ROST_DATA_DIR") else _default_rost_data_dir()
PETA_DATA_DIR = Path(os.environ["PETA_DATA_DIR"]) if os.environ.get("PETA_DATA_DIR") else _default_peta_data_dir()
OUTPUTS_DIR = ROOT / "outputs"
CACHE_DIR = OUTPUTS_DIR / "embeddings"
LOG_CSV = OUTPUTS_DIR / "experiments.csv"
COMPARE_CSV = OUTPUTS_DIR / "comparison.csv"

# PULM_MODELS_ROOT = Path(os.environ.get("PULM_MODELS_ROOT", "/cta/share/users/PULM/models"))
PULM_MODELS_ROOT = Path(os.environ.get("PULM_MODELS_ROOT", "/shared/PULM/models"))

RARE_AA = ["O", "B", "U", "Z", "J"]
MAX_SEQ_LENGTH = 1024

# Learning rate by training *regime* (not by task source):
#   full fine-tune of a pretrained encoder -> small (preserve pretrained weights)
#   frozen encoder + head, or from-scratch encoder -> large (fresh params)
#   LoRA adapters -> mid
LR_FULL_FT = 2e-5
LR_HEAD = 1e-3     # embed_head: frozen encoder, train head only
LR_SCRATCH = 1e-3  # scratch baseline: random init, no weights to preserve
LR_LORA = 3e-4

# Unified downstream recipe — same for every task (Rost + PETA). CLI can override.
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
WEIGHT_DECAY = 0.01

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


def resolve_model(
    name: str | None = None,
    checkpoint: str | None = None,
    tokenizer: str | None = None,
    parent_collapse: bool = False,
) -> ModelConfig:
    models = get_models()

    # scratch: randomly-initialised baseline. The tokenizer is supplied separately
    # (--tokenizer) and names the run, e.g. scratch_AA / scratch_PUMA_..._all
    # (or scratch_PUMA_..._PC_all with --parent-collapse).
    if name == "scratch":
        from plm_benchmark.tokenizers import tokenizer_label

        return ModelConfig(
            name=f"scratch_{tokenizer_label(tokenizer, parent_collapse)}",
            checkpoint=checkpoint or "",
            backend="scratch",
            source="scratch",
        )

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
        raise ValueError(
            f"Unknown model '{name}'. Examples: {available} ... "
            "(run: python -m plm_benchmark.cli list-models)"
        )
    return models[name]
