"""Training hyperparameters and metrics aligned with PETA (ProteinPretraining).

Reference: https://github.com/mingchen-li/ProteinPretraining
"""

from __future__ import annotations

from dataclasses import dataclass

from plm_benchmark.tasks import TaskSpec

# PETA README defaults (peta/train.py example)
PETA_LR = 1e-3
PETA_WEIGHT_DECAY = 0.001
PETA_MAX_EPOCHS = 100
PETA_PATIENCE = 20
PETA_POOLING = "attention1d"
PETA_DEFAULT_BATCH = 128


@dataclass(frozen=True)
class PetaTrainSettings:
    lr: float
    weight_decay: float
    max_epochs: int
    patience: int
    metric: str
    greater_is_better: bool
    train_encoder: bool


def normalize_peta_method(method: str) -> str:
    """Map queue scenario methods to core PETA train mode."""
    if method in ("full_ft", "full_ft_peta20", "full_ft_peta100"):
        return "full_ft"
    if method == "embed_head":
        return "embed_head"
    raise ValueError(f"Unknown PETA method: {method!r}")


def peta_settings(spec: TaskSpec, method: str) -> PetaTrainSettings:
    """Map CLI method to PETA finetune mode: embed_head -> head, full_ft -> all."""
    core = normalize_peta_method(method)
    if core == "embed_head":
        train_encoder = False
    elif core == "full_ft":
        train_encoder = True
    else:
        raise ValueError(
            f"PETA tasks use PETA protocol: --method embed_head (freeze encoder, train head) "
            f"or full_ft (train all). Got {method!r}."
        )

    return PetaTrainSettings(
        lr=PETA_LR,
        weight_decay=PETA_WEIGHT_DECAY,
        max_epochs=PETA_MAX_EPOCHS,
        patience=PETA_PATIENCE,
        metric=spec.metric,
        greater_is_better=spec.greater_is_better,
        train_encoder=train_encoder,
    )
