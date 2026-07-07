
from __future__ import annotations

from plm_benchmark.tasks import TaskSpec

# Task names use prefix peta_ to distinguish from Rost tasks (GB1, AAV, ...).
# Use --split-method for datasets with multiple splits (gb1, aav, meltome, ...).
# Training recipe (lr / epochs / patience / weight-decay) is uniform for all
# tasks — see config.py and methods/train.py.

PETA_TASK_NAMES: list[str] = [
    "peta_fluorescence",
    "peta_stability",
    "peta_remote_homology",
    "peta_gb1",
    "peta_aav",
    "peta_meltome",
    "peta_deepsol",
    "peta_esol",
    "peta_solmut_blat",
    "peta_solmut_cs",
    "peta_solmut_lgk",
    "peta_deeploc_1",
    "peta_deeploc_binary",
    "peta_deeploc_2",
    "peta_deeploc_signal",
    "peta_ppi_yeast",
    "peta_ppi_shs27k",
    "peta_ppi_sun",
]

PETA_TASKS: dict[str, TaskSpec] = {
    "peta_fluorescence": TaskSpec(
        "peta_fluorescence", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="fluorescence",
    ),
    "peta_stability": TaskSpec(
        "peta_stability", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="stability",
    ),
    "peta_remote_homology": TaskSpec(
        "peta_remote_homology", "", "", "", "classification", 1195, "accuracy",
        data_source="peta", peta_key="remote_homology", default_split="family_holdout",
    ),
    "peta_gb1": TaskSpec(
        "peta_gb1", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="gb1", default_split="one_vs_rest",
    ),
    "peta_aav": TaskSpec(
        "peta_aav", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="aav", default_split="seven_vs_many",
    ),
    "peta_meltome": TaskSpec(
        "peta_meltome", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="meltome", default_split="human",
    ),
    "peta_deepsol": TaskSpec(
        "peta_deepsol", "", "", "", "classification", 2, "accuracy",
        data_source="peta", peta_key="deepsol",
    ),
    "peta_esol": TaskSpec(
        "peta_esol", "", "", "", "regression", 1, "mse",
        data_source="peta", peta_key="esol", greater_is_better=False,
    ),
    "peta_solmut_blat": TaskSpec(
        "peta_solmut_blat", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="solmut_blat",
    ),
    "peta_solmut_cs": TaskSpec(
        "peta_solmut_cs", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="solmut_cs",
    ),
    "peta_solmut_lgk": TaskSpec(
        "peta_solmut_lgk", "", "", "", "regression", 1, "spearmanr",
        data_source="peta", peta_key="solmut_lgk",
    ),
    "peta_deeploc_1": TaskSpec(
        "peta_deeploc_1", "", "", "", "multilabel", 10, "accuracy",
        data_source="peta", peta_key="deeploc_1",
    ),
    "peta_deeploc_binary": TaskSpec(
        "peta_deeploc_binary", "", "", "", "classification", 2, "accuracy",
        data_source="peta", peta_key="deeploc_binary",
    ),
    "peta_deeploc_2": TaskSpec(
        "peta_deeploc_2", "", "", "", "multilabel", 10, "accuracy",
        data_source="peta", peta_key="deeploc_2", default_split="test",
    ),
    "peta_deeploc_signal": TaskSpec(
        "peta_deeploc_signal", "", "", "", "multilabel", 9, "accuracy",
        data_source="peta", peta_key="deeploc_signal",
    ),
    "peta_ppi_yeast": TaskSpec(
        "peta_ppi_yeast", "", "", "", "ppi", 2, "accuracy",
        data_source="peta", peta_key="ppi_yeast",
    ),
    "peta_ppi_shs27k": TaskSpec(
        "peta_ppi_shs27k", "", "", "", "ppi", 7, "accuracy",
        data_source="peta", peta_key="ppi_shs27k",
    ),
    "peta_ppi_sun": TaskSpec(
        "peta_ppi_sun", "", "", "", "ppi", 2, "accuracy",
        data_source="peta", peta_key="ppi_sun",
    ),
}
