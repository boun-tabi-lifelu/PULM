# PULM
Protein Units Language Modeling

Replication code for [Rost et al., Nat Commun 2024](https://www.nature.com/articles/s41467-024-51844-2) ([upstream data repo](https://github.com/RSchmirler/data-repo_plm-finetune-eval)).

Training modes: `full_ft`, `lora`, `embed_head`. Per-protein tasks: GB1, AAV, GFP, Meltome, Stab, SubLoc.

## Setup

```bash
conda env create -f environment.yml
conda activate finetune
unzip -q "training data.zip"
```

## Run

```bash
python run.py train --task GB1 --method full_ft --model esm2_8m --gpu 2
python run.py train --task all --method embed_head --model esm2_8m --gpu 2
python run.py compare
python run.py list
```

Scores: `outputs/experiments.csv`. Comparison table: `outputs/comparison.csv`. Fine-tuned weights: `outputs/<task>/finetuned_weights.pth`. Embedding caches: `outputs/embeddings/<task>/`.

## Repo layout

| Path | Purpose |
|------|---------|
| `run.py`, `plm_benchmark/` | CLI replication pipeline |
| `environment.yml` | Conda env |
| `training data.zip` | Task splits (unzip → `training data/`) |
| `outputs/` | Run logs (CSVs committed; weights/embeddings local) |

Per-residue tasks (SecStr, Disorder) are in `notebooks/` and `training data/SecStr/`, not in `run.py` yet. (Will be included soon!)
