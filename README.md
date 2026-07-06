# PULM
Protein Units Language Modeling

Replication code for [Rost et al., Nat Commun 2024](https://www.nature.com/articles/s41467-024-51844-2) ([upstream data repo](https://github.com/RSchmirler/data-repo_plm-finetune-eval)) and the [PETA](https://github.com/mingchen-li/ProteinPretraining) protein-tokenizer tasks.

**One PyTorch pipeline for every task** (Rost + PETA), all sharing an attention1d pooling + linear head. Modes:

- `full_ft` — train the encoder + head.
- `embed_head` — freeze the encoder, train the head only.
- `lora` — LoRA adapters on the encoder + head.
- `full_ft_peta20` — `full_ft` capped at 20 epochs (PETA short-budget scenario).

Per-task recipe (lr / epochs / weight-decay / early-stopping) comes from the task spec: Rost tasks keep the Rost recipe (lr 2e-5, no early stopping), PETA tasks use the PETA protocol (lr 1e-3, wd 0.001, patience 20). Rost tasks now run through the attention1d head (not the old HF classification head), so absolute Rost numbers differ from earlier HF-head runs.

**Tokenizer-only baseline:** prefix any model with `scratch_` (e.g. `scratch_esm2_8m`) to train a randomly-initialised embedding encoder that borrows that model's tokenizer — isolating the tokenizer's effect from pretraining. Set `--scratch-dim` for the embedding size (default 320).

## Setup

```bash
conda env create -f environment.yml
conda activate finetune
unzip -q "training data.zip"
```

## Run

```bash
python run.py train --task GB1 --method full_ft --model esm2_8m --gpu 2
python run.py train --task peta_gb1 --method embed_head --model esm2_8m --gpu 2
python run.py train --task peta_gb1 --method full_ft --model scratch_esm2_8m --gpu 2   # baseline
python run.py train --task GB1 --method full_ft --model esm2_8m --wandb_project pulm    # wandb
python run.py compare
python run.py list
```

wandb: pass `--wandb_project` to log train/eval loss and task metrics per run (`--wandb_run_name`, `--wandb_group`, `--wandb_entity`, `--wandb_mode`, and `--wandb_run_id`/`--wandb_resume` with `--resume-from-checkpoint` to continue a run).

Scores: `outputs/experiments.csv` (now includes a `split` column). Comparison table: `outputs/comparison.csv`. Run dir + weights: `outputs/<model>/<task>/<split>/<method>/seed_<n>/`.

## Repo layout

| Path | Purpose |
|------|---------|
| `run.py`, `plm_benchmark/` | CLI replication pipeline |
| `environment.yml` | Conda env |
| `training data.zip` | Task splits (unzip → `training data/`) |
| `outputs/` | Run logs (CSVs committed; weights/embeddings local) |

Per-residue tasks (SecStr, Disorder) are in `notebooks/` and `training data/SecStr/`, not in `run.py` yet. (Will be included soon!)
