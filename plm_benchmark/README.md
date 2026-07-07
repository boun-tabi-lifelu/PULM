# plm_benchmark — Downstream fine-tuning & evaluation

One **unified PyTorch pipeline** fine-tunes and evaluates protein language models across
two task suites — **Rost / FLIP** and **PETA** — using a single training loop. It is built
to compare **tokenizers**: every task is protein-level (not per-residue), so models with
different vocabularies are directly comparable.

## Design in one picture

```
sequence(s) ──tokenizer──▶ encoder ──▶ Attention1dPooling ──▶ Linear→ReLU→Linear ──▶ logits
                          (PLM or                (pooling)          (task head)
                           scratch embedding)
```

- **Encoder** — a pretrained PLM (`AutoModel`) or the from-scratch embedding baseline.
- **Head** — `attention1d` is only the **pooling** step; the final logits come from the
  linear projection head on top of the pooled vector. PPI tasks pool each protein, sum the
  pair, then project.
- **One loop** — HuggingFace `Trainer` (dynamic padding, `group_by_length`, bf16, early
  stopping). Rost and PETA differ only by per-task **recipe** (lr / epochs / weight-decay /
  patience), not by code path.

## Tasks

```bash
python -m plm_benchmark.cli list-tasks
```

- **Rost** (per-protein): `GB1`, `AAV`, `GFP`, `Meltome`, `Stab`, `SubLoc`.
- **PETA** (`peta_*`): fitness (fluorescence, stability, gb1, aav, meltome), solubility
  (deepsol, esol, solmut_*), localization (deeploc_1, deeploc_binary, deeploc_2,
  deeploc_signal), remote homology, and PPI (yeast, shs27k, sun).

Metrics follow each source: Spearman (regression fitness), MSE (eSol), accuracy
(single-label classification / PPI), and multilabel accuracy via `torchmetrics`
(`Accuracy(task="multilabel")`) for the DeepLoc tasks — matching PETA exactly.

Task groups for `--task`: `all` (Rost only), `peta_all`, `everything`, or a comma-separated
list. PETA datasets with multiple partitions take `--split-method` (e.g. `one_vs_rest`).

## Training modes (`--method`)

| Method | Encoder | Notes |
|--------|---------|-------|
| `full_ft` | trained | Full fine-tuning. |
| `embed_head` | **frozen** | Train the head only (encoder in eval/no-grad). |
| `lora` | LoRA adapters | Base frozen, adapters + head trained. |
| `full_ft_peta20` | trained | `full_ft` capped at 20 epochs (PETA short budget). |

**Recipe** comes from the task spec: Rost tasks use lr 2e-5, no early stopping; PETA tasks
use the PETA protocol (lr 1e-3, weight-decay 1e-3, patience 20, up to 100 epochs). Override
per run with `--lr`, `--epochs`, `--patience`.

> Note: Rost tasks now use the attention1d head (not the old HF classification head), so
> absolute Rost numbers differ from earlier HF-head runs.

## Models

```bash
python -m plm_benchmark.cli list-models
```

- Hub ESM-2 (`esm2_8m`, `esm2_35m`, `esm2_150m`) and auto-discovered PULM checkpoints.
- `--checkpoint <dir>` points at a local checkpoint; its own tokenizer is used.

### From-scratch tokenizer baseline

A randomly-initialised embedding encoder isolates the tokenizer's effect from pretraining.
It shares the exact same head and training recipe, so any score delta reflects the tokenizer.

```bash
# ESM-2 tokenizer (same across all sizes)
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft \
    --model scratch --tokenizer esm2

# A raw tokenizers JSON (PUMA/BPE) — wrapped with ESM-2 specials + <cls>..<eos>
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft \
    --model scratch --tokenizer /path/to/hf_uniref50_bpe_6400.json --scratch-dim 320
```

`--tokenizer` accepts: `esm2`, a `tokenizer.json` path, a saved-tokenizer directory, or a
Hub id. `--scratch-dim` sets the embedding size (default 320, matching ESM2-8M). The run is
named `scratch_<tokenizer-label>`.

## Weights & Biases

Pass `--wandb_project` to log train/eval loss and task metrics per run; the test score is
logged at the end. Each `(task, seed, split)` becomes its own run.

```bash
python -m plm_benchmark.cli train --task GB1 --method full_ft --model esm2_8m \
    --wandb_project pulm --wandb_group esm2_8m
```

Also: `--wandb_run_name`, `--wandb_entity`, `--wandb_mode {online,offline,disabled}`, and
`--wandb_run_id`/`--wandb_resume` together with `--resume-from-checkpoint` to continue a run.

## Data layout

Place the downloaded datasets under `plm_benchmark/data/` (auto-detected):

```
plm_benchmark/data/
├── training data/     # Rost / FLIP task folders (GB1/, AAV/, ...)
└── ft_datasets/       # PETA benchmark datasets
```

Overridable via `ROST_DATA_DIR` and `PETA_DATA_DIR` env vars. (Legacy locations at the repo
root are still detected as a fallback.)

## Outputs

- `outputs/experiments.csv` — one row per run (now includes a `split` column).
- `outputs/comparison.csv` — aggregated mean/std per `(task, model, method, split)` via
  `python -m plm_benchmark.cli compare`.
- `outputs/<model>/<task>/<split>/<method>/seed_<n>/` — Trainer cache + `finetuned_weights.pth`.

## Common commands

```bash
python -m plm_benchmark.cli train --task everything --method full_ft --model esm2_35m --seeds 42,43,44
python -m plm_benchmark.cli train --task peta_aav --method embed_head --model esm2_8m --split-method seven_vs_many
python -m plm_benchmark.cli list
python -m plm_benchmark.cli compare
```
