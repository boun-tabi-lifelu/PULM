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
(single-label classification / PPI), and per-label multilabel accuracy for the DeepLoc
tasks — identical to PETA's `torchmetrics Accuracy(task="multilabel")` (computed in NumPy).

Task groups for `--task`: `all` (Rost only), `peta_all`, `everything`, or a comma-separated
list.

**Splits.** Some FLIP/PETA datasets have multiple partitions (gb1, aav, meltome, deeploc_2,
remote_homology). Each `(task, split)` is a separate run, logged with its own `split` value:
- `everything` / `peta_all` → every task × **every curated split** (`list-tasks` shows the set).
- `--task peta_gb1 --split-method all` → all curated splits of that task.
- `--task peta_gb1 --split-method two_vs_rest` → that one split.
- `--task peta_gb1` (no `--split-method`) → the **default** split only (backward-compatible;
  task names are unchanged, so earlier default-split results stay valid).

## PETA task selection (the curated 19)

`peta_all` / `everything` run a **curated 19** task/splits — 18 protein-wise + `ppi_shs27k` —
chosen from measured cost and discriminative power (~26% of the full grid's cost). Everything
dropped below is **still runnable explicitly** (`--task peta_ppi_yeast`, `--split-method hpa_test`);
it is only excluded from the groups.

### Kept (19)
| task | splits | why |
|---|---|---|
| `peta_fluorescence`, `peta_stability` | default | TAPE fitness; clean vocab trend |
| `peta_gb1` | `two_vs_rest`, `three_vs_rest`, `low_vs_high` | FLIP fitness at 3 difficulties (cheap: 1–5 min) |
| `peta_aav` | `two_vs_many` | one AAV split; the canonical FLIP 28,626/3,181/50,776 |
| `peta_meltome` | `human`, `mixed_split` | thermostability, two regimes |
| `peta_remote_homology` | `family`/`superfamily`/`fold` holdout | structure; all 3 ≈ **free** (shared train, see below) |
| `peta_deepsol`, `peta_esol`, `peta_solmut_{blat,cs,lgk}` | default | solubility: classification + MSE + 3 mutation sets |
| `peta_deeploc_binary`, `peta_deeploc_2` | default / `test` | localization |
| `peta_ppi_shs27k` | default | the **only** protein-pair task that isn't leaky (below) |

### Dropped — and why

**PPI binary tasks (`ppi_yeast`, `ppi_sun`) — measured by memorisation, not biology.**
PETA splits protein *pairs* at random, so the same proteins appear in train and test. A model can
score highly by memorising which proteins are promiscuous. Our from-scratch baseline proves it —
same architecture, same method, **only the tokenizer differs**:

| task | scratch **AA** | scratch best sub-word | gap |
|---|---|---|---|
| `ppi_sun` | 0.502 *(chance)* | 0.989 | **+48.7** |
| `ppi_yeast` | 0.567 | 0.954 | **+38.7** |
| `ppi_shs27k` | 0.548 | 0.574 | **+2.6** ✅ |

A *randomly-initialised single-layer* encoder reaches 0.99 on `sun` with any sub-word tokenizer and
exactly chance with the amino-acid tokenizer — and beats pretrained ESM2-35M (0.948). An untrained
1-layer model contains no biology; the only thing a larger vocabulary buys it is more distinct tokens,
i.e. **protein fingerprinting**. These tasks would therefore reward vocabulary size for an artifactual
reason and invert our conclusion, so they are excluded. `shs27k` predicts interaction *type* (7 classes)
among pairs that already interact, so identity doesn't help — the shortcut is structurally closed, it
follows the global vocab trend in the pretrained arm (AA 0.512 → BPE_25600 0.413), and it reproduces
PETA's reported ~0.52. It is kept as the protein-pair representative.

**Saturated (`deeploc_1`, `deeploc_signal`)** — 0.90–0.96 for every tokenizer; a ceiling effect
leaves almost no signal to separate tokenizers.

**Cost (`aav/des_mut`, `aav/mut_des`, `aav/seven_vs_many`, `ppi_sun`)** — 1.5–9 h *per model per seed*
(`des_mut` alone: ~6 h). `aav/two_vs_many` covers AAV at ~1/7 the cost.

**Redundant** — `deeploc_2/hpa_test` (tracks `test`), `meltome/human_cell` (tracks `human`),
`aav/{one_vs_many, low_vs_high, sampled}`, `gb1/sampled`.

**Degenerate (`gb1/one_vs_rest`)** — after the canonical FLIP rebuild it has **25 train / 3 validation**
sequences. Spearman on 3 points is undefined-to-meaningless, so best-val selection is pure noise.

> Caveat on a kept split: `gb1/two_vs_rest` has only **43** validation sequences (the pipeline warns).
> Its test scores swing 0.005–0.585 at healthy val scores — treat it as low-confidence.

### Shared-train tasks: one training run, many test sets
`remote_homology` (3 holdouts) and `deeploc_2` (`test`/`hpa_test`) **share the same train and
validation set** — only the test file differs. (Proof: `val_score` is identical across the holdouts.)
The pipeline detects this and trains **once**, then scores every requested test set, emitting one row
per split. That's a 3× / 2× saving and it is *exact* — literally the same model.

Because those rows come from one run, their `duration_sec` is the whole run's time and is **not
additive** across the splits of that task.

## Training modes (`--method`)

| Method | Encoder | Notes |
|--------|---------|-------|
| `full_ft` | trained | Full fine-tuning. |
| `embed_head` | **frozen** | Train the head only (encoder in eval/no-grad). |
| `lora` | LoRA adapters | Base frozen, adapters + head trained. |
| `full_ft_peta20` | trained | `full_ft` capped at 20 epochs (PETA short budget). |

**One uniform recipe for every task** (Rost + PETA) — self-consistent by design, not a
replication of either paper:

- `max_epochs = 50`, early-stopping `patience = 10` (eval events), `weight_decay = 0.01`.
- **Learning rate by training regime**, not by task source: full-fine-tuning a pretrained
  encoder needs a small lr; a frozen head or a from-scratch encoder needs a large one.

  | Regime | LR |
  |--------|----|
  | `full_ft` on a pretrained encoder | `2e-5` |
  | `embed_head` (frozen) / `scratch` baseline | `1e-3` |
  | `lora` | `3e-4` |

Override per run with `--lr`, `--epochs`, `--patience`. `load_best_model_at_end` always
reports the best-val checkpoint, so early stopping only saves compute.

> Notes: Rost tasks now use the attention1d head (not the old HF classification head) and
> the unified recipe, so absolute Rost/PETA numbers differ from earlier per-source runs —
> that is the point. Since an "epoch" is not equal compute across tasks of very different
> sizes, the meaningful comparison is **within a task, across tokenizers/models**, where
> the data and recipe are identical.

## Models

```bash
python -m plm_benchmark.cli list-models
```

- Hub ESM-2 (`esm2_8m`, `esm2_35m`, `esm2_150m`) and auto-discovered PULM checkpoints.
- `--checkpoint <dir>` points at a local checkpoint; its own tokenizer is used.
- **PUMA parent-collapsed (`_PC`) checkpoints** are handled automatically: if a checkpoint
  contains `full_tokenizer/` + `collapse.npy`, tokenization is done with the full PUMA vocab
  and remapped child→parent to the model's reduced ids (matching how it was pretrained). No
  flag needed; any `--tokenizer` override is ignored for these.

### From-scratch tokenizer baseline

A randomly-initialised embedding encoder isolates the tokenizer's effect from pretraining.
It shares the exact same head and training recipe, so any score delta reflects the tokenizer.

```bash
# Amino-acid (ESM-2) tokenizer — same across all sizes
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft \
    --model scratch --tokenizer aa

# A raw tokenizers JSON (PUMA/BPE) — wrapped with ESM-2 specials + <cls>..<eos>
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft \
    --model scratch --tokenizer /path/to/hf_uniref50_bpe_6400.json --scratch-dim 320
```

`--tokenizer` accepts: `aa` (amino-acid/ESM-2 tokenizer; `esm2` is an accepted alias), a
`tokenizer.json` path, a saved-tokenizer directory, or a Hub id. `--scratch-dim` sets the
embedding/hidden size (default 320, matching ESM2-8M).

The `tokenizer` label matches the PLM run-slug convention so a scratch run and the PLM using
the same tokenizer share one label (`uniref50` is static → omitted; `_all` = trained on all):

| `--tokenizer` | tokenizer label / run name |
|---|---|
| `aa` | `AA` → `scratch_AA` |
| `.../blosum62/hf_uniref50_mutbpe_0.7_3_12_0.05_12800.json` | `PUMA_blosum62_07_005_12800_all` |
| `.../bpe/hf_uniref50_bpe_12800.json` | `BPE_12800_all` |

#### PUMA parent-collapse: `--parent-collapse`

For a scratch run over a **PUMA** tokenizer, `--parent-collapse` folds each child token onto
its mutational parent — the model then trains on the *reduced* vocab (parents + singletons +
specials), exactly like `plm_train`'s collapse-to-parent pretraining. Segmentation is done
with the full PUMA vocab and the ids are remapped child→parent automatically. It requires the
PUMA **family JSON** next to the tokenizer (the non-`hf_` sibling of the `.json`, same dir).
Scratch + PUMA only (BPE/AA are rejected; pretrained `_PC` checkpoints are auto-detected). The
tokenizer label gains a `_PC` marker (e.g. `PUMA_blosum62_07_005_12800_PC_all`), so collapsed
and uncollapsed scratch runs are distinct in `experiments.csv` and `outputs/`.

```bash
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft --model scratch \
    --tokenizer /path/to/blosum62/hf_uniref50_mutbpe_0.7_3_12_0.05_12800.json --parent-collapse
```

#### Baseline capacity: `--scratch-layers` / `--scratch-heads`

The scratch encoder's strength is a dial. The shared attention1d pooling + linear head is
unchanged in every case, so only the encoder differs from a real PLM.

- **`--scratch-layers 0` (default) — bag-of-tokens floor.** Token embedding + positional
  embedding + LayerNorm, then straight to pooling. No token ever sees another token, so it
  measures *only* the signal the vocabulary carries on its own. It is a deliberately weak
  floor: because it cannot model interactions between tokens (e.g. epistasis between
  mutated positions), the gap to a PLM here over-attributes performance to pretraining.
- **`--scratch-layers 1` or `2` — small non-pretrained model with context.** Adds that many
  Transformer blocks (self-attention + FFN + LayerNorm), so tokens interact. This is a
  *fairer* control: the difference from a PLM is now closer to "pretrained weights" rather
  than "architecture." `--scratch-heads` sets attention heads per block (default 8) and must
  divide `--scratch-dim` (320 ÷ 8 = 40 ✓). More layers/heads = more capacity and a stronger
  baseline, but also a larger model that takes longer and drifts further from "just the
  tokenizer."

Recommended: report **both** — `--scratch-layers 0` (floor) and `--scratch-layers 1-2`
(fair small model). Seeing how much one or two blocks of context close the gap is itself
informative: it shows whether a tokenizer's advantage survives once the model can mix tokens.

```bash
# Floor (bag-of-tokens) and a 2-layer contextual baseline, same tokenizer
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft --model scratch --tokenizer aa
python -m plm_benchmark.cli train --task peta_gb1 --method full_ft --model scratch --tokenizer aa \
    --scratch-layers 2 --scratch-heads 8
```

## Weights & Biases

Pass `--wandb_project` to log train/eval loss and task metrics per run; the test score is
logged at the end. Each `(task, seed, split)` becomes its own run, named
`{model}/{tokenizer}/{task}/{split}/{method}/seed{n}`, and runs are **grouped by
`{model}/{tokenizer}`** automatically (override with `--wandb_group`).

```bash
python -m plm_benchmark.cli train --task GB1 --method full_ft --model esm2_8m \
    --wandb_project pulm
```

Also: `--wandb_run_name`, `--wandb_group`, `--wandb_entity`,
`--wandb_mode {online,offline,disabled}`, and `--wandb_run_id`/`--wandb_resume` together with
`--resume-from-checkpoint` to continue a run.

## FLIP splits: regenerate from `raw/` (required once)

The PETA-shipped JSONs for `flip/{aav,gb1,meltome}` are broken: **`valid.json` is a byte-identical
copy of `test.json`** (so early stopping and best-checkpoint selection ran on test), and
`train.json` is `SET=train` *including* the validation rows. `raw/<split>.fasta` is the source of
truth. Regenerate them:

```bash
python scripts/regen_flip_splits.py --dry-run   # preview counts
python scripts/regen_flip_splits.py            # delete + rewrite train/valid/test.json
```

Canonical FLIP semantics: `SET=train & VALIDATION=False` → train, `SET=train & VALIDATION=True` →
validation, `SET=test` → test, anything else (`SET=nan`) excluded. The script pins
`gb1/three_vs_rest = 2691/299/5743` and `aav/two_vs_many = 28626/3181/50776`.

**Split guard.** Every `load_splits` call runs `assert_splits_sane`: it **hard-fails** if validation
is identical to test (or >25% of it appears in test) and **warns** on the incidental duplicate
sequences FLIP genuinely ships (records stay disjoint). It also warns loudly when a validation set
is tiny — `gb1/one_vs_rest` has only **3** validation sequences after the rebuild, so best-val
selection there is noise; it's excluded from the curated sweep but still runnable explicitly.

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

- `outputs/experiments.csv` — one row per run. Key columns: `start_datetime`, `end_datetime`,
  `duration_sec`, `task`, **`model`**, **`tokenizer`**, **`vocab_size`** (actual embedding rows —
  the *post-collapse* size for parent-collapsed runs), `method`, `split`, `metric`,
  `test_score`, `val_score`, `epochs`, `lr`, `batch`, `seed`, `full_name` (original
  registry/slug name), `scratch_dim/layers/heads` (scratch runs only), `git_commit`,
  `checkpoint_policy`, `checkpoint`, `run_dir`. `model`/`tokenizer` are split from the raw
  name: e.g. `ESM2_35M` + `PUMA_blosum62_07_005_12800_all`; hub ESM2 → tokenizer `AA`;
  scratch → `scratch_d<dim>_l<layers>_h<heads>` + the `--tokenizer` label.
- **`collapsed`** — a NaN metric means the model emitted *constant* predictions: the run **failed**,
  it is not a score. Flagged loudly at run time (`test_nan` / `val_nan`), counted as `n_collapsed`
  in `comparison.csv`, and **never averaged into `test_mean`**. Watch for it on large-vocab
  tokenizers (BPE_25600/51200), where frozen-encoder runs tend to degenerate.
- `outputs/comparison.csv` — aggregated mean/std per `(task, model, tokenizer, method, split)`
  via `python -m plm_benchmark.cli compare`; includes `n_seeds` (valid runs) and `n_collapsed`.
- `outputs/<model>/<tokenizer>/<task>/<split>/<method>/seed_<n>/` — Trainer cache +
  `finetuned_weights.pth`. Keyed on `(model, tokenizer)` so scratch capacity variants don't
  collide.
- `outputs/archives/` — pre-unified-pipeline CSVs (old schema, not comparable).

## Parallel execution on one GPU

Downstream tasks are tiny and use little GPU memory, so a single run leaves most
of an H100 idle. `(model, task, seed)` jobs are fully independent — run many at
once to fill the GPU. No SLURM needed; this is designed for a single screen session.

### 1. Profile one representative run first

Decide where the time goes before tuning `--jobs`.

```bash
# Wall-clock + peak CPU RSS for one job
/usr/bin/time -v python -m plm_benchmark.cli train \
    --task peta_gb1 --method full_ft --model esm2_8m --seed 42 --gpu 1

# In another shell, sample GPU memory/util every 0.5s while it runs, then take the max
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 | tee gpu.log
sort -t, -k1 -n gpu.log | tail -1   # peak memory (MiB) observed
```

Read it as: high **peak memory** → smaller `--jobs`; low **utilization.gpu** → the
GPU is idle and concurrency will help a lot. If per-epoch eval dominates (large
val/test sets over many epochs), also raise `--eval-every` and `--val-batch`.

Pick concurrency: `jobs ≈ 0.9 × 80GB ÷ peak_mem_per_run`, then cap by CPU cores.
For 8M/35M models this is usually 8–16, CPU-bound before memory-bound.

### 2. (Optional) Enable NVIDIA MPS

Default CUDA time-slices the GPU across processes; MPS lets their kernels run
concurrently, which helps when each job underutilizes the GPU (our case).

```bash
export CUDA_VISIBLE_DEVICES=0                       # the physical GPU to share
nvidia-cuda-mps-control -d                          # start the MPS daemon
# ... run the launcher ...
echo quit | nvidia-cuda-mps-control                 # stop it when done
```

Optionally cap per-client memory so one job can't starve the others:
`export CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=0=8G`.

### 3. Launch with bounded concurrency

`plm_benchmark/run_benchmark.py` fans `(model × tokenizer × task × split × seed)` out over a
thread pool of `--jobs` subprocesses, each a normal CLI `train` call. It lowers CPU threads
(`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`) and
sets `--num-workers 0` per job so K processes don't thrash the CPU. Per-job logs
go to `outputs/logs/`; `experiments.csv` is append-locked so parallel writes are safe.

```bash
# every task × every curated split, two PLMs
python -m plm_benchmark.run_benchmark \
    --models esm2_8m,esm2_35m --task everything --seeds 42,43,44 \
    --method full_ft --gpu 1 --jobs 8 --eval-every 5 --val-batch 64 \
    --wandb_project pulm-ft --wandb_group sweep1

# scratch baseline sweeping MULTIPLE tokenizers (comma-separated; scratch only)
python -m plm_benchmark.run_benchmark --models scratch \
    --tokenizer aa,/path/puma.json,/path/bpe.json \
    --task everything --scratch-layers 2 --gpu 1 --jobs 8
```

Tuning knobs:
- `--jobs` — concurrent processes (start at 8, watch `nvidia-smi`, raise until memory/CPU saturate).
- `--tokenizer` — comma-separated list to sweep (scratch only; non-scratch ignore it).
- `--split-method` — one split, `all`, or omit; groups always run every curated split.
- `--num-workers` — keep at 0–1 when `--jobs` is high.
- `--eval-every` — evaluate/checkpoint every N epochs (see below).
- `--val-batch` — larger eval/test batches use the H100 better (default 64).
- `--scratch-dim` / `--scratch-layers` / `--scratch-heads` — scratch-baseline capacity (see the scratch section).

> Job count multiplies fast: `everything` (37 task×split) × models × tokenizers × seeds. Check
> `len(jobs)` printed at launch before committing to a big grid.

### `embed_head` fast path: `--embed-cache`

A frozen encoder produces **identical** per-residue outputs every epoch, so recomputing them for
50 epochs is pure waste. `--embed-cache` computes them once and trains the attention1d head off the
cache, ~epochs-fold faster. Only `train`+`valid` are cached; test is always scored through the real
encoder, so the test evaluation itself is unchanged.

**Equivalence with the uncached path.** The cache is built under the *same* autocast dtype training
uses (bf16), and fp16 storage is exact for bf16 values (10 ≥ 7 mantissa bits), so the cached encoder
outputs match what the live path computes. Length grouping is preserved via an explicit
`LengthGroupedSampler` over the same per-sequence lengths, so batch composition matches too. Expect
cached and uncached runs to agree to within float noise — but they are not bit-guaranteed, so **pick
one and keep it consistent across an arm you're comparing** (don't cache one model's `embed_head` and
not another's). The frozen encoder is forced to `eval()` in both paths, so there is no dropout
nondeterminism either way.

```bash
python -m plm_benchmark.cli train --task peta_gb1 --method embed_head --model esm2_35m \
    --embed-cache --embed-cache-dir /scratch/emb --embed-cache-max-gb 40
```

- **Key**: `<cache-dir>/<model>/<tokenizer>/<task>/<split>/{train,valid}/` with a `meta.json`
  recording checkpoint + `max_length` + dtype; a mismatch rebuilds automatically.
- **Storage**: ragged (no padding waste) — a memmapped `[total_tokens, H]` fp16 array + offsets, so a
  20 GB cache never lands in RAM. fp16 (not bf16) is deliberate: it holds bf16 values exactly and is
  finer than the bf16 the head consumes. Its one risk is range (|x| > 65504), which is guarded — the
  build fails loudly rather than writing `inf`.
- **Budget**: `--embed-cache-max-gb` (default 50) — over budget it warns and encodes on the fly.
  Rough sizes (ESM2-35M, train+val): `gb1` ~0.2 GB, `meltome/human` ~4 GB, `aav/two_vs_many` ~23 GB,
  `aav/des_mut` ~142 GB (falls back).
- Applies only to `--method embed_head`; PPI tasks are unsupported (paired inputs) and fall back.

### Faster long runs: `--eval-every` and `--val-batch`

`--eval-every N` evaluates and checkpoints every N epochs instead of every epoch,
cutting large-val forward passes and checkpoint writes on long runs. Default
is 1 (unchanged behaviour). **Caveat:** early-stopping patience then counts *eval
events*, not epochs — with the default `patience=10` and `--eval-every 5`, that's 50
epochs of no-improvement before stopping, so lower `--patience` accordingly.
Coarser cadence also means the "best" checkpoint is chosen on a coarser grid, so
use it for sweeps/exploration and `--eval-every 1` for final numbers.

## Common commands

```bash
python -m plm_benchmark.cli train --task everything --method full_ft --model esm2_35m --seeds 42,43,44
python -m plm_benchmark.cli train --task peta_aav --method embed_head --model esm2_8m --split-method seven_vs_many
python -m plm_benchmark.cli list
python -m plm_benchmark.cli compare
```
