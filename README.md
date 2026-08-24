# PULM — Protein Units Language Modeling

PULM studies how the **tokenizer** shapes protein language models (PLMs). Standard
ESM-2 uses a single-residue (amino-acid) tokenizer; PULM trains and evaluates ESM-2-style
models under larger-vocabulary tokenizers (**BPE** and **PUMA**, multi-residue units) to
measure the effect of sub-word segmentation on protein modeling.

The repository covers the full lifecycle: build a tokenizer, pretrain (or adapt) a model
under it, then fine-tune and evaluate on downstream tasks.

## Components

| Directory | Purpose |
|-----------|---------|
| [`plm_train/`](plm_train/train.py) | Pretrain ESM-2-shaped models (masked-LM) **from scratch** under a chosen tokenizer (AA / BPE / PUMA). |
| [`plm_cont_train/`](plm_cont_train/train.py) | **Continual** pretraining: start from released ESM-2 weights and keep training, optionally after swapping the tokenizer. |
| [`plm_vocab_expansion/`](plm_vocab_expansion/README.md) | Cost-efficient adaptation of pretrained ESM-2 to a new, larger tokenizer without training from scratch. |
| [`plm_benchmark/`](plm_benchmark/README.md) | **Downstream fine-tuning & evaluation** — one unified PyTorch pipeline over the Rost and PETA task suites. See its README. |
| `scripts/` | Data prep (UniRef50 splits, PETA dataset setup). |
| `outputs/` | Run logs (`experiments.csv`, `comparison.csv`) and local weights. |

## Setup

```bash
conda env create -f environment.yml
conda activate finetune
```

## Downstream evaluation (quick start)

All fine-tuning is driven by the `plm_benchmark` CLI:

```bash
python -m plm_benchmark.cli list-tasks
python -m plm_benchmark.cli train --task GB1 --method full_ft --model esm2_8m --gpu 2
python -m plm_benchmark.cli compare
```

See [`plm_benchmark/README.md`](plm_benchmark/README.md) for the task suites, training
modes, the from-scratch tokenizer baseline, and wandb logging.

## Data

Two downstream task collections are used (see `plm_benchmark/README.md` for layout):

- **Rost / FLIP** — [RSchmirler/data-repo_plm-finetune-eval](https://github.com/RSchmirler/data-repo_plm-finetune-eval)
- **PETA** — [mingchen-li/ProteinPretraining](https://github.com/mingchen-li/ProteinPretraining)
