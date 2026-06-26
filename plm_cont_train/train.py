#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CONTINUAL pretraining (masked-LM) of ESM2 protein language models with a
*new* tokenizer.

Unlike `plm_train/train.py` (which trains an ESM2-shaped model FROM SCRATCH),
this script starts from the released ESM2 *weights* and continues training them
on a specialised corpus, optionally after swapping the tokenizer for a BPE or
PUMA tokenizer that introduces brand-new multi-residue tokens.

What is reused vs. learned
--------------------------
  * Transformer backbone : always inherited from the pretrained ESM2 checkpoint
                           (`EsmForMaskedLM.from_pretrained(...)`).
  * Vocabulary-dependent layers (input embeddings, output projection a.k.a.
    `lm_head.decoder`, and `lm_head.bias`) depend on the tokenizer and are
    handled with `--embedding_init`:
        transfer (default) : rows for tokens that ALSO exist in the ESM2
                             tokenizer (the 20 residues, ambiguity codes,
                             specials) keep their pretrained weights; the new
                             multi-residue tokens are seeded according to
                             `--new_token_init` (mean of their constituent
                             residue embeddings / random / copy <unk>).
        scratch            : the entire embedding + output projection is
                             re-initialised randomly; only the backbone is
                             continually trained.

Tokenizers (`--tokenizer_type`)
  * aa   : the real ESM2 residue tokenizer -> vocab identical to ESM2, so ALL
           weights are reused and this is plain continual pretraining.
  * bpe  : a HuggingFace `tokenizers` JSON.
  * puma : your PUMA `tokenizers` JSON.
  For bpe/puma the file is resolved from the naming convention (or pass
  --tokenizer_file). Special-token *strings* are read from the ESM2 tokenizer so
  they never drift from the original.

Training a new tokenizer onto a pretrained backbone is unstable if you update
the random new embeddings and the highly-tuned backbone together at a high LR
(catastrophic forgetting + garbage gradients from the new tokens). Two knobs
address this:
  * Low LR : single-stage continual training defaults to 2e-5 (NOT 1e-4, which
             is a from-scratch LR and wrecks the pretrained weights).
  * --two_stage : the recommended freeze-and-train strategy for bpe/puma:
        Stage 1 - freeze the ESM2 backbone, train ONLY the new vocab layers
                  (input embeddings + lm_head) so the new tokens align to the
                  frozen backbone (default: half the epochs, lr 1e-4).
                  This protects the pretrained attention maps.
        Stage 2 - unfreeze everything and fine-tune end-to-end at a much lower
                  LR (default 1e-5) for the remaining epochs.
        Two-stage runs append '_TS' to the auto run name.

Data        : --training_data selects the corpus -- 'human' (67k human set),
              'all' (full UniRef50 plm_train/plm_validation split tables built by
              scripts/prepare_uniref50_splits.py), or 'fasta'. All paths use
              on-the-fly random cropping to --max_length CORE tokens with
              ESM-2-style BOS/EOS boundary signalling (specials added only when
              the crop captures the protein's true start/end).
Hardware    : single- or multi-GPU (launch with `torchrun` for DDP).
Tracking    : Weights & Biases.
Sanity check: after training we reload BOTH the original pretrained ESM2 and the
              continually-trained model and compare their masked-residue
              predictions + pseudo-NLL-per-residue on probe sequences.

Examples
--------
    # Continue 35M ESM2 with a PUMA tokenizer (vocab 51200), transfer residue
    # weights and seed new tokens from their residue means:
    python train.py --data_source db --db_subset uniref50 \
        --model_size 35M --tokenizer_type puma \
        --subs_matrix blosum62 --mutation_cutoff 0.7 --min_mutation_freq 0.05 \
        --vocab_size 51200 --embedding_init transfer --new_token_init mean \
        --wandb_project protein-clm

    # Continue 8M ESM2 with the plain AA tokenizer (specialise on the corpus):
    python train.py --fasta data/human.fasta --model_size 8M --tokenizer_type aa \
        --wandb_project protein-clm

    # BPE tokenizer, learn the whole vocab layer from scratch:
    python train.py --fasta data/human.fasta --model_size 150M \
        --tokenizer_type bpe --vocab_size 6400 --embedding_init scratch

    # RECOMMENDED for a new tokenizer: two-stage freeze-and-train
    # (stage1 = frozen backbone @1e-4, stage2 = full model @1e-5):
    python train.py --data_source db --model_size 35M --tokenizer_type puma \
        --vocab_size 51200 --two_stage --num_train_epochs 6 \
        --stage1_ratio 0.5 --wandb_project protein-clm
"""

import argparse
import dataclasses
import inspect
import math
import os
import logging
import random
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np

# Server defaults: don't clobber values the user/torchrun already exported.
os.environ.setdefault("HF_HOME", "/cta/share/users/esm")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("cont_train_plm")


# ============================================================================= #
#  ESM2 REFERENCES
# ============================================================================= #
# Hub repos that provide both the backbone weights AND the canonical tokenizer.
ESM2_HF_NAMES: Dict[str, str] = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    # "3B": "facebook/esm2_t36_3B_UR50D", "15B": "facebook/esm2_t48_15B_UR50D"
}

# The ESM2 tokenizer is identical across sizes; use the smallest as the source
# of special-token strings and the residue alphabet.
REFERENCE_TOKENIZER = "facebook/esm2_t6_8M_UR50D"

# Exact ESM2 vocabulary (order matters -> ids match the real model). 33 tokens.
ESM2_VOCAB: List[str] = (
    ["<cls>", "<pad>", "<eos>", "<unk>"]
    + list("LAGVSERTIDPKQNFYMHWC")   # 20 standard residues, ESM order
    + list("XBUZO")                  # ambiguity / non-standard codes
    + [".", "-", "<null_1>", "<mask>"]
)

# Fallback special-token strings (verified to match ESM2).
ESM2_SPECIALS = {
    "cls_token":  "<cls>",
    "pad_token":  "<pad>",
    "eos_token":  "<eos>",
    "unk_token":  "<unk>",
    "mask_token": "<mask>",
}


# ============================================================================= #
#  PUMA / BPE TOKENIZER NAMING CONVENTION
#  Resolves a tokenizer file on the shared server from its spec, instead of
#  passing a raw path. Mirrors the user's generate_tokenizer_{name,filename}.
#
#  Spec fields are treated as STRINGS and used verbatim in the path, so e.g.
#  --min_mutation_freq 0 -> "_0_" (not "_0.0_"). Don't change them to floats.
#
#  Layout: {tokenizer_dir}/{subs_matrix}/hf_{filename}.json
#          (BPE always lives under the 'blosum62' subfolder.)
# ============================================================================= #
DEFAULT_TOKENIZER_DIR = "/cta/share/users/mutbpe/tokenizers"

# --- protein DB (SQLite) defaults ------------------------------------------- #
DEFAULT_DB_FILE = "/cta/share/users/uniprot/human/human.db"
# {subset} -> uniref50 / uniref90 (table name <subset>_distilled). Override the
# whole thing with --db_query for a different schema.
DEFAULT_DB_QUERY = (
    "SELECT Entry as uniprot_id, Sequence as sequence "
    "FROM proteins "
    "WHERE Entry IN (SELECT uniprot_accession FROM {subset}_distilled)"
)


def tokenizer_display_name(args) -> str:
    """Human-readable label, e.g. 'PUMA blosum62 0.7 0.05 6400' / 'BPE 6400'."""
    if args.tokenizer_type.lower() == "puma":
        return (f"PUMA {args.subs_matrix} {args.mutation_cutoff} "
                f"{args.min_mutation_freq} {args.vocab_size}")
    if args.tokenizer_type.lower() == "bpe":
        return f"BPE {args.vocab_size}"
    return "AA"


def _tokenizer_filename(args, is_mut: bool) -> str:
    if is_mut:
        fn = (f"{args.tok_dataset}_mutbpe_{args.mutation_cutoff}"
              f"_{args.min_mutation_len}_{args.max_mutation_len}"
              f"_{args.min_mutation_freq}_{args.vocab_size}")
    else:
        fn = f"{args.tok_dataset}_bpe_{args.vocab_size}"
    return fn


def resolve_tokenizer_path(args) -> str:
    """Build the full .json path from the spec, or use --tokenizer_file if given."""
    if args.tokenizer_file:                      # explicit override wins
        return args.tokenizer_file
    is_mut = args.tokenizer_type.lower() == "puma"
    subfolder = args.subs_matrix if is_mut else "bpe"
    fname = _tokenizer_filename(args, is_mut)
    return os.path.join(args.tokenizer_dir, subfolder, f"hf_{fname}.json")


def build_run_slug(args) -> str:
    """Auto name for output_dir / wandb run, e.g.
       ESM2_8M_PUMA_blosum62_07_005_51200 / ESM2_35M_BPE_51200 / ESM2_35M_AA.
       The two-stage freeze-and-train strategy appends '_TS'."""
    size = args.model_size or "custom"
    ttype = args.tokenizer_type.lower()
    nodot = lambda v: str(v).replace(".", "")   # 0.7 -> 07, 0.05 -> 005
    if ttype == "puma":
        tok = f"PUMA_{args.subs_matrix}_{nodot(args.mutation_cutoff)}_" \
              f"{nodot(args.min_mutation_freq)}_{args.vocab_size}"
    elif ttype == "bpe":
        tok = f"BPE_{args.vocab_size}"
    else:  # aa
        tok = "AA"
    slug = f"ESM2_{size}_{tok}_{args.embedding_init}"
    if getattr(args, "two_stage", False):
        slug += "_TS"   # Two-Stage freeze-and-train
    slug += f"_{args.training_data}"
    return slug


# ============================================================================= #
#  SPECIAL TOKENS  (sourced from the real ESM2 tokenizer when online)
# ============================================================================= #
def get_special_tokens(offline: bool) -> Dict[str, str]:
    if offline:
        return dict(ESM2_SPECIALS)
    try:
        from transformers import AutoTokenizer
        ref = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
        specials = {
            "cls_token":  ref.cls_token,
            "pad_token":  ref.pad_token,
            "eos_token":  ref.eos_token,
            "unk_token":  ref.unk_token,
            "mask_token": ref.mask_token,
        }
        if any(v is None for v in specials.values()):
            raise ValueError("ESM2 reference tokenizer missing a special token")
        logger.info("Special tokens sourced from %s: %s", REFERENCE_TOKENIZER, specials)
        return specials
    except Exception as e:  # noqa: BLE001
        logger.warning("Falling back to built-in ESM2 special tokens (%s)", e)
        return dict(ESM2_SPECIALS)


def load_reference_tokenizer(offline: bool):
    """The real ESM2 tokenizer (used for weight transfer + before/after probe)."""
    if not offline:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load ESM2 tokenizer from Hub (%s); "
                           "building an identical one offline.", e)
    return _build_aa_tokenizer_offline(1024)


# ============================================================================= #
#  TOKENIZER CONSTRUCTION
# ============================================================================= #
def build_tokenizer(args):
    """Return a HuggingFace tokenizer that wraps sequences as <cls> ... <eos>."""
    tok_type = args.tokenizer_type.lower()

    # ---- aa baseline: use the real ESM2 tokenizer verbatim ----------------- #
    if tok_type == "aa":
        if not args.offline:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
                tok.model_max_length = args.max_length
                logger.info("aa tokenizer == ESM2 tokenizer '%s' (vocab=%d)",
                            REFERENCE_TOKENIZER, len(tok))
                return tok
            except Exception as e:  # noqa: BLE001
                logger.warning("Could not load ESM2 tokenizer from Hub (%s); "
                               "building an identical one offline.", e)
        return _build_aa_tokenizer_offline(args.max_length)

    # ---- puma / bpe: load JSON, attach ESM2 specials + cls/eos template ---- #
    if tok_type in ("puma", "bpe"):
        path = resolve_tokenizer_path(args)
        logger.info("Tokenizer '%s' -> %s", tokenizer_display_name(args), path)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Tokenizer file not found: {path}\n"
                f"  (resolved from --tokenizer_type {tok_type} and its spec flags; "
                f"pass --tokenizer_file to override the path directly.)")
        return _build_json_tokenizer(path, args.max_length, args.offline)

    raise ValueError(f"Unknown --tokenizer_type: {tok_type!r}")


def _build_aa_tokenizer_offline(max_length: int):
    """Single-residue tokenizer with the exact ESM2 vocab/ids (no Hub needed)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, Regex
    from transformers import PreTrainedTokenizerFast

    vocab = {tok: i for i, tok in enumerate(ESM2_VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(""), behavior="isolated")
    _attach_template(backend, ESM2_SPECIALS)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=max_length, **ESM2_SPECIALS)
    logger.info("Built offline aa tokenizer matching ESM2 (vocab=%d)", len(tok))
    return tok


def _build_json_tokenizer(path: str, max_length: int, offline: bool):
    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast

    specials = get_special_tokens(offline)
    backend = Tokenizer.from_file(path)
    logger.info("Loaded tokenizer JSON '%s' (base vocab=%d)", path, backend.get_vocab_size())
    backend.add_special_tokens(list(specials.values()))   # appended at the end
    _attach_template(backend, specials)
    tok = PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=max_length, **specials)
    logger.info("Final tokenizer vocab (incl. specials): %d", len(tok))
    return tok


def _attach_template(backend, specials: Dict[str, str]):
    """Make every sequence become <cls> ... <eos>, like ESM2."""
    from tokenizers.processors import TemplateProcessing
    cls_t, eos_t = specials["cls_token"], specials["eos_token"]
    cls_id, eos_id = backend.token_to_id(cls_t), backend.token_to_id(eos_t)
    backend.post_processor = TemplateProcessing(
        single=f"{cls_t} $A {eos_t}",
        pair=f"{cls_t} $A {eos_t} $B:1 {eos_t}:1",
        special_tokens=[(cls_t, cls_id), (eos_t, eos_id)],
    )


# ============================================================================= #
#  FASTA / DB -> tokenized HuggingFace Dataset
# ============================================================================= #
def iter_fasta(path: str, min_length: int, max_samples: Optional[int]) -> Iterator[Dict[str, str]]:
    n = 0
    chunks: List[str] = []

    def flush():
        nonlocal n
        if not chunks:
            return None
        seq = "".join(chunks).upper().replace(" ", "")
        chunks.clear()
        if len(seq) >= min_length:
            n += 1
            return {"sequence": seq}
        return None

    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                rec = flush()
                if rec is not None:
                    yield rec
                    if max_samples and n >= max_samples:
                        return
            else:
                chunks.append(line)
        rec = flush()
        if rec is not None and not (max_samples and n > max_samples):
            yield rec


def load_human_sequences(args) -> List[str]:
    """Pull the 67k human protein sequences (uniref50_distilled) into a list."""
    import pandas as pd

    query = args.db_query or DEFAULT_DB_QUERY.format(subset=args.db_subset)
    logger.info("Querying human DB %s (subset=%s)", args.db_file, args.db_subset)
    conn = sqlite3.connect(args.db_file)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()

    if "sequence" not in df.columns:
        raise RuntimeError(
            f"DB query must return a 'sequence' column; got {list(df.columns)}. "
            f"Alias it in --db_query (e.g. 'SELECT Sequence as sequence ...').")

    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].str.upper().str.replace(" ", "", regex=False)
    df = df[df["sequence"].str.len() >= args.min_length]
    df = df[df["sequence"].str.len() < args.db_max_length]
    if args.max_samples:
        df = df.head(args.max_samples)
    logger.info("Loaded %d human sequences", len(df))
    return df["sequence"].tolist()


# ============================================================================= #
#  DYNAMIC-CROP DATASETS
#  On-the-fly random crop to `crop_length` CORE tokens (re-rolled every epoch),
#  with ESM-2-style boundary signalling: a BOS/EOS is added ONLY when the crop
#  actually captures the protein's physical start/end.
#    * full protein  (L <= W) : [BOS] + core + [EOS]            (<= W+2 tokens)
#    * crop @ start            : [BOS] + window                 (missed the end)
#    * crop @ end              :         window + [EOS]         (missed the start)
#    * internal crop           :         window                 (missed both ends)
#  Items are {'input_ids': [...]}; the MLM collator pads + builds labels. Because
#  the crop is random per access, length-grouping is disabled by the caller.
# ============================================================================= #
class _CropDatasetBase:
    def __init__(self, tokenizer, crop_length: int):
        self.tokenizer = tokenizer
        self.W = int(crop_length)
        self.bos = tokenizer.cls_token_id          # ESM2 BOS == <cls>
        self.eos = tokenizer.eos_token_id
        if self.bos is None or self.eos is None:
            raise ValueError("Tokenizer must define cls/eos tokens for BOS/EOS.")

    def encode(self, seq: str) -> Dict[str, List[int]]:
        core = self.tokenizer(seq, add_special_tokens=False,
                              truncation=False)["input_ids"]
        L = len(core)
        if L <= self.W:                            # full protein -> BOS..EOS
            ids = [self.bos] + core + [self.eos]
        else:                                      # random window of W tokens
            start = random.randint(0, L - self.W)
            ids = core[start:start + self.W]
            if start == 0:                         # captured the true start
                ids = [self.bos] + ids
            if start + self.W == L:                # captured the true end
                ids = ids + [self.eos]
        return {"input_ids": ids}


class InMemoryCropDataset(_CropDatasetBase):
    """For small corpora (e.g. the 67k human set) held in memory."""
    def __init__(self, sequences: List[str], tokenizer, crop_length: int):
        super().__init__(tokenizer, crop_length)
        self.sequences = sequences
        # residue length per example -> free, monotonic proxy for token length;
        # used by the length-grouped sampler (no tokenization needed).
        self.lengths = [len(s) for s in sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encode(self.sequences[idx])


class SqliteCropDataset(_CropDatasetBase):
    """Lazy reader over a split table (entry_id, sequence) for full UniRef50:
    sequences are fetched by rowid on demand, so the corpus never lives in RAM.
    A read-only sqlite connection is (re)opened per worker process."""
    def __init__(self, db_file: str, table: str, tokenizer, crop_length: int,
                 with_lengths: bool = False):
        super().__init__(tokenizer, crop_length)
        self.db_file, self.table = db_file, table
        con = sqlite3.connect(db_file)
        try:
            mn, mx, cnt = con.execute(
                f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table}").fetchone()
            if cnt == 0:
                raise RuntimeError(f"Split table {table!r} in {db_file} is empty.")
            self.n = cnt
            self.contiguous = (mn == 1 and mx == cnt)
            self.rowids = None
            if not self.contiguous:                # robust fallback
                self.rowids = np.fromiter(
                    (r[0] for r in con.execute(f"SELECT rowid FROM {table}")),
                    dtype=np.int64, count=cnt)
        finally:
            con.close()
        self._conn = None
        self._pid = None
        self.lengths = self._load_lengths() if with_lengths else None

    def _load_lengths(self):
        """Per-row residue length in rowid order (aligned with __getitem__), via a
        stored sequence_length column when present, else SQL length(sequence).
        Cached to a .npy next to the DB so it is a one-time scan."""
        cache = f"{self.db_file}.{self.table}.lengths.npy"
        if os.path.isfile(cache):
            try:
                arr = np.load(cache)
                if len(arr) == self.n:
                    return arr
            except Exception:  # noqa: BLE001
                pass
        con = sqlite3.connect(f"file:{self.db_file}?mode=ro", uri=True)
        try:
            cols = [r[1] for r in con.execute(f"PRAGMA table_info({self.table})")]
            expr = "sequence_length" if "sequence_length" in cols else "length(sequence)"
            logger.info("Building length index for %s.%s via %s (one-time scan) ...",
                        self.table, os.path.basename(self.db_file), expr)
            arr = np.fromiter((r[0] for r in con.execute(
                f"SELECT {expr} FROM {self.table}")), dtype=np.int64, count=self.n)
        finally:
            con.close()
        try:
            np.save(cache, arr)
        except Exception:  # noqa: BLE001
            pass
        return arr

    def _connection(self):
        pid = os.getpid()
        if self._conn is None or self._pid != pid:  # per-worker connection
            self._conn = sqlite3.connect(
                f"file:{self.db_file}?mode=ro", uri=True, check_same_thread=False)
            self._pid = pid
        return self._conn

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        rowid = idx + 1 if self.contiguous else int(self.rowids[idx])
        row = self._connection().execute(
            f"SELECT sequence FROM {self.table} WHERE rowid=?", (rowid,)).fetchone()
        return self.encode(row[0])


def get_dynamic_datasets(args, tokenizer):
    """Build (train_ds, eval_ds) torch datasets with dynamic cropping, selected
    by --training_data: 'all' -> UniRef50 split tables (lazy); 'human'/'fasta'
    -> in-memory corpus with a random --val_split holdout."""
    crop = args.max_length                          # crop window = CORE tokens

    if args.training_data == "all":
        logger.info("training_data=all -> UniRef50 split tables %s/%s in %s",
                    args.train_table, args.val_table, args.uniref_db)
        train_ds = SqliteCropDataset(args.uniref_db, args.train_table, tokenizer, crop,
                                     with_lengths=args.group_by_length)
        eval_ds = SqliteCropDataset(args.uniref_db, args.val_table, tokenizer, crop)
        logger.info("Lazy UniRef50: train=%d val=%d sequences", len(train_ds), len(eval_ds))
        return train_ds, eval_ds

    if args.training_data == "fasta":
        if not args.fasta:
            raise ValueError("--training_data fasta requires --fasta PATH")
        logger.info("Reading FASTA: %s", args.fasta)
        seqs = [r["sequence"] for r in iter_fasta(args.fasta, args.min_length, args.max_samples)]
        logger.info("Loaded %d sequences from FASTA", len(seqs))
    else:  # human
        seqs = load_human_sequences(args)

    if not seqs:
        raise RuntimeError("No sequences loaded for the in-memory pipeline.")

    if args.val_split and args.val_split > 0 and len(seqs) > 1:
        rng = random.Random(args.seed)
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        n_val = max(1, int(len(seqs) * args.val_split))
        val_seqs = [seqs[i] for i in idx[:n_val]]
        train_seqs = [seqs[i] for i in idx[n_val:]]
    else:
        train_seqs, val_seqs = seqs, []

    train_ds = InMemoryCropDataset(train_seqs, tokenizer, crop)
    eval_ds = InMemoryCropDataset(val_seqs, tokenizer, crop) if val_seqs else None
    logger.info("In-memory split (%s) -> train=%d val=%d",
                args.training_data, len(train_seqs), len(val_seqs))
    return train_ds, eval_ds


# ============================================================================= #
#  CONTINUAL MODEL  --  pretrained ESM2 weights, vocab layer adapted to our tok
# ============================================================================= #
def adapt_vocab_layers(model, ref_tokenizer, new_tokenizer, mode, new_token_init, seed):
    """Resize the model's embedding / output layers to the new tokenizer and fill
    them according to `mode`/`new_token_init`. The transformer backbone is left
    untouched (it keeps its pretrained weights either way).

    Returns a dict of statistics for logging.
    """
    import torch
    import torch.nn as nn

    g = torch.Generator().manual_seed(seed)
    std = getattr(model.config, "initializer_range", 0.02)

    # --- snapshot the pretrained vocab-dependent tensors, keyed by token string
    old_input = model.get_input_embeddings().weight.data.clone()  # (old_vocab, h)
    old_bias_param = getattr(getattr(model, "lm_head", None), "bias", None)
    old_bias = old_bias_param.data.clone() if old_bias_param is not None else None
    ref_vocab = ref_tokenizer.get_vocab()                         # str -> old id

    new_vocab = len(new_tokenizer)
    hidden = old_input.shape[1]

    # resize_token_embeddings creates correctly-shaped (input + tied decoder)
    # tensors; we then overwrite their contents row by row. It does NOT resize
    # ESM's separate lm_head.bias, so we rebuild that ourselves below.
    model.resize_token_embeddings(new_vocab)
    emb = model.get_input_embeddings().weight.data               # (new_vocab, h)

    inv_new = {idx: tok for tok, idx in new_tokenizer.get_vocab().items()}
    new_bias = torch.zeros(new_vocab, dtype=emb.dtype)

    stats = {"reused": 0, "mean_init": 0, "random_init": 0, "unk_init": 0,
             "new_vocab": new_vocab, "old_vocab": old_input.shape[0]}

    def rand_row():
        return torch.empty(hidden, dtype=emb.dtype).normal_(0.0, std, generator=g)

    if mode == "scratch":
        # Whole vocab layer learned from scratch; backbone stays pretrained.
        emb.copy_(torch.empty_like(emb).normal_(0.0, std, generator=g))
        new_bias.zero_()
        stats["random_init"] = new_vocab
    else:  # transfer
        unk_id = ref_vocab.get(ref_tokenizer.unk_token)
        for new_id in range(new_vocab):
            s = inv_new[new_id]
            if s in ref_vocab:                       # residue / ambiguity / special
                oid = ref_vocab[s]
                emb[new_id] = old_input[oid]
                if old_bias is not None:
                    new_bias[new_id] = old_bias[oid]
                stats["reused"] += 1
                continue
            # ---- brand-new multi-residue token ----
            if new_token_init == "mean":
                subs = [old_input[ref_vocab[c]] for c in s if c in ref_vocab]
                if subs:
                    emb[new_id] = torch.stack(subs).mean(0)
                    stats["mean_init"] += 1
                else:
                    emb[new_id] = rand_row(); stats["random_init"] += 1
            elif new_token_init == "unk" and unk_id is not None:
                emb[new_id] = old_input[unk_id]
                if old_bias is not None:
                    new_bias[new_id] = old_bias[unk_id]
                stats["unk_init"] += 1
            else:  # random
                emb[new_id] = rand_row(); stats["random_init"] += 1

    # rebuild lm_head.bias to the new size (decoder weight is tied to emb)
    if old_bias_param is not None:
        dev = old_bias_param.device
        model.lm_head.bias = nn.Parameter(new_bias.to(dev))
    model.tie_weights()

    # keep config in sync with the new tokenizer
    model.config.vocab_size = new_vocab
    model.config.pad_token_id = new_tokenizer.pad_token_id
    model.config.mask_token_id = new_tokenizer.mask_token_id
    model.config.bos_token_id = new_tokenizer.cls_token_id
    model.config.eos_token_id = new_tokenizer.eos_token_id
    return stats


def load_pretrained_esm(name: str, attn_impl: str):
    """from_pretrained with a requested attention impl; fall back to eager if the
    installed transformers/ESM build doesn't support it (e.g. sdpa)."""
    from transformers import EsmForMaskedLM
    try:
        m = EsmForMaskedLM.from_pretrained(name, attn_implementation=attn_impl)
        logger.info("Attention implementation: %s", attn_impl)
        return m
    except (ValueError, ImportError, RuntimeError) as e:
        if attn_impl == "eager":
            raise
        logger.warning("attn_implementation=%s unavailable (%s); falling back to "
                       "eager.", attn_impl, e)
        return EsmForMaskedLM.from_pretrained(name, attn_implementation="eager")


def build_model(args, tokenizer, ref_tokenizer):
    name = args.pretrained_name or ESM2_HF_NAMES.get(args.model_size)
    if not name:
        raise ValueError("Provide --model_size (8M/35M/150M/650M) or --pretrained_name.")
    logger.info("Loading pretrained ESM2 weights from %s", name)
    model = load_pretrained_esm(name, args.attn_implementation)

    # AA tokenizer == ESM2 tokenizer: identical vocab + id order, reuse as-is.
    same_as_esm = (args.tokenizer_type.lower() == "aa"
                   and len(tokenizer) == model.config.vocab_size)
    if same_as_esm:
        logger.info("AA tokenizer matches ESM2 vocab exactly (%d tokens); "
                    "reusing ALL pretrained weights (plain continual pretraining).",
                    len(tokenizer))
        # still align dropout if requested
        if args.dropout is not None:
            model.config.hidden_dropout_prob = args.dropout
            model.config.attention_probs_dropout_prob = args.dropout
        stats = {"reused": len(tokenizer), "mean_init": 0, "random_init": 0,
                 "unk_init": 0, "new_vocab": len(tokenizer),
                 "old_vocab": model.config.vocab_size}
    else:
        stats = adapt_vocab_layers(model, ref_tokenizer, tokenizer,
                                   args.embedding_init, args.new_token_init, args.seed)
        if args.dropout is not None:
            model.config.hidden_dropout_prob = args.dropout
            model.config.attention_probs_dropout_prob = args.dropout

    if args.max_length > model.config.max_position_embeddings - 2:
        logger.warning("--max_length=%d exceeds config capacity "
                       "(max_position_embeddings=%d). Reduce --max_length.",
                       args.max_length, model.config.max_position_embeddings)

    n = sum(p.numel() for p in model.parameters())
    logger.info(
        "Continual ESM2 [%s | %s] vocab %d->%d | embed-init=%s new-token-init=%s | "
        "rows: reused=%d mean=%d unk=%d random=%d | %.1fM params",
        args.model_size or "custom", name, stats["old_vocab"], stats["new_vocab"],
        args.embedding_init, args.new_token_init, stats["reused"], stats["mean_init"],
        stats["unk_init"], stats["random_init"], n / 1e6)
    return model, stats


# ============================================================================= #
#  VERSION-ROBUST TrainingArguments
#  Field names drift across transformers releases; map known renames then drop
#  anything the installed version doesn't accept.
# ============================================================================= #
def make_training_args(TrainingArguments, kwargs: dict):
    valid = {f.name for f in dataclasses.fields(TrainingArguments) if f.init}
    kwargs = dict(kwargs)

    if "group_by_length" in kwargs and "group_by_length" not in valid:
        grouped = kwargs.pop("group_by_length")
        if "train_sampling_strategy" in valid:
            kwargs["train_sampling_strategy"] = "group_by_length" if grouped else "random"
    if "train_sampling_strategy" in kwargs and "train_sampling_strategy" not in valid:
        strat = kwargs.pop("train_sampling_strategy")
        if "group_by_length" in valid:
            kwargs["group_by_length"] = (strat == "group_by_length")

    if "eval_strategy" in kwargs and "eval_strategy" not in valid and \
            "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    for k in [k for k in list(kwargs) if k not in valid]:
        logger.warning("Dropping TrainingArguments kwarg unsupported by this "
                       "transformers version: %s", k)
        kwargs.pop(k)
    return TrainingArguments(**kwargs)


# ============================================================================= #
#  PRECISION / WARMUP
# ============================================================================= #
def resolve_precision(choice: str):
    import torch
    if choice == "fp16":
        return True, False
    if choice == "bf16":
        return False, True
    if choice == "no":
        return False, False
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return False, True
    return torch.cuda.is_available(), False


def compute_warmup_steps(args, n_train: int, num_epochs: Optional[float] = None) -> int:
    if args.warmup_steps and args.warmup_steps > 0:
        return args.warmup_steps
    if not args.warmup_ratio or args.warmup_ratio <= 0:
        return 0
    num_epochs = args.num_train_epochs if num_epochs is None else num_epochs
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")
    eff_batch = max(1, args.per_device_train_batch_size *
                    args.gradient_accumulation_steps * world)
    if args.max_steps and args.max_steps > 0:
        total = args.max_steps
    else:
        steps_per_epoch = math.ceil(n_train / eff_batch)
        total = int(steps_per_epoch * num_epochs)
    steps = max(1, int(args.warmup_ratio * total))
    logger.info("warmup: ratio %.3f x ~%d total steps -> %d warmup steps "
                "(world_size=%d)", args.warmup_ratio, total, steps, world)
    return steps


# ============================================================================= #
#  TWO-STAGE FREEZE-AND-TRAIN
#  Stage 1: freeze the pretrained transformer backbone and train ONLY the new
#           vocabulary layers (input embeddings + lm_head). This lets the freshly
#           seeded token embeddings align to the frozen backbone's expectations
#           without corrupting the pretrained attention maps.
#  Stage 2: unfreeze everything and fine-tune end-to-end at a much lower LR.
# ============================================================================= #
def set_backbone_trainable(model, trainable: bool):
    """Freeze/unfreeze the transformer backbone, always keeping the
    vocabulary-dependent layers (input embeddings + lm_head) trainable."""
    for p in model.parameters():
        p.requires_grad = trainable
    if not trainable:
        for p in model.get_input_embeddings().parameters():
            p.requires_grad = True
        if getattr(model, "lm_head", None) is not None:
            for p in model.lm_head.parameters():
                p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info("Backbone %s | trainable params: %.1fM / %.1fM (%.1f%%)",
                "UNFROZEN" if trainable else "FROZEN (vocab layers only)",
                n_train / 1e6, n_all / 1e6, 100.0 * n_train / max(1, n_all))


# ============================================================================= #
#  ARGUMENTS
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(
        description="Continually pretrain a pretrained ESM2 model with a new tokenizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    p.add_argument("--training_data", default=None, choices=["human", "all", "fasta"],
                   help="Training corpus: 'human' = 67k human set (human DB, in "
                        "memory); 'all' = full UniRef50 split tables (lazy); "
                        "'fasta' = a FASTA file. Default: derived from --data_source.")
    p.add_argument("--data_source", default="fasta", choices=["fasta", "db"],
                   help="[legacy] fasta/db; only used when --training_data is unset "
                        "(db -> human, fasta -> fasta).")
    p.add_argument("--fasta", default=None, help="FASTA path (--training_data fasta).")
    p.add_argument("--db_file", default=DEFAULT_DB_FILE,
                   help="Human SQLite DB (--training_data human).")
    p.add_argument("--db_subset", default="uniref50", choices=["uniref50", "uniref90"],
                   help="Selects the <subset>_distilled table in the default human query.")
    p.add_argument("--db_query", default=None,
                   help="Custom SQL overriding the default human query; must return a 'sequence' column.")
    p.add_argument("--db_max_length", type=int, default=1000,
                   help="Keep human-DB sequences with residue length < this.")
    # --- full UniRef50 split tables (--training_data all) --- #
    p.add_argument("--uniref_db",
                   default="/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db",
                   help="SQLite DB holding the plm_train/plm_validation split tables.")
    p.add_argument("--train_table", default="plm_train",
                   help="Train split table (entry_id, sequence) for --training_data all.")
    p.add_argument("--val_table", default="plm_validation",
                   help="Validation split table for --training_data all.")
    p.add_argument("--min_length", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024,
                   help="Random-crop window in CORE tokens (BOS/EOS added on top; "
                        "ESM2 = 1024, fits the model's 1026 positions).")
    p.add_argument("--val_split", type=float, default=0.01,
                   help="Holdout fraction for in-memory corpora (human/fasta); "
                        "ignored for 'all' (uses the plm_validation table).")
    p.add_argument("--max_samples", type=int, default=None, help="Cap #sequences (debug).")
    p.add_argument("--preprocessing_num_workers", type=int, default=4)

    # tokenizer
    p.add_argument("--tokenizer_type", required=True, choices=["puma", "bpe", "aa"],
                   help="puma/bpe -> resolve JSON from spec; aa -> ESM2 residue tokenizer.")
    p.add_argument("--tokenizer_file", default=None,
                   help="Explicit JSON path; overrides the naming-convention resolver.")
    p.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR,
                   help="Shared folder holding the tokenizer JSONs.")
    p.add_argument("--tok_dataset", default="uniref50", choices=["uniref50", "uniref90"],
                   help="Corpus the tokenizer was trained on (filename component).")
    p.add_argument("--subs_matrix", default="blosum62",
                   choices=["blosum45", "blosum62", "pam70", "pam250"],
                   help="PUMA substitution matrix (subfolder + filename; PUMA only).")
    p.add_argument("--mutation_cutoff", default="0.7",
                   help="PUMA mutation cutoff, e.g. 0.7/0.8/0.9 (string, used verbatim).")
    p.add_argument("--min_mutation_freq", default="0.05",
                   help="PUMA min mutation freq, e.g. 0/0.005/0.05/0.1/0.2 (verbatim).")
    p.add_argument("--min_mutation_len", default="3", help="PUMA min mutation length.")
    p.add_argument("--max_mutation_len", default="12", help="PUMA max mutation length.")
    p.add_argument("--vocab_size", default="6400",
                   help="Tokenizer vocab size in the filename "
                        "(800/1600/3200/6400/12800/25600/51200). NOT the model vocab, "
                        "which is derived from the loaded tokenizer.")

    # model / continual-training behaviour
    p.add_argument("--model_size", default="8M",
                   choices=list(ESM2_HF_NAMES.keys()) + [""],
                   help="Preset ESM2 size whose pretrained weights to continue.")
    p.add_argument("--pretrained_name", default=None,
                   help="Override the Hub repo / local dir of the pretrained ESM2.")
    p.add_argument("--embedding_init", default="transfer",
                   choices=["transfer", "scratch"],
                   help="transfer: keep pretrained weights for tokens shared with "
                        "ESM2, seed new tokens; scratch: re-init the whole vocab layer.")
    p.add_argument("--new_token_init", default="mean",
                   choices=["mean", "random", "unk"],
                   help="How to seed NEW multi-residue tokens in transfer mode: "
                        "mean of constituent residue embeddings / random / copy <unk>.")
    p.add_argument("--dropout", type=float, default=None,
                   help="Override hidden/attention dropout (default: keep ESM2's).")
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"],
                   help="Attention kernel; sdpa is faster + lower memory (auto-"
                        "falls back to eager if unsupported).")
    p.add_argument("--offline", action="store_true",
                   help="Don't touch the Hub; use cached weights + built-in specials.")

    # MLM
    p.add_argument("--mlm_probability", type=float, default=0.15)

    # optimisation (continual-training-flavoured defaults: MUCH lower LR than
    # scratch -- a high LR on the pretrained backbone causes catastrophic
    # forgetting, so single-stage continual training defaults to 2e-5).
    p.add_argument("--per_device_train_batch_size", type=int, default=128)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=float, default=3.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="HF optimizer (e.g. adamw_torch_fused, adamw_torch, "
                        "adamw_bnb_8bit). Fused is faster on CUDA.")

    # two-stage freeze-and-train (recommended for new bpe/puma tokenizers)
    p.add_argument("--two_stage", action="store_true",
                   help="Stage 1: freeze backbone, train ONLY the new vocab layers; "
                        "Stage 2: unfreeze and fine-tune end-to-end at a lower LR. "
                        "Appends '_TS' to the auto run name.")
    p.add_argument("--stage1_ratio", type=float, default=0.5,
                   help="Fraction of --num_train_epochs spent in the frozen stage 1 "
                        "(ignored if --stage1_epochs is set).")
    p.add_argument("--stage1_epochs", type=float, default=None,
                   help="Explicit epoch count for stage 1 (overrides --stage1_ratio).")
    p.add_argument("--stage1_learning_rate", type=float, default=1e-4,
                   help="LR for stage 1 (only the new vocab layers train, so this "
                        "can be high -- they start random).")
    p.add_argument("--stage2_learning_rate", type=float, default=1e-5,
                   help="LR for stage 2 end-to-end fine-tuning (keep it low to avoid "
                        "catastrophic forgetting of the pretrained backbone).")

    # hardware / efficiency
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "no"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--group_by_length", action="store_true", default=True,
                   help="Length-grouped batching via known residue lengths "
                        "(cuts padding waste; no tokenization pass).")
    p.add_argument("--no_group_by_length", dest="group_by_length", action="store_false")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--torch_compile", action="store_true")

    # logging / checkpointing
    p.add_argument("--output_dir", default=None,
                   help="Defaults to the auto run name, e.g. ESM2_8M_PUMA_blosum62_07_005_51200.")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Checkpoint dir to resume from, or 'true'/'latest' to pick "
                        "the newest checkpoint in --output_dir.")
    p.add_argument("--seed", type=int, default=42)

    # post-training sanity check (before vs after)
    p.add_argument("--sanity_check", action="store_true", default=True,
                   help="After training, compare the original ESM2 and the trained model.")
    p.add_argument("--no_sanity_check", dest="sanity_check", action="store_false")
    p.add_argument("--sanity_topk", type=int, default=10)
    p.add_argument("--sanity_sequences", default=None,
                   help="Comma-separated probe sequences (default: two human proteins).")

    # wandb
    p.add_argument("--wandb_project", default=None, help="Enables wandb when set.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None, help="Group runs (e.g. by model size).")
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    p.add_argument("--wandb_run_id", default=None,
                   help="Continue THIS existing wandb run id (use together with "
                        "--resume_from_checkpoint to merge into the same run).")
    p.add_argument("--wandb_resume", default="allow",
                   choices=["allow", "must", "never", "auto"],
                   help="wandb resume mode when --wandb_run_id is set.")

    return p.parse_args()


# ============================================================================= #
#  POST-TRAINING SANITY CHECK  (before vs after)
# ============================================================================= #
DEFAULT_PROBES = [
    "MLKAAAKRPELSGKNTISNNSDMAEVKSMFREVLPKQGPLFVEDIMTMVLCKPKLLPLKSLTLEKLEKMHQAAQNTIRQQEMAEKDQRQITH",  # A8MTZ0
    "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD",  # A6NNB3
]


def _masked_topk(model, tok, seq, topk, max_length, device):
    """Mask the middle non-special token and return (true_tok, [(tok,prob)...])."""
    import torch
    enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)
    ids = enc["input_ids"][0]
    special = set(tok.all_special_ids)
    positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
    if not positions:
        return None, []
    pos = positions[len(positions) // 2]
    true_tok = tok.convert_ids_to_tokens([ids[pos].item()])[0]
    masked = ids.clone()
    masked[pos] = tok.mask_token_id
    with torch.no_grad():
        logits = model(masked.unsqueeze(0).to(device)).logits[0, pos]
    probs = logits.float().softmax(-1)
    top = torch.topk(probs, k=min(topk, probs.numel()))
    pairs = list(zip(tok.convert_ids_to_tokens(top.indices.tolist()), top.values.tolist()))
    return true_tok, pairs


def _pseudo_nll_per_residue(model, tok, seqs, max_length, device):
    """Mask each non-special token in turn; return mean NLL of the true token
    divided by the number of residues it spans (bits-per-residue, comparable
    across tokenizers regardless of how many tokens a sequence splits into)."""
    import torch
    special = set(tok.all_special_ids)
    total_nll, total_res = 0.0, 0
    for seq in seqs:
        enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"][0]
        positions = [i for i, t in enumerate(ids.tolist()) if t not in special]
        for pos in positions:
            true = ids[pos].item()
            masked = ids.clone()
            masked[pos] = tok.mask_token_id
            with torch.no_grad():
                logits = model(masked.unsqueeze(0).to(device)).logits[0, pos]
            logp = logits.float().log_softmax(-1)[true].item()
            total_nll += -logp
            total_res += len(tok.convert_ids_to_tokens([true])[0])  # residues in token
    if total_res == 0:
        return float("nan")
    return total_nll / total_res


def sanity_check(orig_name, trained_dir, sequences, topk, max_length, offline):
    """Load the ORIGINAL pretrained ESM2 and the TRAINED model and compare."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 78)
    print("POST-TRAINING SANITY CHECK  (before = original ESM2  vs  after = trained)")
    print("=" * 78)

    def load(path):
        tok = AutoTokenizer.from_pretrained(path)
        mdl = AutoModelForMaskedLM.from_pretrained(path).to(device).eval()
        return tok, mdl

    logger.info("Loading ORIGINAL %s and TRAINED %s on %s", orig_name, trained_dir, device)
    orig_tok, orig_mdl = load(orig_name)
    new_tok, new_mdl = load(trained_dir)

    for label, (tok, mdl) in [("BEFORE", (orig_tok, orig_mdl)),
                              ("AFTER ", (new_tok, new_mdl))]:
        ppr = _pseudo_nll_per_residue(mdl, tok, sequences, max_length, device)
        print(f"\n[{label}]  vocab={len(tok):>6d}  "
              f"pseudo-NLL/residue={ppr:.4f}  (lower = better)")

    for i, seq in enumerate(sequences):
        print(f"\n--- probe {i + 1}: {seq[:48]}{'...' if len(seq) > 48 else ''}")
        for label, (tok, mdl) in [("BEFORE", (orig_tok, orig_mdl)),
                                  ("AFTER ", (new_tok, new_mdl))]:
            true_tok, pairs = _masked_topk(mdl, tok, seq, topk, max_length, device)
            if true_tok is None:
                print(f"  [{label}] (no maskable tokens)")
                continue
            in_top = any(t == true_tok for t, _ in pairs)
            preds = ", ".join(f"{t}:{p:.2f}" for t, p in pairs[:topk])
            print(f"  [{label}] masked true='{true_tok}' | true-in-top{topk}={in_top}")
            print(f"           top-{topk}: {preds}")

    print("\n" + "=" * 78)
    print("Note: BEFORE/AFTER use different tokenizers, so masked positions and the")
    print("token inventory differ. pseudo-NLL/residue normalises by residues spanned")
    print("and is the most comparable single number; top-k is a qualitative check.\n")


# ============================================================================= #
#  RESUME FIX
#  In recent transformers, TrainerState stores logging_steps/eval_steps/
#  save_steps, and `TrainerState.load_from_json()` on resume OVERWRITES the
#  values you pass on the CLI with the ones baked into the checkpoint. So after
#  resuming, the cadence silently reverts to the checkpoint's old schedule.
#  on_train_begin runs AFTER the state is loaded -> re-assert the current args.
# ============================================================================= #
def make_resume_step_override_callback():
    from transformers import TrainerCallback

    class ResumeStepOverride(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            state.logging_steps = args.logging_steps
            state.eval_steps = args.eval_steps
            state.save_steps = args.save_steps
            logger.info("Re-asserted schedule after resume: logging_steps=%s "
                        "eval_steps=%s save_steps=%s",
                        args.logging_steps, args.eval_steps, args.save_steps)
            return control

    return ResumeStepOverride()


# ============================================================================= #
#  LENGTH-GROUPED SAMPLER
#  Batches similar-length sequences together to cut padding waste, driven by the
#  dataset's *known* residue lengths (dataset.lengths) -- so no tokenization pass
#  is needed (works for the lazy UniRef50 dataset). Replaces HF's group_by_length,
#  whose default path would tokenize the whole corpus to derive lengths. Dynamic
#  random cropping is preserved; only the index order changes.
# ============================================================================= #
def build_length_grouped_sampler(trainer, enabled):
    lengths = getattr(trainer.train_dataset, "lengths", None)
    if not enabled or lengths is None:
        return None
    try:
        from transformers.trainer_pt_utils import (
            LengthGroupedSampler, DistributedLengthGroupedSampler)
    except Exception:  # noqa: BLE001
        logger.warning("LengthGroupedSampler unavailable; using the default sampler.")
        return None
    mega = trainer.args.train_batch_size * trainer.args.gradient_accumulation_steps
    world = getattr(trainer.args, "world_size", 1) or 1
    if world <= 1:
        return LengthGroupedSampler(mega, lengths=lengths)
    return DistributedLengthGroupedSampler(
        mega, num_replicas=world, rank=trainer.args.process_index, lengths=lengths)


def make_trainer_class(group_by_length):
    from transformers import Trainer

    class LengthGroupedTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):
            s = build_length_grouped_sampler(self, group_by_length)
            return s if s is not None else super()._get_train_sampler(*a, **k)

    return LengthGroupedTrainer


# ============================================================================= #
#  TRAINER FACTORY  (shared by single-stage and the two-stage strategy)
# ============================================================================= #
def build_trainer(args, model, tokenizer, train_ds, eval_ds, collator, *,
                  learning_rate, num_train_epochs, output_dir, run_name,
                  report_to, load_best):
    from transformers import Trainer, TrainingArguments

    fp16, bf16 = resolve_precision(args.precision)
    do_eval = eval_ds is not None
    warmup_steps = compute_warmup_steps(args, len(train_ds), num_train_epochs)

    ta_kwargs = dict(
        output_dir=output_dir, seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=num_train_epochs, max_steps=args.max_steps,
        learning_rate=learning_rate, weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon, max_grad_norm=args.max_grad_norm,
        warmup_steps=warmup_steps, lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        fp16=fp16, bf16=bf16, gradient_checkpointing=args.gradient_checkpointing,
        dataloader_num_workers=args.dataloader_num_workers,
        torch_compile=args.torch_compile,
        do_eval=do_eval, eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps, logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=load_best,
        metric_for_best_model="eval_loss" if load_best else None,
        greater_is_better=False, report_to=report_to, run_name=run_name,
        ddp_find_unused_parameters=False,
    )
    training_args = make_training_args(TrainingArguments, ta_kwargs)

    # Trainer renamed `tokenizer` -> `processing_class` in transformers 4.46+.
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds,
                          eval_dataset=eval_ds, data_collator=collator,
                          callbacks=[make_resume_step_override_callback()])
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    # length grouping handled by our sampler (known lengths), not TrainingArguments.
    return make_trainer_class(args.group_by_length)(**trainer_kwargs)


def log_adapt_stats(args, trainer, adapt_stats):
    """Record the embedding-adaptation breakdown to the active wandb run."""
    if not (args.wandb_project and trainer.is_world_process_zero()):
        return
    try:
        import wandb
        if wandb.run is not None:
            wandb.config.update({f"adapt/{k}": v for k, v in adapt_stats.items()},
                                allow_val_change=True)
    except Exception:  # noqa: BLE001
        pass


def finish_wandb():
    """Close the current wandb run so the next stage starts a fresh one."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass


# ============================================================================= #
#  MAIN
# ============================================================================= #
def main():
    args = parse_args()
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from transformers import DataCollatorForLanguageModeling, set_seed
    set_seed(args.seed)

    # resolve the training corpus (back-compat: derive from legacy --data_source)
    if args.training_data is None:
        args.training_data = "fasta" if args.data_source == "fasta" else "human"
        logger.info("--training_data not set -> '%s' (from --data_source=%s)",
                    args.training_data, args.data_source)
    if args.training_data == "fasta" and not args.fasta:
        raise ValueError("--training_data fasta requires --fasta PATH")
    if args.group_by_length:
        logger.info("Length-grouped batching ON (known residue lengths; dynamic "
                    "cropping preserved).")

    # normalise --resume_from_checkpoint: accept 'true'/'latest'/'auto' -> True
    # (let the Trainer auto-pick the newest checkpoint in --output_dir).
    if isinstance(args.resume_from_checkpoint, str) and \
            args.resume_from_checkpoint.lower() in {"true", "1", "latest", "auto"}:
        args.resume_from_checkpoint = True

    slug = build_run_slug(args)
    if not args.output_dir:
        args.output_dir = slug
        logger.info("output_dir not given -> using '%s'", args.output_dir)

    # --- wandb wiring ------------------------------------------------------- #
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_MODE"] = args.wandb_mode
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        # Continue the SAME wandb run when given an id (merge with predecessor).
        if args.wandb_run_id:
            os.environ["WANDB_RUN_ID"] = args.wandb_run_id
            os.environ["WANDB_RESUME"] = args.wandb_resume
            logger.info("Continuing wandb run id=%s (resume=%s)",
                        args.wandb_run_id, args.wandb_resume)
        elif args.resume_from_checkpoint:
            logger.warning("Resuming training but --wandb_run_id was not given: "
                           "wandb will start a NEW run. Pass --wandb_run_id <id> "
                           "to append to the original run.")
        report_to = ["wandb"]
        run_name = args.wandb_run_name or slug
    else:
        report_to, run_name = ["none"], None

    # --- build the pieces --------------------------------------------------- #
    ref_tokenizer = load_reference_tokenizer(args.offline)   # real ESM2 tokenizer
    tokenizer = build_tokenizer(args)
    train_ds, eval_ds = get_dynamic_datasets(args, tokenizer)
    model, adapt_stats = build_model(args, tokenizer, ref_tokenizer)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8)

    do_eval = eval_ds is not None
    common = dict(args=args, model=model, tokenizer=tokenizer, train_ds=train_ds,
                  eval_ds=eval_ds, collator=collator, report_to=report_to)

    # ------------------------------------------------------------------ #
    #  TWO-STAGE freeze-and-train  (stage 1: vocab layers only, frozen
    #  backbone; stage 2: unfreeze, fine-tune end-to-end at a lower LR)
    # ------------------------------------------------------------------ #
    if args.two_stage:
        if args.resume_from_checkpoint:
            logger.warning("--resume_from_checkpoint is ignored in --two_stage mode.")
        if args.wandb_run_id:                       # one id can't span 2 stage runs
            logger.warning("--wandb_run_id is ignored in --two_stage mode "
                           "(each stage is its own run).")
            os.environ.pop("WANDB_RUN_ID", None)
            os.environ.pop("WANDB_RESUME", None)
        total = args.num_train_epochs
        s1 = args.stage1_epochs if args.stage1_epochs is not None else args.stage1_ratio * total
        s1 = max(0.0, min(round(s1, 6), total))
        s2 = max(0.0, round(total - s1, 6))
        if args.wandb_project and not os.environ.get("WANDB_RUN_GROUP"):
            os.environ["WANDB_RUN_GROUP"] = slug    # group both stages together

        trainer = None
        # ---- Stage 1: frozen backbone, vocab layers only ---- #
        if s1 > 0:
            set_backbone_trainable(model, False)
            logger.info("=== STAGE 1/2: frozen backbone, vocab layers only "
                        "(%.3g epochs @ lr=%g) ===", s1, args.stage1_learning_rate)
            trainer = build_trainer(
                **common, learning_rate=args.stage1_learning_rate,
                num_train_epochs=s1,
                output_dir=os.path.join(args.output_dir, "stage1"),
                run_name=(run_name + "_s1") if run_name else None,
                load_best=False)
            log_adapt_stats(args, trainer, adapt_stats)
            trainer.train()
            finish_wandb()          # so stage 2 logs to its own run

        # ---- Stage 2: unfreeze, fine-tune end-to-end ---- #
        if s2 > 0:
            set_backbone_trainable(model, True)
            logger.info("=== STAGE 2/2: full model end-to-end "
                        "(%.3g epochs @ lr=%g) ===", s2, args.stage2_learning_rate)
            trainer = build_trainer(
                **common, learning_rate=args.stage2_learning_rate,
                num_train_epochs=s2,
                output_dir=os.path.join(args.output_dir, "stage2"),
                run_name=(run_name + "_s2") if run_name else None,
                load_best=do_eval)
            if s1 <= 0:
                log_adapt_stats(args, trainer, adapt_stats)
            trainer.train()

        if trainer is None:
            raise ValueError("Two-stage training ran zero epochs; check "
                             "--num_train_epochs / --stage1_ratio.")
    # ------------------------------------------------------------------ #
    #  SINGLE-STAGE continual training (whole model, low LR)
    # ------------------------------------------------------------------ #
    else:
        trainer = build_trainer(
            **common, learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
            output_dir=args.output_dir, run_name=run_name, load_best=do_eval)
        log_adapt_stats(args, trainer, adapt_stats)
        logger.info("Starting single-stage continual training (lr=%g) ...",
                    args.learning_rate)
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = os.path.join(args.output_dir, "final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved final model + tokenizer to %s", final_dir)

    if do_eval:
        metrics = trainer.evaluate()
        try:
            metrics["perplexity"] = math.exp(metrics["eval_loss"])
        except (KeyError, OverflowError, ValueError):
            pass
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # --- post-training sanity check (main process only) -------------------- #
    if args.sanity_check and trainer.is_world_process_zero():
        probes = ([s.strip() for s in args.sanity_sequences.split(",") if s.strip()]
                  if args.sanity_sequences else DEFAULT_PROBES)
        orig_name = args.pretrained_name or ESM2_HF_NAMES.get(args.model_size)
        try:
            sanity_check(orig_name, final_dir, probes, args.sanity_topk,
                         args.max_length, args.offline)
        except Exception as e:  # noqa: BLE001
            logger.warning("Sanity check failed (training still succeeded): %s", e)


if __name__ == "__main__":
    main()
