#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Pretrain ESM2-style protein language models (masked LM) FROM SCRATCH.

Maximum-compatibility design: the backbone config and the special tokens are
pulled from the *real* ESM2 checkpoints on the HuggingFace Hub, so the models
you train differ from the originals only where you intend them to:

  * Architecture : `EsmConfig.from_pretrained(facebook/esm2_...)` -> exact layer
                   count, hidden size, heads, FFN (4x hidden), RoPE,
                   token_dropout, max_position_embeddings=1026, etc.
                   We override ONLY `vocab_size` and the special-token ids to
                   match the tokenizer you chose.
  * Tokenizer    : - char  : the ESM2 tokenizer itself (a single-residue
                             tokenizer) -> byte-identical to ESM2.
                   - puma  : your PUMA `tokenizers` JSON.
                   - bpe   : any HuggingFace `tokenizers` JSON.
                   For puma/bpe the special-token *strings* are read from the
                   ESM2 tokenizer so they never drift from the original.
  * Data         : --training_data selects 'human' (67k human set), 'all' (full
                   UniRef50 plm_train/plm_validation split tables from
                   scripts/prepare_uniref50_splits.py), or 'fasta'. All use
                   on-the-fly random cropping to --max_length CORE tokens with
                   ESM-2-style BOS/EOS boundary signalling.
  * Hardware     : single- or multi-GPU (launch with `torchrun` for DDP).
  * Tracking     : Weights & Biases.

Switch experiments with three flags: --model_size, --tokenizer_type,
--tokenizer_file. Everything else has sane ESM2-flavoured defaults.

Examples
--------
    # PUMA tokenizer, 35M model
    python train_plm.py --fasta data/uniref50.fasta \
        --tokenizer_type puma --tokenizer_file puma_tokenizer.json \
        --model_size 35M --output_dir runs/esm2_35M_puma \
        --wandb_project protein-lm --per_device_train_batch_size 64

    # Single-amino-acid tokenizer (== ESM2 tokenizer), 8M, quick smoke test
    python train_plm.py --fasta data/sample.fasta --tokenizer_type aa \
        --model_size 8M --output_dir runs/dbg --max_samples 2000 --max_steps 50

    # Multi-GPU (Trainer auto-detects DDP)
    torchrun --nproc_per_node=4 train_plm.py ...

    # Air-gapped GPU node (no Hub access): add --offline to use the built-in
    # ESM2 shape table + vocab instead of downloading.
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["HF_HOME"] = "/cta/share/users/esm"

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger("train_plm")


# ============================================================================= #
#  ESM2 REFERENCES
# ============================================================================= #
# Hub repos whose config.json defines each backbone. We read the architecture
# from these instead of hardcoding it.
ESM2_HF_NAMES: Dict[str, str] = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
    # "3B": "facebook/esm2_t36_3B_UR50D", "15B": "facebook/esm2_t48_15B_UR50D"
}

# The ESM2 tokenizer is identical across all sizes; use the smallest as the
# canonical source for special-token strings and the residue alphabet.
REFERENCE_TOKENIZER = "facebook/esm2_t6_8M_UR50D"

# --- OFFLINE fallbacks (used only with --offline or if the Hub is unreachable) --

# Canonical ESM2 backbone shapes (heads = 20 for every released size; FFN = 4*hidden).
@dataclass
class EsmShape:
    num_hidden_layers: int
    hidden_size: int
    num_attention_heads: int
    intermediate_size: int


ESM2_SHAPES: Dict[str, EsmShape] = {
    "8M":   EsmShape(6,   320, 20, 1280),
    "35M":  EsmShape(12,  480, 20, 1920),
    "150M": EsmShape(30,  640, 20, 2560),
    "650M": EsmShape(33, 1280, 20, 5120),
}

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
    subfolder = args.subs_matrix if is_mut else "blosum62"
    fname = _tokenizer_filename(args, is_mut)
    return os.path.join(args.tokenizer_dir, subfolder, f"hf_{fname}.json")


def build_run_slug(args) -> str:
    """Auto name for output_dir / wandb run, e.g.
       ESM2_8M_PUMA_blosum62_07_005_51200 / ESM2_35M_BPE_51200 / ESM2_35M_AA."""
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
    return f"ESM2_{size}_{tok}"


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
    from tokenizers.processors import TemplateProcessing
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
#  FASTA -> tokenized HuggingFace Dataset
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

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encode(self.sequences[idx])


class SqliteCropDataset(_CropDatasetBase):
    """Lazy reader over a split table (entry_id, sequence) for full UniRef50:
    sequences are fetched by rowid on demand, so the corpus never lives in RAM.
    A read-only sqlite connection is (re)opened per worker process."""
    def __init__(self, db_file: str, table: str, tokenizer, crop_length: int):
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
        train_ds = SqliteCropDataset(args.uniref_db, args.train_table, tokenizer, crop)
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
#  MODEL  --  ESM2 config from the Hub, vocab swapped to our tokenizer
# ============================================================================= #
def _manual_config(args):
    """Offline / custom EsmConfig builder (mirrors ESM2 defaults)."""
    from transformers import EsmConfig
    if args.model_size and args.model_size in ESM2_SHAPES:
        s = ESM2_SHAPES[args.model_size]
        layers, hidden, heads, ffn = (
            s.num_hidden_layers, s.hidden_size, s.num_attention_heads, s.intermediate_size)
        max_pos = 1026  # ESM2 value
    else:
        layers, hidden, heads = args.num_hidden_layers, args.hidden_size, args.num_attention_heads
        ffn = args.intermediate_size or 4 * hidden
        max_pos = args.max_length + 2
    return EsmConfig(
        hidden_size=hidden, num_hidden_layers=layers, num_attention_heads=heads,
        intermediate_size=ffn, max_position_embeddings=max_pos,
        position_embedding_type="rotary", emb_layer_norm_before=False,
        token_dropout=True, hidden_dropout_prob=args.dropout,
        attention_probs_dropout_prob=args.dropout, layer_norm_eps=1e-5,
    )


def build_model(args, tokenizer):
    from transformers import EsmConfig, EsmForMaskedLM

    ref = ESM2_HF_NAMES.get(args.model_size) if args.model_size else None
    config_name = args.esm_config_name or ref

    if config_name and not args.offline:
        try:
            config = EsmConfig.from_pretrained(config_name)
            source = f"from_pretrained({config_name})"
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not fetch ESM2 config (%s); using offline shape table.", e)
            config, source = _manual_config(args), "offline-shape-table"
    else:
        config, source = _manual_config(args), "custom/offline"

    # ---- override ONLY what depends on our tokenizer ----------------------- #
    config.vocab_size   = len(tokenizer)
    config.pad_token_id = tokenizer.pad_token_id
    config.mask_token_id = tokenizer.mask_token_id
    config.bos_token_id = tokenizer.cls_token_id
    config.eos_token_id = tokenizer.eos_token_id
    if args.dropout is not None:
        config.hidden_dropout_prob = args.dropout
        config.attention_probs_dropout_prob = args.dropout

    # sanity: make sure positions cover our sequences
    if args.max_length > config.max_position_embeddings - 2:
        logger.warning("--max_length=%d exceeds config capacity (max_position_embeddings"
                       "=%d). Reduce --max_length or it will error at runtime.",
                       args.max_length, config.max_position_embeddings)

    model = EsmForMaskedLM(config)   # random init -> trained from scratch
    n = sum(p.numel() for p in model.parameters())
    logger.info("ESM2 [%s | config=%s]: layers=%d hidden=%d heads=%d ffn=%d "
                "max_pos=%d vocab=%d -> %.1fM params",
                args.model_size or "custom", source, config.num_hidden_layers,
                config.hidden_size, config.num_attention_heads,
                config.intermediate_size, config.max_position_embeddings,
                config.vocab_size, n / 1e6)
    return model


# ============================================================================= #
#  VERSION-ROBUST TrainingArguments
#  Field names drift across transformers releases (e.g. the boolean
#  `group_by_length` became `train_sampling_strategy="group_by_length"` in v5).
#  Map known renames, then drop anything the installed version doesn't accept.
# ============================================================================= #
def make_training_args(TrainingArguments, kwargs: dict):
    valid = {f.name for f in dataclasses.fields(TrainingArguments) if f.init}
    kwargs = dict(kwargs)

    # group_by_length (bool, <=v4)  <->  train_sampling_strategy (str, >=v5)
    if "group_by_length" in kwargs and "group_by_length" not in valid:
        grouped = kwargs.pop("group_by_length")
        if "train_sampling_strategy" in valid:
            kwargs["train_sampling_strategy"] = "group_by_length" if grouped else "random"
    if "train_sampling_strategy" in kwargs and "train_sampling_strategy" not in valid:
        strat = kwargs.pop("train_sampling_strategy")
        if "group_by_length" in valid:
            kwargs["group_by_length"] = (strat == "group_by_length")

    # eval_strategy (>=4.41)  <->  evaluation_strategy (older)
    if "eval_strategy" in kwargs and "eval_strategy" not in valid and \
            "evaluation_strategy" in valid:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    dropped = [k for k in list(kwargs) if k not in valid]
    for k in dropped:
        logger.warning("Dropping TrainingArguments kwarg unsupported by this "
                       "transformers version: %s", k)
        kwargs.pop(k)
    return TrainingArguments(**kwargs)


# ============================================================================= #
#  PRECISION
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


def compute_warmup_steps(args, n_train: int) -> int:
    """Resolve warmup to an absolute step count (warmup_ratio is deprecated in
    transformers and we never pass it through). Explicit --warmup_steps wins;
    otherwise convert --warmup_ratio against the estimated total step count."""
    if args.warmup_steps and args.warmup_steps > 0:
        return args.warmup_steps
    if not args.warmup_ratio or args.warmup_ratio <= 0:
        return 0
    world = int(os.environ.get("WORLD_SIZE", "1") or "1")        # set by torchrun
    eff_batch = max(1, args.per_device_train_batch_size *
                    args.gradient_accumulation_steps * world)
    if args.max_steps and args.max_steps > 0:
        total = args.max_steps
    else:
        steps_per_epoch = math.ceil(n_train / eff_batch)
        total = int(steps_per_epoch * args.num_train_epochs)
    steps = max(1, int(args.warmup_ratio * total))
    logger.info("warmup: ratio %.3f x ~%d total steps -> %d warmup steps "
                "(world_size=%d)", args.warmup_ratio, total, steps, world)
    return steps


# ============================================================================= #
#  ARGUMENTS
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(
        description="Pretrain an ESM2 protein LM from scratch.",
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
    # --- naming-convention resolver (puma/bpe) -- values used VERBATIM in paths --
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

    # model
    p.add_argument("--model_size", default="8M",
                   choices=list(ESM2_HF_NAMES.keys()) + [""],
                   help="Preset ESM2 size; '' to use the custom flags below.")
    p.add_argument("--esm_config_name", default=None,
                   help="Override the Hub repo whose config to inherit.")
    p.add_argument("--num_hidden_layers", type=int, default=6)
    p.add_argument("--hidden_size", type=int, default=320)
    p.add_argument("--num_attention_heads", type=int, default=20)
    p.add_argument("--intermediate_size", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--offline", action="store_true",
                   help="Don't touch the Hub; use built-in ESM2 shapes/vocab.")

    # MLM
    p.add_argument("--mlm_probability", type=float, default=0.15)

    # optimisation (ESM2-flavoured defaults)
    p.add_argument("--per_device_train_batch_size", type=int, default=128)
    p.add_argument("--per_device_eval_batch_size", type=int, default=128)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_ratio", type=float, default=0.02)
    p.add_argument("--warmup_steps", type=int, default=0)
    p.add_argument("--lr_scheduler_type", default="cosine")

    # hardware / efficiency
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "no"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--group_by_length", action="store_true", default=True)
    p.add_argument("--no_group_by_length", dest="group_by_length", action="store_false")
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--torch_compile", action="store_true")

    # logging / checkpointing
    p.add_argument("--output_dir", default=None,
                   help="Defaults to the auto run name, e.g. ESM2_8M_PUMA_blosum62_07_005_51200.")
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=500)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=1)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--seed", type=int, default=42)

    # post-training sanity check
    p.add_argument("--sanity_check", action="store_true", default=True,
                   help="After training, reload final/ and run a masked-residue test.")
    p.add_argument("--no_sanity_check", dest="sanity_check", action="store_false")
    p.add_argument("--sanity_topk", type=int, default=10)
    p.add_argument("--sanity_sequences", default=None,
                   help="Comma-separated test sequences (default: ubiquitin + a fragment).")

    # wandb
    p.add_argument("--wandb_project", default=None, help="Enables wandb when set.")
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None, help="Group runs (e.g. by model size).")
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])

    return p.parse_args()


# ============================================================================= #
#  POST-TRAINING SANITY CHECK
#  Reload from disk (validates save/load), mask one residue per test sequence,
#  and report the model's top-k predictions at that position.
# ============================================================================= #
# A couple of well-known small proteins as default probes (human ubiquitin +
# insulin A-chain). For a fully trained model the masked residue should appear
# in the top-k; for a 50-step smoke test this just confirms the plumbing works.
# DEFAULT_PROBES = [
#     "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
#     "GIVEQCCTSICSLYQLENYCN",
# ]

DEFAULT_PROBES = [
    "MLKAAAKRPELSGKNTISNNSDMAEVKSMFREVLPKQGPLFVEDIMTMVLCKPKLLPLKSLTLEKLEKMHQAAQNTIRQQEMAEKDQRQITH", # A8MTZ0
    "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD", # A6NNB3
]


def sanity_check(model_dir, sequences, topk, max_length):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Sanity check: reloading from %s on %s", model_dir, device)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForMaskedLM.from_pretrained(model_dir).to(device).eval()

    if tok.mask_token_id is None:
        logger.error("Tokenizer has no mask token - cannot run MLM check.")
        return

    special_ids = set(tok.all_special_ids)
    print("\n" + "=" * 70)
    print("POST-TRAINING SANITY CHECK  (masked-residue prediction)")
    print("=" * 70)

    for seq in sequences:
        enc = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)
        ids = enc["input_ids"][0]
        positions = [i for i, t in enumerate(ids.tolist()) if t not in special_ids]
        if not positions:
            print(f"  (skipped, no maskable tokens) {seq[:30]}...")
            continue

        pos = positions[len(positions) // 2]          # mask a middle token
        original_id = ids[pos].item()
        original_tok = tok.convert_ids_to_tokens([original_id])[0]

        masked = ids.clone()
        masked[pos] = tok.mask_token_id
        with torch.no_grad():
            logits = model(input_ids=masked.unsqueeze(0).to(device)).logits[0, pos]
        probs = logits.float().softmax(-1)
        top = torch.topk(probs, k=min(topk, probs.numel()))
        top_tokens = tok.convert_ids_to_tokens(top.indices.tolist())
        top_probs = top.values.tolist()

        in_topk = original_tok in top_tokens
        # "sensible" = predictions are real (non-special) sequence tokens
        sensible = all(t not in tok.all_special_tokens for t in top_tokens)

        print(f"\n  seq  : {seq[:48]}{'...' if len(seq) > 48 else ''}")
        print(f"  masked token #{pos} (true = '{original_tok}')")
        preds = ", ".join(f"{t}:{p:.2f}" for t, p in zip(top_tokens, top_probs))
        print(f"  top-{topk}: {preds}")
        print(f"  -> true-in-top{topk}: {in_topk} | predictions-are-residues: {sensible}")

    print("=" * 70)
    print("Plumbing OK if predictions are residue tokens (not <pad>/<unk>/etc.).")
    print("Accuracy is only meaningful after real training, not a smoke test.\n")


# ============================================================================= #
#  MAIN
# ============================================================================= #
def main():
    args = parse_args()
    from transformers import (DataCollatorForLanguageModeling, Trainer,
                              TrainingArguments, set_seed)
    set_seed(args.seed)

    # --- resolve training corpus (back-compat: derive from legacy --data_source)
    if args.training_data is None:
        args.training_data = "fasta" if args.data_source == "fasta" else "human"
        logger.info("--training_data not set -> '%s' (from --data_source=%s)",
                    args.training_data, args.data_source)
    if args.training_data == "fasta" and not args.fasta:
        raise ValueError("--training_data fasta requires --fasta PATH")
    # dynamic random cropping makes precomputed lengths meaningless and a lazy
    # full pass over UniRef50 untenable -> disable length grouping.
    if args.group_by_length:
        logger.info("Disabling group_by_length (incompatible with dynamic cropping).")
        args.group_by_length = False

    # auto run name, e.g. ESM2_8M_PUMA_blosum62_07_005_51200 / ESM2_35M_AA
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
        report_to = ["wandb"]
        run_name = args.wandb_run_name or slug
    else:
        report_to, run_name = ["none"], None

    # --- build the three pieces -------------------------------------------- #
    tokenizer = build_tokenizer(args)
    train_ds, eval_ds = get_dynamic_datasets(args, tokenizer)
    model = build_model(args, tokenizer)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8)

    fp16, bf16 = resolve_precision(args.precision)
    do_eval = eval_ds is not None
    warmup_steps = compute_warmup_steps(args, len(train_ds))

    ta_kwargs = dict(
        output_dir=args.output_dir, seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1, adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon, max_grad_norm=args.max_grad_norm,
        warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        fp16=fp16, bf16=bf16, gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length, length_column_name="length",
        dataloader_num_workers=args.dataloader_num_workers,
        torch_compile=args.torch_compile,
        do_eval=do_eval, eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps, logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=do_eval,
        metric_for_best_model="eval_loss" if do_eval else None,
        greater_is_better=False, report_to=report_to, run_name=run_name,
        ddp_find_unused_parameters=False,
    )
    training_args = make_training_args(TrainingArguments, ta_kwargs)

    # Trainer renamed `tokenizer` -> `processing_class` in transformers 4.46+.
    trainer_kwargs = dict(model=model, args=training_args, train_dataset=train_ds,
                          eval_dataset=eval_ds, data_collator=collator)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = Trainer(**trainer_kwargs)

    logger.info("Starting training ...")
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
        try:
            sanity_check(final_dir, probes, args.sanity_topk, args.max_length)
        except Exception as e:  # noqa: BLE001
            logger.warning("Sanity check failed (training still succeeded): %s", e)


if __name__ == "__main__":
    main()