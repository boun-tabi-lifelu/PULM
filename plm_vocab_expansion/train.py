#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Cost-efficient adaptation of a pretrained ESM-2 to a new tokenizer (AA / BPE /
PUMA) WITHOUT training from scratch -- the "vocab-expansion" pipeline.

This implements `puma_vocab_expansion_methodology_v3.md` end to end. The
transformer backbone is frozen; only the parts that genuinely break when the
tokenizer changes are re-initialised analytically or adapted with small
low-rank adapters:

  Step 1  Embedding init   : new multi-residue tokens = mean of their constituent
                             AA embeddings, projected onto the AA norm shell
                             (e = ē/‖ē‖ · mean_AA_norm). Specials/singles are
                             copied. PUMA *children* additionally blend with the
                             already-initialised parent before projecting
                             (e = shell(γ·e_pool + (1-γ)·e_parent)).
  Step 2  LM head          : inner Linear+LayerNorm kept; output projection tied
                             to the embedding (transpose); bias = smoothed log-
                             frequency prior, with PUMA family-shrinkage for
                             children.
  Step 3  RoPE             : residue-based position IDs (no base change). A token
                             at token-index i gets the residue offset of its
                             first residue; AA collapses to sequential IDs.
  Step 4  LoRA             : freeze the body, inject LoRA on Q/K/V/W_o/FFN by a
                             preset (uniform_r16 / layerwise / attn_only_r16).
  Step 5  Staged training  : (1) embeddings-only warm-up, (2) LoRA+head+emb,
                             (3) + LayerNorms. PUMA family-alignment regulariser
                             (cosine, table-wide) in Stage 1 only.
  Step 6  Mask rate        : kept at 0.15 (residue density is k̄-invariant).

The AA tokenizer collapses to standard fine-tuning (k̄=1) and the PUMA-only
extensions activate only where genealogy exists, so this one pipeline covers all
three tokenizers. Notation / infra (tokenizer naming, dynamic crop dataset,
wandb, resume fix, sdpa/optim) follow the sibling plm_train / plm_cont_train.

Requires: torch, transformers, datasets, peft, numpy, wandb (optional).

Example
-------
    python train.py --tokenizer_type puma --vocab_size 1600 \
        --subs_matrix blosum62 --mutation_cutoff 0.7 --min_mutation_freq 0.05 \
        --model_size 35M --lora_preset layerwise --training_data all \
        --wandb_project plm-vocab-expansion
"""

import argparse
import glob
import inspect
import json
import math
import os
import logging
import random
import sqlite3
import threading
from typing import Dict, List, Optional

import numpy as np

os.environ.setdefault("HF_HOME", "/cta/share/users/esm")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("vocab_expansion")


# ============================================================================= #
#  ESM2 REFERENCES
# ============================================================================= #
ESM2_HF_NAMES: Dict[str, str] = {
    "8M":   "facebook/esm2_t6_8M_UR50D",
    "35M":  "facebook/esm2_t12_35M_UR50D",
    "150M": "facebook/esm2_t30_150M_UR50D",
    "650M": "facebook/esm2_t33_650M_UR50D",
}
REFERENCE_TOKENIZER = "facebook/esm2_t6_8M_UR50D"

# ESM2 standard residue alphabet (the AA token set we mean-pool from / measure
# the norm shell over). Order matches ESM2 but only identity matters here.
ESM2_AA_LETTERS = list("LAGVSERTIDPKQNFYMHWC") + list("XBUZO")

ESM2_VOCAB: List[str] = (
    ["<cls>", "<pad>", "<eos>", "<unk>"]
    + list("LAGVSERTIDPKQNFYMHWC") + list("XBUZO")
    + [".", "-", "<null_1>", "<mask>"])
ESM2_SPECIALS = {
    "cls_token": "<cls>", "pad_token": "<pad>", "eos_token": "<eos>",
    "unk_token": "<unk>", "mask_token": "<mask>"}

DEFAULT_TOKENIZER_DIR = "/cta/share/users/mutbpe/tokenizers"
DEFAULT_DB_FILE = "/cta/share/users/uniprot/human/human.db"
DEFAULT_UNIREF_DB = ("/cta/share/users/uniprot/uniref/uniref_2024_06/"
                     "uniref50_representatives.db")
DEFAULT_DB_QUERY = (
    "SELECT Entry as uniprot_id, Sequence as sequence FROM proteins "
    "WHERE Entry IN (SELECT uniprot_accession FROM {subset}_distilled)")


# ============================================================================= #
#  TOKENIZER NAMING CONVENTION  (mirrors the sibling pipelines)
# ============================================================================= #
def tokenizer_display_name(args) -> str:
    if args.tokenizer_type.lower() == "puma":
        return (f"PUMA {args.subs_matrix} {args.mutation_cutoff} "
                f"{args.min_mutation_freq} {args.vocab_size}")
    if args.tokenizer_type.lower() == "bpe":
        return f"BPE {args.vocab_size}"
    return "AA"


def _tokenizer_filename(args, is_mut: bool) -> str:
    if is_mut:
        return (f"{args.tok_dataset}_mutbpe_{args.mutation_cutoff}"
                f"_{args.min_mutation_len}_{args.max_mutation_len}"
                f"_{args.min_mutation_freq}_{args.vocab_size}")
    return f"{args.tok_dataset}_bpe_{args.vocab_size}"


def resolve_tokenizer_paths(args):
    """Return (hf_json_path, attributes_json_path). The attributes file is the
    same name WITHOUT the 'hf_' prefix and carries PUMA family metadata."""
    if args.tokenizer_file:
        hf = args.tokenizer_file
        d, base = os.path.split(hf)
        attr = os.path.join(d, base[3:]) if base.startswith("hf_") else None
        return hf, attr
    is_mut = args.tokenizer_type.lower() == "puma"
    subfolder = args.subs_matrix if is_mut else "blosum62"
    fname = _tokenizer_filename(args, is_mut)
    folder = os.path.join(args.tokenizer_dir, subfolder)
    return os.path.join(folder, f"hf_{fname}.json"), os.path.join(folder, f"{fname}.json")


def build_run_slug(args) -> str:
    size = args.model_size or "custom"
    ttype = args.tokenizer_type.lower()
    nodot = lambda v: str(v).replace(".", "")
    if ttype == "puma":
        tok = (f"PUMA_{args.subs_matrix}_{nodot(args.mutation_cutoff)}_"
               f"{nodot(args.min_mutation_freq)}_{args.vocab_size}")
    elif ttype == "bpe":
        tok = f"BPE_{args.vocab_size}"
    else:
        tok = "AA"
    return f"ESM2_{size}_{tok}_VE_{args.lora_preset}"


# ============================================================================= #
#  SPECIAL TOKENS + TOKENIZER CONSTRUCTION
# ============================================================================= #
def get_special_tokens(offline: bool) -> Dict[str, str]:
    if not offline:
        try:
            from transformers import AutoTokenizer
            ref = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
            sp = {"cls_token": ref.cls_token, "pad_token": ref.pad_token,
                  "eos_token": ref.eos_token, "unk_token": ref.unk_token,
                  "mask_token": ref.mask_token}
            if all(v is not None for v in sp.values()):
                return sp
        except Exception as e:  # noqa: BLE001
            logger.warning("Falling back to built-in ESM2 specials (%s)", e)
    return dict(ESM2_SPECIALS)


def load_reference_tokenizer(offline: bool):
    if not offline:
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not load ESM2 tokenizer (%s); building offline.", e)
    return _build_aa_tokenizer_offline(1024)


def _attach_template(backend, specials):
    from tokenizers.processors import TemplateProcessing
    cls_t, eos_t = specials["cls_token"], specials["eos_token"]
    backend.post_processor = TemplateProcessing(
        single=f"{cls_t} $A {eos_t}",
        pair=f"{cls_t} $A {eos_t} $B:1 {eos_t}:1",
        special_tokens=[(cls_t, backend.token_to_id(cls_t)),
                        (eos_t, backend.token_to_id(eos_t))])


def _build_aa_tokenizer_offline(max_length: int):
    from tokenizers import Tokenizer, models, pre_tokenizers, Regex
    from transformers import PreTrainedTokenizerFast
    vocab = {t: i for i, t in enumerate(ESM2_VOCAB)}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Split(pattern=Regex(""), behavior="isolated")
    _attach_template(backend, ESM2_SPECIALS)
    return PreTrainedTokenizerFast(tokenizer_object=backend,
                                   model_max_length=max_length, **ESM2_SPECIALS)


def build_tokenizer(args):
    tok_type = args.tokenizer_type.lower()
    if tok_type == "aa":
        if not args.offline:
            try:
                from transformers import AutoTokenizer
                tok = AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
                tok.model_max_length = args.max_length
                return tok
            except Exception as e:  # noqa: BLE001
                logger.warning("Hub AA tokenizer failed (%s); building offline.", e)
        return _build_aa_tokenizer_offline(args.max_length)

    from tokenizers import Tokenizer
    from transformers import PreTrainedTokenizerFast
    hf_path, _ = resolve_tokenizer_paths(args)
    if not os.path.isfile(hf_path):
        raise FileNotFoundError(f"Tokenizer file not found: {hf_path}")
    logger.info("Tokenizer '%s' -> %s", tokenizer_display_name(args), hf_path)
    specials = get_special_tokens(args.offline)
    backend = Tokenizer.from_file(hf_path)
    backend.add_special_tokens(list(specials.values()))
    _attach_template(backend, specials)
    tok = PreTrainedTokenizerFast(tokenizer_object=backend,
                                  model_max_length=args.max_length, **specials)
    logger.info("Final tokenizer vocab (incl. specials): %d", len(tok))
    return tok


# ============================================================================= #
#  PUMA GENEALOGY  (parsed from the attributes JSON, the non-'hf_' file)
# ============================================================================= #
def load_genealogy(args):
    """Return {token_str: {'is_child', 'parent', 'similarity'}} or {} if absent.
    BPE files have no parent/is_parent keys -> every multi-residue unit is flat."""
    import json
    _, attr_path = resolve_tokenizer_paths(args)
    if not attr_path or not os.path.isfile(attr_path):
        if args.tokenizer_type.lower() == "puma":
            logger.warning("PUMA attributes file not found (%s); treating every "
                           "unit as flat (no lineage extensions).", attr_path)
        return {}
    with open(attr_path) as fh:
        raw = json.load(fh)
    gen = {}
    for tok, meta in raw.items():
        if "parent" in meta:
            gen[tok] = {"is_child": True, "parent": meta["parent"],
                        "similarity": float(meta.get("similarity", 1.0))}
    logger.info("Loaded PUMA genealogy from %s: %d children", attr_path, len(gen))
    return gen


# ============================================================================= #
#  DYNAMIC-CROP DATASETS  (token-space crop, ESM2 boundary signalling)
#  Identical contract to the sibling pipelines; the model derives residue
#  position IDs internally, so items carry only input_ids.
# ============================================================================= #
def iter_fasta(path, min_length, max_samples):
    n, chunks = 0, []
    def flush():
        nonlocal n
        if not chunks:
            return None
        seq = "".join(chunks).upper().replace(" ", ""); chunks.clear()
        if len(seq) >= min_length:
            n += 1
            return seq
        return None
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                s = flush()
                if s is not None:
                    yield s
                    if max_samples and n >= max_samples:
                        return
            else:
                chunks.append(line)
        s = flush()
        if s is not None:
            yield s


def load_human_sequences(args) -> List[str]:
    import pandas as pd
    query = args.db_query or DEFAULT_DB_QUERY.format(subset=args.db_subset)
    conn = sqlite3.connect(args.db_file)
    try:
        df = pd.read_sql(query, conn)
    finally:
        conn.close()
    df = df.dropna(subset=["sequence"]).copy()
    df["sequence"] = df["sequence"].str.upper().str.replace(" ", "", regex=False)
    df = df[df["sequence"].str.len() >= args.min_length]
    df = df[df["sequence"].str.len() < args.db_max_length]
    if args.max_samples:
        df = df.head(args.max_samples)
    return df["sequence"].tolist()


class _CropDatasetBase:
    def __init__(self, tokenizer, crop_length):
        self.tokenizer = tokenizer
        self.W = int(crop_length)
        self.bos = tokenizer.cls_token_id
        self.eos = tokenizer.eos_token_id

    def encode(self, seq):
        core = self.tokenizer(seq, add_special_tokens=False, truncation=False)["input_ids"]
        L = len(core)
        if L <= self.W:
            ids = [self.bos] + core + [self.eos]
        else:
            start = random.randint(0, L - self.W)
            ids = core[start:start + self.W]
            if start == 0:
                ids = [self.bos] + ids
            if start + self.W == L:
                ids = ids + [self.eos]
        return {"input_ids": ids}


class InMemoryCropDataset(_CropDatasetBase):
    def __init__(self, sequences, tokenizer, crop_length):
        super().__init__(tokenizer, crop_length)
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.encode(self.sequences[idx])


class SqliteCropDataset(_CropDatasetBase):
    def __init__(self, db_file, table, tokenizer, crop_length):
        super().__init__(tokenizer, crop_length)
        self.db_file, self.table = db_file, table
        con = sqlite3.connect(db_file)
        try:
            mn, mx, cnt = con.execute(
                f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table}").fetchone()
        finally:
            con.close()
        if not cnt:
            raise RuntimeError(f"Split table {table!r} in {db_file} is empty.")
        self.n, self.contiguous, self.rowids = cnt, (mn == 1 and mx == cnt), None
        if not self.contiguous:
            con = sqlite3.connect(db_file)
            self.rowids = np.fromiter((r[0] for r in con.execute(
                f"SELECT rowid FROM {table}")), dtype=np.int64, count=cnt)
            con.close()
        self._conn, self._pid = None, None

    def _connection(self):
        pid = os.getpid()
        if self._conn is None or self._pid != pid:
            self._conn = sqlite3.connect(f"file:{self.db_file}?mode=ro", uri=True,
                                         check_same_thread=False)
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
    crop = args.max_length
    if args.training_data == "all":
        train_ds = SqliteCropDataset(args.uniref_db, args.train_table, tokenizer, crop)
        eval_ds = SqliteCropDataset(args.uniref_db, args.val_table, tokenizer, crop)
        logger.info("UniRef50 (lazy): train=%d val=%d", len(train_ds), len(eval_ds))
        return train_ds, eval_ds
    if args.training_data == "fasta":
        if not args.fasta:
            raise ValueError("--training_data fasta requires --fasta PATH")
        seqs = list(iter_fasta(args.fasta, args.min_length, args.max_samples))
    else:
        seqs = load_human_sequences(args)
    if not seqs:
        raise RuntimeError("No sequences loaded.")
    if args.val_split and args.val_split > 0 and len(seqs) > 1:
        rng = random.Random(args.seed)
        idx = list(range(len(seqs)))
        rng.shuffle(idx)
        nv = max(1, int(len(seqs) * args.val_split))
        val_seqs = [seqs[i] for i in idx[:nv]]
        train_seqs = [seqs[i] for i in idx[nv:]]
    else:
        train_seqs, val_seqs = seqs, []
    train_ds = InMemoryCropDataset(train_seqs, tokenizer, crop)
    eval_ds = InMemoryCropDataset(val_seqs, tokenizer, crop) if val_seqs else None
    logger.info("In-memory (%s): train=%d val=%d", args.training_data,
                len(train_seqs), len(val_seqs))
    return train_ds, eval_ds


def sample_sequences(args, n):
    """A bounded sample of raw sequences for the LM-head frequency prior."""
    if args.training_data == "all":
        con = sqlite3.connect(f"file:{args.uniref_db}?mode=ro", uri=True)
        try:
            rows = con.execute(
                f"SELECT sequence FROM {args.train_table} LIMIT {int(n)}").fetchall()
        finally:
            con.close()
        return [r[0] for r in rows]
    if args.training_data == "fasta":
        out = []
        for s in iter_fasta(args.fasta, args.min_length, n):
            out.append(s)
            if len(out) >= n:
                break
        return out
    return load_human_sequences(args)[:n]


# ============================================================================= #
#  STEP 3 -- RESIDUE-BASED RoPE
#  ESM2's rotary embedding ignores position_ids and rotates by arange(seq_len).
#  We monkeypatch it to consume per-sample residue positions stashed in a
#  thread-local holder (set by the model's forward). With the holder empty it
#  delegates to the original implementation -> exact AA / sequential behaviour.
# ============================================================================= #
_ROPE_STATE = threading.local()


def patch_esm_rotary():
    import torch
    from transformers.models.esm import modeling_esm as esm
    if getattr(esm.EsmRotaryEmbedding, "_residue_patched", False):
        return
    orig_forward = esm.EsmRotaryEmbedding.forward

    def patched_forward(self, q, k):
        positions = getattr(_ROPE_STATE, "positions", None)
        # delegate to stock arange RoPE unless valid positions for THIS tensor are
        # set (shape guard stops a stale holder leaking into a different forward,
        # e.g. the original ESM2 in the sanity check).
        if (positions is None or positions.shape[0] != k.shape[0]
                or positions.shape[1] != k.shape[-2]):
            return orig_forward(self, q, k)
        inv_freq = self.inv_freq.to(k.device)
        t = positions.to(k.device).type_as(inv_freq)          # (batch, seq)
        freqs = torch.einsum("bs,d->bsd", t, inv_freq)         # (batch, seq, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                # (batch, seq, dim)
        cos, sin = emb.cos()[:, None, :, :], emb.sin()[:, None, :, :]
        return (esm.apply_rotary_pos_emb(q, cos, sin),
                esm.apply_rotary_pos_emb(k, cos, sin))

    esm.EsmRotaryEmbedding.forward = patched_forward
    esm.EsmRotaryEmbedding._residue_patched = True
    logger.info("Patched EsmRotaryEmbedding for residue-based position IDs.")


def build_residue_len_table(tokenizer):
    """res_len[token_id] = #residues the token spans. pad=0 (doesn't advance);
    cls/eos/other specials=1 (so AA collapses to sequential IDs); normal tokens
    = len(token_string)."""
    import torch
    vocab = len(tokenizer)
    res_len = torch.ones(vocab, dtype=torch.long)
    id2tok = {i: t for t, i in tokenizer.get_vocab().items()}
    specials = set(tokenizer.all_special_tokens)
    for i in range(vocab):
        tok = id2tok.get(i, "")
        if i == tokenizer.pad_token_id:
            res_len[i] = 0
        elif tok in specials:
            res_len[i] = 1
        else:
            res_len[i] = max(1, len(tok))
    return res_len


def make_vocab_expansion_model_class():
    """Subclass that derives residue position IDs from input_ids (reconstructing
    the pre-masking ids from labels so masked multi-residue spans keep their
    length) and hands them to the patched rotary via the thread-local holder."""
    import torch
    from transformers import EsmForMaskedLM

    class EsmForVocabExpansion(EsmForMaskedLM):
        def set_residue_positions(self, res_len, enabled):
            self.register_buffer("_res_len", res_len, persistent=False)
            self._use_residue_pos = bool(enabled)

        def forward(self, input_ids=None, labels=None, **kw):
            if getattr(self, "_use_residue_pos", False) and input_ids is not None:
                ids = input_ids
                if labels is not None:                 # undo MLM corruption for pos
                    ids = torch.where(labels >= 0, labels, input_ids)
                rl = self._res_len.to(ids.device)[ids]
                # exclusive prefix sum -> residue offset of each token
                _ROPE_STATE.positions = torch.cumsum(rl, dim=1) - rl
            # NB: not cleared afterwards on purpose -> survives gradient-
            # checkpointing recomputation; every forward overwrites it.
            return super().forward(input_ids=input_ids, labels=labels, **kw)

    return EsmForVocabExpansion


# ============================================================================= #
#  STEP 1 -- EMBEDDING INITIALISATION
# ============================================================================= #
def init_embeddings(model, tokenizer, ref_tokenizer, genealogy, args):
    """Mean-pool + norm-shell projection; PUMA children blend with their parent.
    Returns stats dict."""
    import torch

    old_emb = model.get_input_embeddings().weight.data.clone()  # (old_vocab, d)
    ref_vocab = ref_tokenizer.get_vocab()                       # str -> old id
    d = old_emb.shape[1]

    aa_ids = [ref_vocab[c] for c in ESM2_AA_LETTERS if c in ref_vocab]
    mean_aa_norm = old_emb[aa_ids].norm(dim=1).mean().item()
    aa_vec = {c: old_emb[ref_vocab[c]] for c in ESM2_AA_LETTERS if c in ref_vocab}
    glob_mean, glob_std = old_emb.mean().item(), old_emb.std().item()
    g = torch.Generator().manual_seed(args.seed)

    model.resize_token_embeddings(len(tokenizer))
    emb = model.get_input_embeddings().weight.data
    id2tok = {i: t for t, i in tokenizer.get_vocab().items()}
    gamma = args.gamma
    stats = {"copied": 0, "flat": 0, "child": 0, "fallback": 0}

    def shell(v):
        n = v.norm()
        return v / n * mean_aa_norm if n > 0 else v

    def pooled(tok_str):
        vs = [aa_vec[c] for c in tok_str if c in aa_vec]
        return torch.stack(vs).mean(0) if vs else None

    # pass 1: specials/singles copied; flat multi-residue (incl PUMA parents) pooled
    child_ids = []
    for i in range(len(tokenizer)):
        tok = id2tok[i]
        if tok in ref_vocab:                       # special or single residue
            emb[i] = old_emb[ref_vocab[tok]]
            stats["copied"] += 1
            continue
        if genealogy.get(tok, {}).get("is_child"):  # handled in pass 2
            child_ids.append(i)
            continue
        ep = pooled(tok)
        if ep is None:
            emb[i] = torch.empty(d).normal_(glob_mean, glob_std, generator=g)
            stats["fallback"] += 1
        else:
            emb[i] = shell(ep)
            stats["flat"] += 1

    # pass 2: PUMA children -- blend raw pool with parent, THEN project
    tok2id = tokenizer.get_vocab()
    for i in child_ids:
        tok = id2tok[i]
        parent = genealogy[tok]["parent"]
        ep_child = pooled(tok)
        pid = tok2id.get(parent)
        if pid is None or ep_child is None:        # parent absent -> treat as flat
            emb[i] = shell(ep_child) if ep_child is not None else \
                torch.empty(d).normal_(glob_mean, glob_std, generator=g)
            stats["flat" if ep_child is not None else "fallback"] += 1
            continue
        e_parent = emb[pid]                        # already projected (pass 1)
        emb[i] = shell(gamma * ep_child + (1.0 - gamma) * e_parent)
        stats["child"] += 1

    # aligned (child, parent) id lists for the Stage-1 family regulariser
    reg_child, reg_parent = [], []
    for i in range(len(tokenizer)):
        tok = id2tok[i]
        meta = genealogy.get(tok)
        if meta and meta.get("is_child"):
            pid = tok2id.get(meta["parent"])
            if pid is not None:
                reg_child.append(i)
                reg_parent.append(pid)

    logger.info("Embedding init: copied=%d flat=%d child(blend)=%d fallback=%d | "
                "mean_AA_norm=%.3f gamma=%.2f", stats["copied"], stats["flat"],
                stats["child"], stats["fallback"], mean_aa_norm, gamma)
    return mean_aa_norm, reg_child, reg_parent


# ============================================================================= #
#  STEP 2 -- LM HEAD: tie to embedding + smoothed log-frequency bias
# ============================================================================= #
def count_tokens(args, tokenizer):
    """Tally token counts over a bounded sample of the training corpus."""
    counts = np.zeros(len(tokenizer), dtype=np.int64)
    seqs = sample_sequences(args, args.bias_count_samples)
    logger.info("Counting tokens over %d sampled sequences for the bias prior ...",
                len(seqs))
    B = 1000
    for i in range(0, len(seqs), B):
        enc = tokenizer(seqs[i:i + B], add_special_tokens=False)["input_ids"]
        for ids in enc:
            if ids:
                np.add.at(counts, ids, 1)
    return counts


def init_lm_head_bias(model, tokenizer, genealogy, counts, args):
    """bias = log((c̃ + ε)/(total + ε·V)); PUMA children use a family-shrunk c̃."""
    import torch
    V = len(tokenizer)
    eps, delta = args.bias_epsilon, args.delta
    total = int(counts.sum())
    tok2id = tokenizer.get_vocab()
    id2tok = {i: t for t, i in tok2id.items()}

    # family aggregates (parent + its children) for child shrinkage
    families = {}                                  # parent_str -> [member ids]
    for tok, meta in genealogy.items():
        if meta.get("is_child"):
            families.setdefault(meta["parent"], []).append(tok2id.get(tok))
    fam_count, fam_size = {}, {}
    for parent, child_ids in families.items():
        pid = tok2id.get(parent)
        members = [pid] + child_ids if pid is not None else list(child_ids)
        members = [m for m in members if m is not None]
        fam_count[parent] = int(sum(int(counts[m]) for m in members))
        fam_size[parent] = max(1, len(members))

    c_tilde = counts.astype(np.float64).copy()
    n_smoothed = 0
    for i in range(V):
        meta = genealogy.get(id2tok[i])
        if meta and meta.get("is_child"):
            parent = meta["parent"]
            if parent in fam_count:
                c_tilde[i] += delta * (fam_count[parent] / fam_size[parent])
                n_smoothed += 1
    bias = np.log((c_tilde + eps) / (total + eps * V))

    lm_bias = getattr(model.lm_head, "bias", None)
    if lm_bias is not None:
        with torch.no_grad():
            model.lm_head.bias = torch.nn.Parameter(
                torch.tensor(bias, dtype=lm_bias.dtype, device=lm_bias.device))
    model.tie_weights()                            # keep decoder == embedding^T
    logger.info("LM-head bias: log-frequency prior over %d tokens (total=%d, "
                "ε=%.3g); family-smoothed %d PUMA children (δ=%.3g).",
                V, total, eps, n_smoothed, delta)


# ============================================================================= #
#  STEP 4 -- LoRA PRESETS
# ============================================================================= #
_LORA_SUFFIX = {
    "q": "attention.self.query", "k": "attention.self.key",
    "v": "attention.self.value", "o": "attention.output.dense",
    "up": "intermediate.dense", "down": "output.dense"}


def lora_plan(preset, L):
    """List of (layer_idx, matrix_key, rank)."""
    plan = []
    if preset == "uniform_r16":
        for i in range(L):
            for m in ("q", "k", "v", "o", "up", "down"):
                plan.append((i, m, 16))
    elif preset == "attn_only_r16":
        for i in range(L):
            for m in ("q", "k"):
                plan.append((i, m, 16))
    elif preset == "layerwise":
        e, mid = L // 3, 2 * L // 3
        for i in range(L):
            if i < e:
                spec = [("q", 16), ("k", 16), ("v", 16), ("o", 16), ("up", 16)]
            elif i < mid:
                spec = [("q", 8), ("k", 8), ("v", 8), ("o", 8)]
            else:
                spec = [("q", 8), ("k", 8)]
            for m, r in spec:
                plan.append((i, m, r))
    else:
        raise ValueError(f"Unknown lora_preset {preset!r}")
    return plan


def build_lora_config(preset, L, dropout):
    from peft import LoraConfig
    plan = lora_plan(preset, L)
    targets, rank_pattern, alpha_pattern = [], {}, {}
    for (i, m, r) in plan:
        full = f"esm.encoder.layer.{i}.{_LORA_SUFFIX[m]}"  # exact module name
        key = f"layer.{i}.{_LORA_SUFFIX[m]}"               # suffix for patterns
        targets.append(full)
        rank_pattern[key] = r
        alpha_pattern[key] = 2 * r
    n_params = sum(r for _, _, r in plan)
    logger.info("LoRA preset '%s': %d adapted matrices across %d layers.",
                preset, len(plan), L)
    return LoraConfig(r=16, lora_alpha=32, lora_dropout=dropout, bias="none",
                      target_modules=sorted(set(targets)),
                      rank_pattern=rank_pattern, alpha_pattern=alpha_pattern,
                      task_type=None)


# ============================================================================= #
#  MODEL ASSEMBLY
# ============================================================================= #
def build_model(args, tokenizer, ref_tokenizer, genealogy, counts):
    import torch
    name = args.pretrained_name or ESM2_HF_NAMES.get(args.model_size)
    if not name:
        raise ValueError("Provide --model_size or --pretrained_name.")

    if args.rope_mode == "residue" and args.tokenizer_type.lower() != "aa":
        patch_esm_rotary()

    EsmVE = make_vocab_expansion_model_class()
    logger.info("Loading pretrained ESM2 weights from %s", name)
    try:
        model = EsmVE.from_pretrained(name, attn_implementation=args.attn_implementation)
        logger.info("Attention implementation: %s", args.attn_implementation)
    except (ValueError, ImportError, RuntimeError) as e:
        logger.warning("attn=%s unavailable (%s); using eager.", args.attn_implementation, e)
        model = EsmVE.from_pretrained(name, attn_implementation="eager")

    # Step 1 + 2 (analytical init on the un-adapted backbone)
    mean_aa_norm, reg_child, reg_parent = init_embeddings(
        model, tokenizer, ref_tokenizer, genealogy, args)
    init_lm_head_bias(model, tokenizer, genealogy, counts, args)

    model.config.vocab_size = len(tokenizer)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.mask_token_id = tokenizer.mask_token_id
    model.config.bos_token_id = tokenizer.cls_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    # Step 3 wiring (AA collapses to sequential, so we don't patch/enable it)
    res_len = build_residue_len_table(tokenizer)
    use_res = args.rope_mode == "residue" and args.tokenizer_type.lower() != "aa"
    model.set_residue_positions(res_len, use_res)
    logger.info("RoPE mode: %s position IDs", "residue-based" if use_res else "sequential")

    # Step 4: freeze body, inject LoRA
    from peft import get_peft_model
    cfg = build_lora_config(args.lora_preset, model.config.num_hidden_layers, args.lora_dropout)
    model = get_peft_model(model, cfg)
    n_lora = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    n_all = sum(p.numel() for p in model.parameters())
    logger.info("LoRA params: %.2fM (%.1f%% of %.1fM total)",
                n_lora / 1e6, 100 * n_lora / n_all, n_all / 1e6)

    reg_child_t = torch.tensor(reg_child, dtype=torch.long) if reg_child else None
    reg_parent_t = torch.tensor(reg_parent, dtype=torch.long) if reg_parent else None
    return model, reg_child_t, reg_parent_t


# ============================================================================= #
#  STEP 5 -- STAGED TRAINING
# ============================================================================= #
def classify_param(name):
    if "lora_" in name:
        return "lora"
    if "word_embeddings" in name:
        return "emb"
    if name.endswith("lm_head.bias"):
        return "head_bias"
    if "lm_head.dense" in name:
        return "head_inner"
    if "LayerNorm" in name or "layer_norm" in name:
        return "ln"
    if "lm_head.decoder" in name:
        return "emb"                                # tied to word_embeddings
    return "frozen"


# group -> LR for each stage (built from args in main)
def stage_param_groups(model, lr_map):
    """Set requires_grad and return optimizer param groups for the active groups."""
    buckets = {g: [] for g in lr_map}
    for name, p in model.named_parameters():
        grp = classify_param(name)
        if grp in lr_map:
            p.requires_grad = True
            buckets[grp].append(p)
        else:
            p.requires_grad = False
    groups = [{"params": ps, "lr": lr_map[g]} for g, ps in buckets.items() if ps]
    n = sum(p.numel() for ps in buckets.values() for p in ps)
    return groups, n


def make_trainer_class():
    from transformers import Trainer

    class _Trainer(Trainer):
        puma_lambda = 0.0
        reg_child = None
        reg_parent = None

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            outputs = model(**inputs)
            loss = outputs.loss
            if self.puma_lambda > 0 and self.reg_child is not None:
                import torch
                emb = model.get_input_embeddings().weight
                ec = emb[self.reg_child.to(emb.device)]
                ep = emb[self.reg_parent.to(emb.device)]
                reg = (1.0 - torch.nn.functional.cosine_similarity(ec, ep, dim=-1)).mean()
                loss = loss + self.puma_lambda * reg
            return (loss, outputs) if return_outputs else loss

    return _Trainer


def make_resume_step_override_callback():
    """After resuming, TrainerState.load_from_json restores the OLD logging/eval/
    save steps baked into the checkpoint; re-assert the current args."""
    from transformers import TrainerCallback

    class ResumeStepOverride(TrainerCallback):
        def on_train_begin(self, args, state, control, **kw):
            state.logging_steps = args.logging_steps
            state.eval_steps = args.eval_steps
            state.save_steps = args.save_steps
            return control
    return ResumeStepOverride()


def make_full_state_save_callback():
    """PEFT checkpoints save only the adapter, but we also train the embeddings /
    LM-head bias (base-model params). Dump the FULL state_dict into each
    checkpoint so a resumed stage restores those too (pruned with the checkpoint
    by save_total_limit)."""
    from transformers import TrainerCallback

    class FullStateSave(TrainerCallback):
        def on_save(self, args, state, control, model=None, **kw):
            if not state.is_world_process_zero or model is None:
                return
            import torch
            ckpt = os.path.join(args.output_dir, f"checkpoint-{int(state.global_step)}")
            if os.path.isdir(ckpt):
                torch.save(model.state_dict(), os.path.join(ckpt, "full_state.pt"))
    return FullStateSave()


def make_wandb_stage_callback(step_offset, stage_idx):
    """Log every stage into the SAME (manually-managed) wandb run on a continuous
    x-axis: global x = step_offset + per-stage global_step. Trainer's own wandb
    integration is disabled (report_to='none') so there is no double logging."""
    from transformers import TrainerCallback

    class WandbStage(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not state.is_world_process_zero or not logs:
                return
            try:
                import wandb
                if wandb.run is None:
                    return
                data = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                data["stage"] = stage_idx
                wandb.log(data, step=step_offset + int(state.global_step))
            except Exception:  # noqa: BLE001
                pass
    return WandbStage()


# ============================================================================= #
#  STAGE-AWARE RESUME  (checkpoint + progress bookkeeping)
# ============================================================================= #
def stage_dir(output_dir, k):
    return os.path.join(output_dir, f"stage{k}")


def find_latest_checkpoint(d):
    cks = glob.glob(os.path.join(d, "checkpoint-*"))
    cks = [c for c in cks if os.path.isdir(c) and c.split("-")[-1].isdigit()]
    return max(cks, key=lambda c: int(c.split("-")[-1])) if cks else None


def load_progress(output_dir):
    p = os.path.join(output_dir, "progress.json")
    if os.path.isfile(p):
        try:
            return set(json.load(open(p)).get("completed", []))
        except Exception:  # noqa: BLE001
            return set()
    return set()


def save_progress(output_dir, completed):
    json.dump({"completed": sorted(completed)},
              open(os.path.join(output_dir, "progress.json"), "w"))


def resolve_precision(choice):
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


def run_stage(args, name, stage_idx, model, tokenizer, train_ds, eval_ds, collator, *,
              lr_map, max_steps, output_dir, puma_lambda, reg_child, reg_parent,
              step_offset, resume_from_checkpoint, wandb_active):
    import torch
    from transformers import TrainingArguments, get_scheduler, Trainer

    groups, n_trainable = stage_param_groups(model, lr_map)
    logger.info("=== %s | trainable=%.2fM | groups=%s | max_steps=%d%s ===",
                name, n_trainable / 1e6,
                {g: f"{lr:.0e}" for g, lr in lr_map.items()}, max_steps,
                " | RESUMING" if resume_from_checkpoint else "")

    fused = args.optim == "adamw_torch_fused" and torch.cuda.is_available()
    try:
        optimizer = torch.optim.AdamW(groups, betas=(args.adam_beta1, args.adam_beta2),
                                      eps=args.adam_epsilon,
                                      weight_decay=args.weight_decay, fused=fused)
    except (TypeError, RuntimeError):                 # older torch: no fused kwarg
        optimizer = torch.optim.AdamW(groups, betas=(args.adam_beta1, args.adam_beta2),
                                      eps=args.adam_epsilon, weight_decay=args.weight_decay)
    warmup = max(1, int(args.warmup_ratio * max_steps))
    scheduler = get_scheduler(args.lr_scheduler_type, optimizer,
                              num_warmup_steps=warmup, num_training_steps=max_steps)

    fp16, bf16 = resolve_precision(args.precision)
    do_eval = eval_ds is not None
    ta = TrainingArguments(
        output_dir=output_dir, max_steps=max_steps, seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_grad_norm=args.max_grad_norm, fp16=fp16, bf16=bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        group_by_length=args.group_by_length, torch_compile=args.torch_compile,
        dataloader_num_workers=args.dataloader_num_workers,
        do_eval=do_eval, eval_strategy="steps" if do_eval else "no",
        eval_steps=args.eval_steps, logging_steps=args.logging_steps,
        save_strategy="steps", save_steps=args.save_steps,
        save_total_limit=args.save_total_limit, report_to=["none"],
        ddp_find_unused_parameters=False, remove_unused_columns=False)

    callbacks = [make_resume_step_override_callback(), make_full_state_save_callback()]
    if wandb_active:
        callbacks.append(make_wandb_stage_callback(step_offset, stage_idx))

    TrainerCls = make_trainer_class()
    kwargs = dict(model=model, args=ta, train_dataset=train_ds, eval_dataset=eval_ds,
                  data_collator=collator, optimizers=(optimizer, scheduler),
                  callbacks=callbacks)
    if "processing_class" in inspect.signature(Trainer.__init__).parameters:
        kwargs["processing_class"] = tokenizer
    else:
        kwargs["tokenizer"] = tokenizer
    trainer = TrainerCls(**kwargs)
    trainer.puma_lambda = puma_lambda
    trainer.reg_child, trainer.reg_parent = reg_child, reg_parent
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    return trainer


def init_wandb_run(args, slug):
    """Open ONE wandb run for the whole 3-stage schedule. The run id is taken
    from --wandb_run_id, else a saved id in output_dir (so a plain re-run with
    --resume_from_checkpoint reattaches automatically), else freshly generated
    and persisted. Returns the run id (or None if wandb is off / not rank 0)."""
    if not args.wandb_project or int(os.environ.get("RANK", "0")) != 0:
        return None
    import wandb
    id_file = os.path.join(args.output_dir, "wandb_run_id.txt")
    run_id = args.wandb_run_id
    if not run_id and os.path.isfile(id_file):
        run_id = open(id_file).read().strip()
    if not run_id:
        run_id = wandb.util.generate_id()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity,
               name=args.wandb_run_name or slug, id=run_id,
               group=args.wandb_group or slug, resume=args.wandb_resume,
               mode=args.wandb_mode, config=vars(args))
    os.makedirs(args.output_dir, exist_ok=True)
    with open(id_file, "w") as fh:
        fh.write(run_id)
    logger.info("wandb run id=%s (resume=%s) -> all 3 stages log here.",
                run_id, args.wandb_resume)
    return run_id


def finish_wandb():
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish()
    except Exception:  # noqa: BLE001
        pass


# ============================================================================= #
#  POST-TRAINING SANITY CHECK  (before vs after, in memory)
# ============================================================================= #
DEFAULT_PROBES = [
    "MLKAAAKRPELSGKNTISNNSDMAEVKSMFREVLPKQGPLFVEDIMTMVLCKPKLLPLKSLTLEKLEKMHQAAQNTIRQQEMAEKDQRQITH",
    "MDTAYPREDTRAPTPSKAGAHTALTLGAPHPPPRDHLIWSVFSTLYLNLCCLGFLALAYSIKARDQKVVGDLEAARRFGSKAKCYNILAAMWTLVPPLLLLGLVVTGALHLARLAKDSAAFFSTKFDDADYD"]


def _pll_per_residue(model, tok, seqs, max_length, device):
    import torch
    special = set(tok.all_special_ids)
    tot_nll, tot_res = 0.0, 0
    model.eval()
    for seq in seqs:
        ids = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
        pos = [i for i, t in enumerate(ids.tolist()) if t not in special]
        for p in pos:
            true = ids[p].item()
            masked = ids.clone(); masked[p] = tok.mask_token_id
            with torch.no_grad():
                logits = model(input_ids=masked.unsqueeze(0).to(device)).logits[0, p]
            tot_nll += -logits.float().log_softmax(-1)[true].item()
            tot_res += len(tok.convert_ids_to_tokens([true])[0])
    return tot_nll / max(1, tot_res)


def _masked_topk(model, tok, seq, topk, max_length, device):
    import torch
    ids = tok(seq, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"][0]
    special = set(tok.all_special_ids)
    pos = [i for i, t in enumerate(ids.tolist()) if t not in special]
    if not pos:
        return None, []
    p = pos[len(pos) // 2]
    true_tok = tok.convert_ids_to_tokens([ids[p].item()])[0]
    masked = ids.clone(); masked[p] = tok.mask_token_id
    with torch.no_grad():
        logits = model(input_ids=masked.unsqueeze(0).to(device)).logits[0, p]
    top = torch.topk(logits.float().softmax(-1), k=min(topk, logits.numel()))
    pairs = list(zip(tok.convert_ids_to_tokens(top.indices.tolist()), top.values.tolist()))
    return true_tok, pairs


def sanity_check(trained_model, trained_tok, orig_name, probes, topk, max_length, offline):
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    _ROPE_STATE.positions = None        # don't let a stale holder hit the baseline
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("\n" + "=" * 78)
    print("POST-ADAPTATION SANITY CHECK  (before = original ESM2 vs after = adapted)")
    print("=" * 78)
    orig_tok = AutoTokenizer.from_pretrained(orig_name)
    orig_model = AutoModelForMaskedLM.from_pretrained(orig_name).to(device).eval()
    trained_model = trained_model.to(device).eval()

    for label, (m, t) in [("BEFORE", (orig_model, orig_tok)),
                          ("AFTER ", (trained_model, trained_tok))]:
        ppr = _pll_per_residue(m, t, probes, max_length, device)
        print(f"\n[{label}] vocab={len(t):>6d}  pseudo-NLL/residue={ppr:.4f}  (lower=better)")

    for i, seq in enumerate(probes):
        print(f"\n--- probe {i + 1}: {seq[:48]}{'...' if len(seq) > 48 else ''}")
        for label, (m, t) in [("BEFORE", (orig_model, orig_tok)),
                              ("AFTER ", (trained_model, trained_tok))]:
            true_tok, pairs = _masked_topk(m, t, seq, topk, max_length, device)
            if true_tok is None:
                continue
            preds = ", ".join(f"{tk}:{p:.2f}" for tk, p in pairs[:topk])
            print(f"  [{label}] masked true='{true_tok}' in-top{topk}="
                  f"{any(tk == true_tok for tk, _ in pairs)} | top-{topk}: {preds}")
    print("\n" + "=" * 78 + "\n")


# ============================================================================= #
#  ARGUMENTS
# ============================================================================= #
def parse_args():
    p = argparse.ArgumentParser(
        description="Vocab-expansion adaptation of ESM-2 (analytic init + LoRA + staged).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # data
    p.add_argument("--training_data", default="all", choices=["human", "all", "fasta"])
    p.add_argument("--fasta", default=None)
    p.add_argument("--db_file", default=DEFAULT_DB_FILE)
    p.add_argument("--db_subset", default="uniref50", choices=["uniref50", "uniref90"])
    p.add_argument("--db_query", default=None)
    p.add_argument("--db_max_length", type=int, default=1000)
    p.add_argument("--uniref_db", default=DEFAULT_UNIREF_DB)
    p.add_argument("--train_table", default="plm_train")
    p.add_argument("--val_table", default="plm_validation")
    p.add_argument("--min_length", type=int, default=1)
    p.add_argument("--max_length", type=int, default=1024,
                   help="Random-crop window in CORE tokens (BOS/EOS added on top).")
    p.add_argument("--val_split", type=float, default=0.01)
    p.add_argument("--max_samples", type=int, default=None)

    # tokenizer
    p.add_argument("--tokenizer_type", required=True, choices=["puma", "bpe", "aa"])
    p.add_argument("--tokenizer_file", default=None)
    p.add_argument("--tokenizer_dir", default=DEFAULT_TOKENIZER_DIR)
    p.add_argument("--tok_dataset", default="uniref50", choices=["uniref50", "uniref90"])
    p.add_argument("--subs_matrix", default="blosum62",
                   choices=["blosum45", "blosum62", "pam70", "pam250"])
    p.add_argument("--mutation_cutoff", default="0.7")
    p.add_argument("--min_mutation_freq", default="0.05")
    p.add_argument("--min_mutation_len", default="3")
    p.add_argument("--max_mutation_len", default="12")
    p.add_argument("--vocab_size", default="1600")

    # model / backbone
    p.add_argument("--model_size", default="35M", choices=list(ESM2_HF_NAMES) + [""])
    p.add_argument("--pretrained_name", default=None)
    p.add_argument("--attn_implementation", default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--offline", action="store_true")

    # Step 1/2/3 hyperparameters
    p.add_argument("--gamma", type=float, default=0.6,
                   help="PUMA child blend weight on the pooled (vs parent) vector.")
    p.add_argument("--delta", type=float, default=0.1,
                   help="PUMA family-shrinkage strength for the child bias prior.")
    p.add_argument("--bias_epsilon", type=float, default=1.0,
                   help="Laplace add-ε for the log-frequency bias.")
    p.add_argument("--bias_count_samples", type=int, default=200000,
                   help="#sequences sampled to estimate token frequencies.")
    p.add_argument("--rope_mode", default="residue", choices=["residue", "sequential"],
                   help="residue: residue-based position IDs (Step 3); sequential: "
                        "stock ESM2 token-index RoPE (ablation baseline).")

    # Step 4 LoRA
    p.add_argument("--lora_preset", default="layerwise",
                   choices=["uniform_r16", "layerwise", "attn_only_r16"])
    p.add_argument("--lora_dropout", type=float, default=0.05)

    # Step 6 masking
    p.add_argument("--mlm_probability", type=float, default=0.15,
                   help="Token mask rate; 0.15 keeps residue density k̄-invariant.")

    # Step 5 staged training
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Total optimizer steps across all 3 stages; -1 -> from epochs.")
    p.add_argument("--stage1_frac", type=float, default=0.05)
    p.add_argument("--stage2_frac", type=float, default=0.75)
    p.add_argument("--stage3_frac", type=float, default=0.20)
    p.add_argument("--stage1_emb_lr", type=float, default=5e-4)
    p.add_argument("--stage2_lora_lr", type=float, default=1e-4)
    p.add_argument("--stage2_head_lr", type=float, default=1e-4)
    p.add_argument("--stage2_emb_lr", type=float, default=1e-5)
    p.add_argument("--stage3_lr", type=float, default=3e-5)
    p.add_argument("--stage3_ln_lr", type=float, default=1e-5)
    p.add_argument("--puma_lambda", type=float, default=0.03,
                   help="Stage-1 family-alignment regulariser weight (0 disables).")

    # optimisation / hardware
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--per_device_eval_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta1", type=float, default=0.9)
    p.add_argument("--adam_beta2", type=float, default=0.98)
    p.add_argument("--adam_epsilon", type=float, default=1e-8)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--lr_scheduler_type", default="cosine")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="Optimizer kernel: adamw_torch_fused (CUDA, faster) or "
                        "adamw_torch. The staged trainer builds AdamW directly.")
    p.add_argument("--precision", default="auto", choices=["auto", "bf16", "fp16", "no"])
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--group_by_length", action="store_true", default=True)
    p.add_argument("--no_group_by_length", dest="group_by_length", action="store_false")
    p.add_argument("--torch_compile", action="store_true")
    p.add_argument("--dataloader_num_workers", type=int, default=4)

    # logging / io
    p.add_argument("--output_dir", default=None)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--eval_steps", type=int, default=1000)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--resume_from_checkpoint", default=None,
                   help="Any value enables stage-aware resume: progress.json + the "
                        "latest stage checkpoint decide which stage/step to continue "
                        "from (mid-stage or at a boundary). Re-use the SAME --output_dir.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sanity_check", action="store_true", default=True)
    p.add_argument("--no_sanity_check", dest="sanity_check", action="store_false")
    p.add_argument("--sanity_topk", type=int, default=10)
    p.add_argument("--sanity_sequences", default=None,
                   help="Comma-separated probe sequences (default: two human proteins).")

    # wandb
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_group", default=None, help="Group runs (e.g. by model size).")
    p.add_argument("--wandb_run_id", default=None,
                   help="Reattach to THIS wandb run id (all 3 stages log into it). "
                        "Auto-saved to output_dir/wandb_run_id.txt for resume.")
    p.add_argument("--wandb_resume", default="allow",
                   choices=["allow", "must", "never", "auto"])
    p.add_argument("--wandb_mode", default="online",
                   choices=["online", "offline", "disabled"])
    return p.parse_args()


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

    if args.training_data == "fasta" and not args.fasta:
        raise ValueError("--training_data fasta requires --fasta PATH")
    # dynamic random cropping makes precomputed lengths meaningless -> no grouping.
    if args.group_by_length:
        logger.info("Disabling group_by_length (incompatible with dynamic cropping).")
        args.group_by_length = False

    slug = build_run_slug(args)
    if not args.output_dir:
        args.output_dir = slug
        logger.info("output_dir -> %s", args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)

    want_resume = bool(args.resume_from_checkpoint)

    # --- build pieces ---------------------------------------------------- #
    import torch
    ref_tokenizer = load_reference_tokenizer(args.offline)
    tokenizer = build_tokenizer(args)
    genealogy = load_genealogy(args) if args.tokenizer_type.lower() == "puma" else {}
    train_ds, eval_ds = get_dynamic_datasets(args, tokenizer)
    counts = count_tokens(args, tokenizer)
    model, reg_child, reg_parent = build_model(
        args, tokenizer, ref_tokenizer, genealogy, counts)
    if args.gradient_checkpointing:
        model.config.use_cache = False

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability,
        pad_to_multiple_of=8)

    # --- total step budget & per-stage allocation ------------------------ #
    if args.max_steps and args.max_steps > 0:
        total = args.max_steps
    else:
        world = int(os.environ.get("WORLD_SIZE", "1") or "1")
        eff = max(1, args.per_device_train_batch_size *
                  args.gradient_accumulation_steps * world)
        total = max(3, int(math.ceil(len(train_ds) / eff) * args.num_train_epochs))
    steps = [max(1, int(args.stage1_frac * total)), max(1, int(args.stage2_frac * total))]
    steps.append(max(1, total - steps[0] - steps[1]))
    offsets = [0, steps[0], steps[0] + steps[1]]      # continuous wandb x-axis
    logger.info("Step budget: total=%d -> stage1=%d stage2=%d stage3=%d",
                total, *steps)

    stage_specs = [
        ("STAGE 1/3 (emb warm-up)", {"emb": args.stage1_emb_lr}, args.puma_lambda),
        ("STAGE 2/3 (LoRA+head+emb)",
         {"lora": args.stage2_lora_lr, "head_bias": args.stage2_head_lr,
          "emb": args.stage2_emb_lr}, 0.0),
        ("STAGE 3/3 (+ LayerNorms)",
         {"lora": args.stage3_lr, "head_bias": args.stage3_lr,
          "head_inner": args.stage3_lr, "emb": args.stage3_lr,
          "ln": args.stage3_ln_lr}, 0.0),
    ]

    # --- stage-aware resume orchestration -------------------------------- #
    completed = load_progress(args.output_dir) if want_resume else set()
    start_stage, resume_ckpt, preload = 1, None, None
    if want_resume:
        incomplete = [k for k in (1, 2, 3) if k not in completed]
        start_stage = incomplete[0] if incomplete else 4
        if start_stage <= 3:
            resume_ckpt = find_latest_checkpoint(stage_dir(args.output_dir, start_stage))
            if resume_ckpt:                            # interrupted mid-stage
                preload = os.path.join(resume_ckpt, "full_state.pt")
            elif start_stage > 1:                      # clean boundary -> prior weights
                preload = os.path.join(stage_dir(args.output_dir, start_stage - 1),
                                       "final_state.pt")
        else:                                          # all done -> load final weights
            preload = os.path.join(stage_dir(args.output_dir, 3), "final_state.pt")
        logger.info("Resume: completed=%s -> start_stage=%s ckpt=%s",
                    sorted(completed), start_stage, resume_ckpt)
        if preload and os.path.isfile(preload):
            logger.info("Preloading weights from %s", preload)
            model.load_state_dict(torch.load(preload, map_location="cpu"), strict=False)
        elif preload:
            logger.warning("Expected preload weights missing (%s); starting that "
                           "stage from analytic init.", preload)

    # --- single continuous wandb run for all 3 stages -------------------- #
    init_wandb_run(args, slug)
    wandb_active = bool(args.wandb_project) and int(os.environ.get("RANK", "0")) == 0

    for k in range(start_stage, 4):
        name, lr_map, lam = stage_specs[k - 1]
        run_stage(args, name, k, model, tokenizer, train_ds, eval_ds, collator,
                  lr_map=lr_map, max_steps=steps[k - 1],
                  output_dir=stage_dir(args.output_dir, k),
                  puma_lambda=lam, reg_child=reg_child, reg_parent=reg_parent,
                  step_offset=offsets[k - 1],
                  resume_from_checkpoint=(resume_ckpt if k == start_stage else None),
                  wandb_active=wandb_active)
        # mark stage complete + snapshot full weights for the next stage / restarts
        torch.save(model.state_dict(),
                   os.path.join(stage_dir(args.output_dir, k), "final_state.pt"))
        completed.add(k)
        save_progress(args.output_dir, completed)
    finish_wandb()

    # --- save: LoRA adapter + merged full model + tokenizer -------------- #
    adapter_dir = os.path.join(args.output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    logger.info("Saved LoRA adapter to %s", adapter_dir)

    merged = model.merge_and_unload()
    final_dir = os.path.join(args.output_dir, "final")
    merged.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Saved merged model + tokenizer to %s", final_dir)

    if args.sanity_check:
        probes = ([s.strip() for s in args.sanity_sequences.split(",") if s.strip()]
                  if args.sanity_sequences else DEFAULT_PROBES)
        orig_name = args.pretrained_name or ESM2_HF_NAMES.get(args.model_size)
        try:
            sanity_check(merged, tokenizer, orig_name, probes, args.sanity_topk,
                         args.max_length, args.offline)
        except Exception as e:  # noqa: BLE001
            logger.warning("Sanity check failed (training still succeeded): %s", e)


if __name__ == "__main__":
    main()
