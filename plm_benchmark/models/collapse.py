"""PUMA parent-collapse tokenization.

A parent-collapsed model uses a *reduced* vocab (parents + singletons + specials),
but segmentation is done with the *full* PUMA vocab and then remapped child->parent
(see plm_train.build_puma_collapse / _CropDatasetBase.encode). ``CollapseTokenizer``
reproduces that "tokenize-with-full -> collapse -> reduced ids" path and implements only
the slice of the tokenizer API the collators use, so it is a drop-in for them.

Two sources:
* pretrained ``_PC`` checkpoint -> ``from_checkpoint`` (loads full_tokenizer/ + collapse.npy)
* scratch baseline over a PUMA tokenizer -> ``from_puma`` (builds the collapse in-memory
  from the tokenizer JSON + its family/genealogy JSON).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

COLLAPSE_FILE = "collapse.npy"
FULL_TOKENIZER_DIR = "full_tokenizer"


def is_collapse_checkpoint(checkpoint: str) -> bool:
    """True iff the checkpoint dir carries the parent-collapse artifacts."""
    if not checkpoint:
        return False
    p = Path(checkpoint)
    return (p / COLLAPSE_FILE).is_file() and (p / FULL_TOKENIZER_DIR).is_dir()


def load_genealogy(tokenizer_json_path: str) -> dict[str, str]:
    """Load {child_token: parent_token} from the PUMA family JSON.

    The family JSON is the non-'hf_' sibling of the tokenizer file (same dir),
    mirroring plm_train.load_genealogy. Raises if it is missing.
    """
    p = Path(tokenizer_json_path)
    sibling = p.name[3:] if p.name.startswith("hf_") else p.name
    attr = p.parent / sibling
    if not attr.is_file():
        raise FileNotFoundError(
            f"PUMA family JSON not found: {attr} (expected the non-'hf_' sibling of "
            f"{p.name}). --parent-collapse requires it."
        )
    with attr.open() as fh:
        raw = json.load(fh)
    return {t: m["parent"] for t, m in raw.items() if isinstance(m, dict) and "parent" in m}


def build_puma_collapse(full_tok, genealogy: dict[str, str]):
    """Fold PUMA children onto parents. Mirrors plm_train.build_puma_collapse.

    Returns (collapse, reduced_size, cls_id, eos_id, pad_id) where collapse is an
    int64 array full_id -> reduced_id and the special ids are in the reduced space.
    """
    full_vocab = full_tok.get_vocab()  # {token_str: id}
    id2str = {i: s for s, i in full_vocab.items()}

    reduced_ids: dict[int, int] = {}  # old_id -> new (reduced) id
    for old_id in sorted(id2str):
        if id2str[old_id] in genealogy:  # child -> dropped from reduced vocab
            continue
        reduced_ids[old_id] = len(reduced_ids)

    unk_new = reduced_ids.get(full_vocab.get(full_tok.unk_token), 0)
    collapse = np.empty(len(full_vocab), dtype=np.int64)
    for old_id, s in id2str.items():
        if s in genealogy:
            pid = full_vocab.get(genealogy[s])
            collapse[old_id] = reduced_ids.get(pid, unk_new)
        else:
            collapse[old_id] = reduced_ids[old_id]

    cls_id = reduced_ids[full_vocab[full_tok.cls_token]]
    eos_id = reduced_ids[full_vocab[full_tok.eos_token]]
    pad_id = reduced_ids[full_vocab[full_tok.pad_token]]
    return collapse, len(reduced_ids), cls_id, eos_id, pad_id


class CollapseTokenizer:
    """Full-vocab segmentation -> child->parent remap -> reduced (model) ids."""

    def __init__(
        self,
        *,
        full_tok,
        collapse,
        cls_token_id,
        eos_token_id,
        pad_token_id,
        reduced_size,
        model_max_length,
        reduced_tok=None,
    ):
        self.full_tok = full_tok
        self.collapse = np.asarray(collapse, dtype=np.int64)
        self.cls_token_id = cls_token_id
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.reduced_size = reduced_size
        self.model_max_length = model_max_length
        self.reduced_tok = reduced_tok  # only for save_pretrained round-trip
        self.model_input_names = ["input_ids", "attention_mask"]

    # -- constructors -------------------------------------------------------- #
    @classmethod
    def from_checkpoint(cls, checkpoint: str) -> "CollapseTokenizer":
        p = Path(checkpoint)
        full_tok = AutoTokenizer.from_pretrained(str(p / FULL_TOKENIZER_DIR))
        reduced_tok = AutoTokenizer.from_pretrained(str(p))  # model id space + specials
        collapse = np.load(p / COLLAPSE_FILE).astype(np.int64)
        n_full = len(full_tok.get_vocab())
        if len(collapse) != n_full:
            raise ValueError(
                f"collapse.npy length {len(collapse)} != full_tokenizer vocab {n_full} in {checkpoint}"
            )
        return cls(
            full_tok=full_tok,
            collapse=collapse,
            cls_token_id=reduced_tok.cls_token_id,
            eos_token_id=reduced_tok.eos_token_id,
            pad_token_id=reduced_tok.pad_token_id,
            reduced_size=len(reduced_tok.get_vocab()),
            model_max_length=reduced_tok.model_max_length,
            reduced_tok=reduced_tok,
        )

    @classmethod
    def from_puma(cls, full_tok, genealogy: dict[str, str], model_max_length: int) -> "CollapseTokenizer":
        collapse, reduced_size, cls_id, eos_id, pad_id = build_puma_collapse(full_tok, genealogy)
        return cls(
            full_tok=full_tok,
            collapse=collapse,
            cls_token_id=cls_id,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            reduced_size=reduced_size,
            model_max_length=model_max_length,
        )

    # -- tokenizer API used by the pipeline ---------------------------------- #
    def __len__(self) -> int:
        return self.reduced_size

    def validate(self, model_vocab_size: int) -> None:
        """Fail loud if any remapped id would be out of the embedding range."""
        mx = int(self.collapse.max()) if len(self.collapse) else -1
        if mx >= model_vocab_size:
            raise ValueError(
                f"collapse maps to id {mx} >= model vocab_size {model_vocab_size}; "
                "reduced tokenizer / model mismatch."
            )

    def _encode_one(self, seq: str, max_length: int, truncation: bool) -> list[int]:
        core = self.full_tok(seq, add_special_tokens=False, truncation=False)["input_ids"]
        core = self.collapse[np.asarray(core, dtype=np.int64)] if core else np.empty(0, np.int64)
        if truncation and max_length:
            core = core[: max(0, max_length - 2)]  # room for cls/eos
        return [self.cls_token_id, *core.tolist(), self.eos_token_id]

    def __call__(
        self,
        sequences,
        *,
        max_length=None,
        padding=True,
        truncation=True,
        return_tensors=None,
        **kwargs,
    ):
        if isinstance(sequences, str):
            sequences = [sequences]
        max_length = max_length or self.model_max_length
        encoded = [self._encode_one(s, max_length, truncation) for s in sequences]

        width = max((len(x) for x in encoded), default=0)
        input_ids, attention_mask = [], []
        for ids in encoded:
            n_pad = width - len(ids)
            input_ids.append(ids + [self.pad_token_id] * n_pad)
            attention_mask.append([1] * len(ids) + [0] * n_pad)

        if return_tensors == "pt":
            import torch

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def save_pretrained(self, save_directory, **kwargs):
        """Persist a valid _PC layout (Trainer calls this when checkpointing)."""
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)
        if self.reduced_tok is not None:
            self.reduced_tok.save_pretrained(str(dest))
        self.full_tok.save_pretrained(str(dest / FULL_TOKENIZER_DIR))
        np.save(dest / COLLAPSE_FILE, self.collapse)


def build_collapse_tokenizer_from_spec(tokenizer_spec: str, max_length: int) -> CollapseTokenizer:
    """Build a scratch parent-collapse tokenizer from a PUMA tokenizer JSON spec."""
    p = Path(tokenizer_spec)
    if p.suffix != ".json" or not p.is_file():
        raise ValueError(
            "--parent-collapse requires a PUMA tokenizer .json path (with its family JSON "
            f"sibling); got {tokenizer_spec!r}."
        )
    from plm_benchmark.tokenizers import resolve_tokenizer

    full_tok = resolve_tokenizer(tokenizer_spec, max_length=max_length)
    genealogy = load_genealogy(tokenizer_spec)
    return CollapseTokenizer.from_puma(full_tok, genealogy, max_length)
