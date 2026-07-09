"""Tokenization for PUMA parent-collapsed (`_PC`) checkpoints.

Such a model is trained on a *reduced* vocab (parents + singletons + specials),
but segmentation is done with the *full* PUMA vocab and then remapped child->parent
(see plm_train.build_puma_collapse / _CropDatasetBase.encode). The reduced tokenizer
saved as the checkpoint's main tokenizer is character-level and CANNOT reproduce this,
so plm_train also persists `full_tokenizer/` + `collapse.npy` for inference.

``CollapseTokenizer`` reproduces the training-time "tokenize-with-full -> collapse ->
reduced ids" path. It implements only the slice of the tokenizer API the pipeline uses
(``__call__`` + a few special-token attributes), so it is a drop-in for the collators.
"""

from __future__ import annotations

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


class CollapseTokenizer:
    """Full-vocab segmentation -> child->parent remap -> reduced (model) ids."""

    def __init__(self, checkpoint: str):
        self.checkpoint = str(checkpoint)
        p = Path(checkpoint)
        self.full_tok = AutoTokenizer.from_pretrained(str(p / FULL_TOKENIZER_DIR))
        self.reduced_tok = AutoTokenizer.from_pretrained(str(p))  # model id space + specials
        self.collapse = np.load(p / COLLAPSE_FILE).astype(np.int64)

        self.cls_token_id = self.reduced_tok.cls_token_id
        self.eos_token_id = self.reduced_tok.eos_token_id
        self.pad_token_id = self.reduced_tok.pad_token_id
        self.model_max_length = self.reduced_tok.model_max_length
        self.model_input_names = ["input_ids", "attention_mask"]

        n_full = len(self.full_tok.get_vocab())
        if len(self.collapse) != n_full:
            raise ValueError(
                f"collapse.npy length {len(self.collapse)} != full_tokenizer vocab {n_full} "
                f"in {checkpoint}"
            )

    def __len__(self) -> int:
        return len(self.reduced_tok.get_vocab())

    def validate(self, model_vocab_size: int) -> None:
        """Guarantee every remapped id is a valid embedding index (fail loud, not silent)."""
        mx = int(self.collapse.max()) if len(self.collapse) else -1
        if mx >= model_vocab_size:
            raise ValueError(
                f"collapse maps to id {mx} >= model vocab_size {model_vocab_size} "
                f"({self.checkpoint}); reduced tokenizer / model mismatch."
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
        """Persist as a valid _PC checkpoint dir (Trainer calls this when checkpointing)."""
        dest = Path(save_directory)
        dest.mkdir(parents=True, exist_ok=True)
        self.reduced_tok.save_pretrained(str(dest))
        self.full_tok.save_pretrained(str(dest / FULL_TOKENIZER_DIR))
        np.save(dest / COLLAPSE_FILE, self.collapse)
