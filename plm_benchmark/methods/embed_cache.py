"""Per-residue encoder-output cache for the frozen-encoder (`embed_head`) fast path.

`embed_head` freezes the encoder, so its per-residue outputs are identical every
epoch — recomputing them 50x is pure waste. This caches `last_hidden_state` once
per (model, tokenizer, task, split) and trains the attention1d head off the cache.

Exactness: the encoder is already forced to eval() when frozen, so the cached states
are byte-for-byte what the on-the-fly path would produce. The head, loss, metrics,
early stopping and best-val selection are unchanged.

Storage is ragged (no padding waste): a memmapped [total_tokens, H] fp16 array plus
int64 offsets, so a 20 GB cache never lands in RAM. Only train+valid are cached; test
is scored once at the end through the real encoder.
"""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

CACHE_DTYPE = np.float16
META_FILE = "meta.json"


def _subset_dir(cache_dir: Path, model: str, tokenizer: str, task: str, split: str, subset: str) -> Path:
    return Path(cache_dir) / model / tokenizer / task.lower() / (split or "default") / subset


def _meta(checkpoint: str, max_length: int, hidden: int, n: int, total: int) -> dict:
    return {
        "checkpoint": str(checkpoint),
        "max_length": int(max_length),
        "hidden_size": int(hidden),
        "n": int(n),
        "total_tokens": int(total),
        "dtype": np.dtype(CACHE_DTYPE).name,
    }


def estimate_bytes(seq_lens, hidden_size: int, max_length: int) -> int:
    """Upper-bound cache size: sum(min(len+2, max_length)) * H * itemsize."""
    per = sum(min(int(n) + 2, max_length) for n in seq_lens)
    return per * hidden_size * np.dtype(CACHE_DTYPE).itemsize


class CachedHidden:
    """Ragged memmapped store of per-residue encoder outputs."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.meta = json.loads((self.path / META_FILE).read_text())
        self.offsets = np.load(self.path / "offsets.npy")
        self.hidden = np.memmap(
            self.path / "hidden.npy",
            dtype=np.dtype(self.meta["dtype"]),
            mode="r",
            shape=(self.meta["total_tokens"], self.meta["hidden_size"]),
        )

    def __len__(self) -> int:
        return int(self.meta["n"])

    def get(self, i: int) -> np.ndarray:
        return np.asarray(self.hidden[self.offsets[i] : self.offsets[i + 1]])

    @staticmethod
    def matches(path: Path, checkpoint: str, max_length: int) -> bool:
        meta_path = Path(path) / META_FILE
        if not meta_path.is_file():
            return False
        try:
            m = json.loads(meta_path.read_text())
        except Exception:
            return False
        return m.get("checkpoint") == str(checkpoint) and m.get("max_length") == int(max_length)


@torch.no_grad()
def build_cache(
    path: Path,
    encoder,
    tokenizer,
    sequences,
    *,
    checkpoint: str,
    max_length: int,
    batch_size: int,
    device,
    amp_dtype=None,
) -> None:
    """Run the frozen encoder once over `sequences` and write the ragged cache.

    `amp_dtype` must be the SAME autocast dtype training uses (bf16), so the cached
    states equal what the on-the-fly path would produce. fp16 storage is exact for
    bf16 values (10 >= 7 mantissa bits); the guard below catches the one fp16 risk,
    which is range (|x| > 65504 -> inf).
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    encoder.to(device).eval()
    hidden_size = encoder.config.hidden_size

    # Pass 1 is fused into the write: collect lengths as we go, grow via a temp file.
    tmp = path / "hidden.tmp.npy"
    offsets = [0]
    written = 0
    with open(tmp, "wb") as fh:
        for start in range(0, len(sequences), batch_size):
            chunk = list(sequences[start : start + batch_size])
            tok = tokenizer(chunk, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
            ids = tok["input_ids"].to(device)
            mask = tok["attention_mask"].to(device)
            ctx = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with ctx:
                out = encoder(input_ids=ids, attention_mask=mask).last_hidden_state  # [B, L, H]
            lens = mask.sum(dim=1).tolist()
            out16 = out.to(torch.float16)
            if not torch.isfinite(out16).all():
                raise ValueError(
                    f"embed cache: non-finite values after the fp16 cast (max|x|="
                    f"{out.abs().max().item():.1f}). Encoder outputs exceed fp16 range (65504) — "
                    f"the cache would be silently corrupted. Re-run without --embed-cache."
                )
            out = out16.cpu().numpy()
            for j, n in enumerate(lens):
                fh.write(out[j, : int(n)].astype(CACHE_DTYPE).tobytes())
                written += int(n)
                offsets.append(written)

    tmp.rename(path / "hidden.npy")
    np.save(path / "offsets.npy", np.asarray(offsets, dtype=np.int64))
    # meta written last: its presence marks the cache complete/valid
    (path / META_FILE).write_text(
        json.dumps(_meta(checkpoint, max_length, hidden_size, len(sequences), written))
    )


class CachedHiddenDataset(torch.utils.data.Dataset):
    """Yields cached per-residue states + label; `length` kept for group_by_length."""

    def __init__(self, store: CachedHidden, labels, lengths):
        self.store = store
        self.labels = list(labels)
        self.lengths = list(lengths)
        if len(self.labels) != len(store):
            raise ValueError(f"cache has {len(store)} rows but {len(self.labels)} labels")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i):
        return {"hidden": self.store.get(i), "labels": self.labels[i], "length": self.lengths[i]}


def collate_cached(batch, label_dtype):
    """Pad ragged cached states to [B, Lmax, H] and build the attention mask."""
    hs = [b["hidden"] for b in batch]
    width = max(h.shape[0] for h in hs)
    H = hs[0].shape[1]
    out = np.zeros((len(hs), width, H), dtype=np.float32)
    mask = np.zeros((len(hs), width), dtype=np.int64)
    for i, h in enumerate(hs):
        out[i, : h.shape[0]] = h
        mask[i, : h.shape[0]] = 1
    return {
        "hidden_states": torch.from_numpy(out),
        "attention_mask": torch.from_numpy(mask),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=label_dtype),
    }
