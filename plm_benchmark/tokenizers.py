"""Tokenizer resolution for the scratch baseline (and optional overrides).

A ``--tokenizer`` spec may be:

* ``esm2``                     -> the ESM-2 tokenizer (identical across all sizes)
* ``/path/to/tokenizer.json``  -> a raw ``tokenizers`` JSON (PUMA/BPE), wrapped
                                  with ESM-2 special tokens + <cls>..<eos> template
* ``/path/to/dir`` or hub id   -> ``AutoTokenizer.from_pretrained``
"""

from __future__ import annotations

from pathlib import Path

REFERENCE_TOKENIZER = "facebook/esm2_t6_8M_UR50D"
ESM2_SPECIALS = {
    "cls_token": "<cls>",
    "pad_token": "<pad>",
    "eos_token": "<eos>",
    "unk_token": "<unk>",
    "mask_token": "<mask>",
}


def tokenizer_label(spec: str | None) -> str:
    """Short, path-safe label used to name scratch runs. No heavy imports."""
    if not spec:
        return "notok"
    if spec.lower() == "esm2":
        return "esm2"
    p = Path(spec)
    if p.suffix == ".json":
        stem = p.stem
        return stem[3:] if stem.startswith("hf_") else stem
    if p.exists():
        return p.name
    return spec.rstrip("/").split("/")[-1]


def resolve_tokenizer(spec: str | None, *, max_length: int = 1024):
    if spec is None:
        return None
    from transformers import AutoTokenizer

    if spec.lower() == "esm2":
        return AutoTokenizer.from_pretrained(REFERENCE_TOKENIZER)
    p = Path(spec)
    if p.is_dir():
        return AutoTokenizer.from_pretrained(str(p))
    if p.is_file() and p.suffix == ".json":
        return _build_json_tokenizer(str(p), max_length)
    return AutoTokenizer.from_pretrained(spec)


def _build_json_tokenizer(path: str, max_length: int):
    """Wrap a raw tokenizers JSON with ESM-2 specials and a <cls>..<eos> template."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    backend = Tokenizer.from_file(path)
    backend.add_special_tokens(list(ESM2_SPECIALS.values()))
    cls_t, eos_t = ESM2_SPECIALS["cls_token"], ESM2_SPECIALS["eos_token"]
    backend.post_processor = TemplateProcessing(
        single=f"{cls_t} $A {eos_t}",
        pair=f"{cls_t} $A {eos_t} $B:1 {eos_t}:1",
        special_tokens=[(cls_t, backend.token_to_id(cls_t)), (eos_t, backend.token_to_id(eos_t))],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, model_max_length=max_length, **ESM2_SPECIALS
    )
