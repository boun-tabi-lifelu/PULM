"""Tokenizer resolution for the scratch baseline (and optional overrides).

A ``--tokenizer`` spec may be:

* ``aa``                       -> the amino-acid (ESM-2) tokenizer, identical across
                                  all ESM-2 sizes (``esm2`` is accepted as an alias)
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

# Tokenizers live under a uniref50_<date> folder -> models trained on "all".
# This matches the training-data suffix in the PLM run slugs (ESM2_35M_..._all),
# so a scratch run and the PLM using the same tokenizer share one tokenizer label.
TRAINING_DATA_SUFFIX = "all"


def _nodot(v: str) -> str:
    return str(v).replace(".", "")  # 0.7 -> 07, 0.05 -> 005 (matches plm_train)


def _json_tokenizer_label(path: Path) -> str:
    """Convert a PUMA/BPE tokenizer JSON path to the PLM run-slug tokenizer label.

    Mirrors plm_train.build_run_slug's tokenizer part (uniref50 is static -> omitted):
      .../blosum62/hf_uniref50_mutbpe_0.7_3_12_0.05_12800.json -> PUMA_blosum62_07_005_12800_all
      .../bpe/hf_uniref50_bpe_12800.json                       -> BPE_12800_all
    """
    subfolder = path.parent.name  # 'bpe' or the PUMA substitution matrix (e.g. blosum62)
    stem = path.stem[3:] if path.stem.startswith("hf_") else path.stem
    parts = stem.split("_")
    vocab = parts[-1]
    if subfolder.lower() == "bpe":
        core = f"BPE_{vocab}"
    else:
        # PUMA: cutoff and min-mutation-freq are the two floats; min/max length ignored.
        floats = [p for p in parts if "." in p]
        cutoff = _nodot(floats[0]) if len(floats) >= 1 else "NA"
        minfreq = _nodot(floats[1]) if len(floats) >= 2 else "NA"
        core = f"PUMA_{subfolder}_{cutoff}_{minfreq}_{vocab}"
    return f"{core}_{TRAINING_DATA_SUFFIX}"


def tokenizer_label(spec: str | None) -> str:
    """Short, path-safe label used to name scratch runs. No heavy imports."""
    if not spec:
        return "notok"
    if spec.lower() in ("aa", "esm2"):
        return "AA"  # uppercase to match hub/PULM AA-tokenizer naming
    p = Path(spec)
    if p.suffix == ".json":
        return _json_tokenizer_label(p)
    if p.exists():
        return p.name
    return spec.rstrip("/").split("/")[-1]


def resolve_tokenizer(spec: str | None, *, max_length: int = 1024):
    if spec is None:
        return None
    from transformers import AutoTokenizer

    if spec.lower() in ("aa", "esm2"):
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
