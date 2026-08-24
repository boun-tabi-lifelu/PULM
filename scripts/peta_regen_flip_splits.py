#!/usr/bin/env python3
"""Regenerate FLIP train/valid/test splits from the authoritative raw/ FASTAs.

The shipped PETA JSONs for flip/{aav,gb1,meltome} are broken:
  * valid.json is a byte-identical COPY of test.json  -> val/test leak
    (early stopping + best-checkpoint selection happen on the test set)
  * train.json is SET=train INCLUDING the VALIDATION=True rows

raw/<split>.fasta is the source of truth. Canonical FLIP semantics:
    SET=train & VALIDATION=False -> train
    SET=train & VALIDATION=True  -> validation
    SET=test                     -> test
    anything else (e.g. SET=nan) -> excluded from this split

This deletes the existing {train,valid,test}.json for each flip split and
rewrites them from raw/, preserving the loader's schema:
    [{"sequence": <str>, "label": [<float>]}, ...]

Usage:
    python scripts/regen_flip_splits.py --dry-run     # preview counts, write nothing
    python scripts/regen_flip_splits.py               # delete + regenerate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DATASETS = ("aav", "gb1", "meltome")
HEADER_KV = re.compile(r"(\w+)=([^\s]+)")

# Regression pins: rebuilt (train, val, test) counts must match FLIP's reported numbers.
EXPECTED_COUNTS = {
    ("gb1", "three_vs_rest"): (2691, 299, 5743),
    ("aav", "two_vs_many"): (28626, 3181, 50776),
}


def parse_fasta(path: Path) -> list[tuple[dict, str]]:
    """Yield (header_kv, sequence) for each record."""
    recs: list[tuple[str, list[str]]] = []
    header, chunks = None, []
    for line in path.open():
        line = line.rstrip("\n")
        if line.startswith(">"):
            if header is not None:
                recs.append((header, chunks))
            header, chunks = line[1:].strip(), []
        else:
            chunks.append(line.strip())
    if header is not None:
        recs.append((header, chunks))
    return [(dict(HEADER_KV.findall(h)), "".join(c)) for h, c in recs]


def partition(records: list[tuple[dict, str]]):
    """Canonical FLIP partition. Returns (train, valid, test, n_excluded)."""
    train, valid, test, excluded = [], [], [], 0
    for kv, seq in records:
        s, v = kv.get("SET"), kv.get("VALIDATION")
        if "TARGET" not in kv or v is None:
            raise ValueError(f"Malformed header (need TARGET and VALIDATION): {kv}")
        row = {"sequence": seq, "label": [float(kv["TARGET"])]}
        if s == "train" and v == "False":
            train.append(row)
        elif s == "train" and v == "True":
            valid.append(row)
        elif s == "test":
            test.append(row)
        else:
            excluded += 1  # e.g. SET=nan -> not part of this split
    return train, valid, test, excluded


def overlaps(train, valid, test) -> tuple[int, int, int]:
    t = {r["sequence"] for r in train}
    v = {r["sequence"] for r in valid}
    e = {r["sequence"] for r in test}
    return len(t & v), len(t & e), len(v & e)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default=None, help="ft_datasets dir (default: resolved PETA_DATA_DIR)")
    p.add_argument("--datasets", default=",".join(DATASETS), help="Comma-separated: aav,gb1,meltome")
    p.add_argument("--dry-run", action="store_true", help="Report only; do not delete/write.")
    args = p.parse_args()

    if args.data_dir:
        base = Path(args.data_dir)
    else:
        from plm_benchmark.config import PETA_DATA_DIR

        base = PETA_DATA_DIR
    flip = base / "flip"
    if not flip.is_dir():
        raise SystemExit(f"flip/ not found under {base}. Pass --data-dir <ft_datasets>.")

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    print(f"{'split':30} {'train':>7} {'val':>6} {'test':>7} {'excl':>7}   warnings")
    failures, total = [], 0

    for ds in datasets:
        raw_dir = flip / ds / "raw"
        if not raw_dir.is_dir():
            print(f"  SKIP {ds}: no raw/ dir")
            continue
        for fasta in sorted(raw_dir.glob("*.fasta")):
            split = fasta.stem
            out_dir = flip / ds / split
            train, valid, test, excluded = partition(parse_fasta(fasta))

            warn = []
            exp = EXPECTED_COUNTS.get((ds, split))
            got = (len(train), len(valid), len(test))
            if exp and got != exp:
                failures.append(f"{ds}/{split}: expected {exp}, got {got}")
                warn.append("COUNT-MISMATCH")
            if not valid:
                warn.append("EMPTY-VAL")
            elif len(valid) < 50:
                warn.append(f"TINY-VAL({len(valid)})")
            tv, tt, vt = overlaps(train, valid, test)
            if vt:
                warn.append(f"val&test={vt}")
            if tt:
                warn.append(f"train&test={tt}")
            if tv:
                warn.append(f"train&val={tv}")

            print(f"  {ds+'/'+split:28} {len(train):7} {len(valid):6} {len(test):7} {excluded:7}   {' '.join(warn)}")

            if args.dry_run:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for name, rows in (("train", train), ("valid", valid), ("test", test)):
                path = out_dir / f"{name}.json"
                if path.exists():
                    path.unlink()  # delete the broken file first
                with path.open("w") as fh:
                    json.dump(rows, fh)
            total += 1

    if failures:
        raise SystemExit("\nFAILED count assertions:\n  " + "\n  ".join(failures))
    if args.dry_run:
        print("\nDry run: nothing written.")
    else:
        print(f"\nRegenerated {total} splits from raw/. valid is now SET=train&VALIDATION=True (no test leak).")
    print("Note: train&val / train&test overlaps above are duplicate SEQUENCES as FLIP ships them "
          "(records are disjoint by construction).")


if __name__ == "__main__":
    main()
