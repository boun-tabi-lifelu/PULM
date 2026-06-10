#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Prepare the UniRef50 train / validation splits for pLM (continual) pretraining,
replicating the ESM-2 / Meier-et-al. data-handling procedure.

Pipeline (run ONCE; the result is persisted as two new tables)
--------------------------------------------------------------
  1. PARTITION : randomly hold out `--val_frac` (default 0.5% ~= 250k) of the
                 sequences as the validation set; the rest is the train set.
  2. HOMOLOGY  : remove train sequences that are homologous to ANY validation
     REDUCTION   sequence. We run an MMseqs2 search with the TRAIN set as the
                 query and the VALIDATION set as the target:
                     --min-seq-id 0.5 --alignment-mode 3 --max-seqs 300
                     -s 7 -c 0.8 --cov-mode 0
                 Every train sequence that hits a validation sequence at
                 >= 50% identity (`--identity_cutoff`) is dropped from train.
  3. PERSIST   : write the final splits back into the SAME sqlite db as two
                 tables, `plm_train` and `plm_validation`, each storing
                 (entry_id, sequence), and CREATE an INDEX on entry_id for both.

Because MMseqs2 only consumes FASTA, the splits are dumped to temporary FASTA
files, MMseqs2 is invoked via its CLI, the m8 hit table is parsed, and the
temporary files (FASTA + MMseqs2 tmp/result dirs) are cleaned up at the end.

Example
-------
    python scripts/prepare_uniref50_splits.py \
        --db /cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db \
        --source_table uniref50_representatives \
        --threads 32 --tmp_dir /scratch/$USER/mmseqs_tmp

Re-running is safe: pass --overwrite to drop existing plm_train/plm_validation.
Use --limit N for a quick end-to-end smoke test on N source rows.
"""

import argparse
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from typing import Iterator, Set, Tuple

import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("prep_uniref50")

DEFAULT_DB = "/cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db"


# --------------------------------------------------------------------------- #
#  CLI
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description="Build homology-reduced UniRef50 pLM train/val splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # db / tables
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite db (read + write).")
    p.add_argument("--source_table", default="uniref50_representatives",
                   help="Table with (entry_id, sequence, ...) to split.")
    p.add_argument("--train_table", default="plm_train")
    p.add_argument("--val_table", default="plm_validation")
    p.add_argument("--overwrite", action="store_true",
                   help="Drop existing train/val tables before writing.")
    # partition
    p.add_argument("--val_frac", type=float, default=0.005,
                   help="Fraction held out for validation (ESM-2 used ~0.5%).")
    p.add_argument("--val_size", type=int, default=None,
                   help="Absolute validation size; overrides --val_frac if set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: only consider the first N source rows.")
    # mmseqs
    p.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 binary / path.")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1))
    p.add_argument("--tmp_dir", default=None,
                   help="Working dir for FASTA + MMseqs2 tmp (default: system temp).")
    p.add_argument("--identity_cutoff", type=float, default=0.5,
                   help="Drop train seqs hitting a val seq at >= this identity.")
    p.add_argument("--min_seq_id", default="0.5")
    p.add_argument("--alignment_mode", default="3")
    p.add_argument("--max_seqs", default="300")
    p.add_argument("--sensitivity", default="7", help="MMseqs2 -s.")
    p.add_argument("--coverage", default="0.8", help="MMseqs2 -c.")
    p.add_argument("--cov_mode", default="0")
    p.add_argument("--skip_homology", action="store_true",
                   help="Debug: skip the MMseqs2 step (no train pruning).")
    p.add_argument("--keep_tmp", action="store_true",
                   help="Don't delete the temporary FASTA / MMseqs2 files.")
    return p.parse_args()


# --------------------------------------------------------------------------- #
#  SQLITE HELPERS
# --------------------------------------------------------------------------- #
def table_exists(conn, name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    ).fetchone()
    return row is not None


def source_rowids(conn, table: str, limit) -> np.ndarray:
    """Return the source rowids to split (fast contiguous path when possible)."""
    mn, mx, cnt = conn.execute(
        f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table}").fetchone()
    if cnt == 0:
        raise RuntimeError(f"Source table {table!r} is empty.")
    if mn == 1 and mx == cnt:                       # contiguous 1..N
        rowids = np.arange(1, cnt + 1, dtype=np.int64)
    else:
        logger.info("Non-contiguous rowids; loading the full rowid list ...")
        rowids = np.fromiter(
            (r[0] for r in conn.execute(f"SELECT rowid FROM {table}")),
            dtype=np.int64, count=cnt)
    if limit:
        rowids = rowids[:limit]
    return rowids


def stream_rows(conn, table: str, rowids: np.ndarray, batch: int = 50_000
                ) -> Iterator[Tuple[int, str, str]]:
    """Yield (rowid, entry_id, sequence) for the given rowids, in batches."""
    cur = conn.cursor()
    for i in range(0, len(rowids), batch):
        chunk = rowids[i:i + batch].tolist()
        qmarks = ",".join("?" * len(chunk))
        cur.execute(
            f"SELECT rowid, entry_id, sequence FROM {table} "
            f"WHERE rowid IN ({qmarks})", chunk)
        for row in cur.fetchall():
            yield row


# --------------------------------------------------------------------------- #
#  STEP 1 -- PARTITION + dump FASTA
# --------------------------------------------------------------------------- #
def write_fasta_handle(fh, entry_id: str, sequence: str):
    fh.write(f">{entry_id}\n{sequence}\n")


def partition_and_dump(conn, args, work_dir: str):
    """Choose the validation rowids, dump train/val FASTA, return paths + counts."""
    rowids = source_rowids(conn, args.source_table, args.limit)
    n = len(rowids)
    val_size = args.val_size if args.val_size is not None else int(round(n * args.val_frac))
    val_size = max(1, min(val_size, n - 1))
    logger.info("Source rows: %d | validation: %d | train (pre-homology): %d",
                n, val_size, n - val_size)

    rng = np.random.default_rng(args.seed)
    val_pos = rng.choice(n, size=val_size, replace=False)
    val_rowids: Set[int] = set(rowids[val_pos].tolist())

    train_fa = os.path.join(work_dir, "train.fasta")
    val_fa = os.path.join(work_dir, "val.fasta")
    n_train = n_val = 0
    t0 = time.time()
    with open(train_fa, "w") as ftr, open(val_fa, "w") as fva:
        for rowid, entry_id, seq in stream_rows(conn, args.source_table, rowids):
            if not seq:
                continue
            if rowid in val_rowids:
                write_fasta_handle(fva, entry_id, seq); n_val += 1
            else:
                write_fasta_handle(ftr, entry_id, seq); n_train += 1
    logger.info("Dumped FASTA in %.1fs -> train=%d val=%d (%s, %s)",
                time.time() - t0, n_train, n_val, train_fa, val_fa)
    return train_fa, val_fa, n_train, n_val


# --------------------------------------------------------------------------- #
#  STEP 2 -- MMseqs2 homology search train(query) vs val(target)
# --------------------------------------------------------------------------- #
def run_mmseqs(args, train_fa: str, val_fa: str, work_dir: str) -> Set[str]:
    """Return the set of TRAIN entry_ids that hit a VAL seq at >= cutoff identity."""
    result_m8 = os.path.join(work_dir, "train_vs_val.m8")
    mmseqs_tmp = os.path.join(work_dir, "mmseqs_tmp")
    os.makedirs(mmseqs_tmp, exist_ok=True)

    # easy-search consumes FASTA directly: query=train, target=val.
    cmd = [
        args.mmseqs, "easy-search", train_fa, val_fa, result_m8, mmseqs_tmp,
        "--min-seq-id", str(args.min_seq_id),
        "--alignment-mode", str(args.alignment_mode),
        "--max-seqs", str(args.max_seqs),
        "-s", str(args.sensitivity),
        "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode),
        "--threads", str(args.threads),
        # m8 cols: query target fident alnlen mismatch gapopen qstart qend tstart tend evalue bits
        "--format-output", "query,target,fident",
    ]
    logger.info("Running MMseqs2:\n  %s", " ".join(cmd))
    import subprocess
    t0 = time.time()
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise RuntimeError(f"MMseqs2 failed (exit {proc.returncode}). "
                           f"Check that '{args.mmseqs}' is installed and on PATH.")
    logger.info("MMseqs2 finished in %.1fs -> %s", time.time() - t0, result_m8)

    to_remove: Set[str] = set()
    if not os.path.isfile(result_m8):
        logger.warning("No m8 output produced; assuming zero homologous hits.")
        return to_remove
    cutoff = args.identity_cutoff
    n_lines = 0
    with open(result_m8) as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            n_lines += 1
            query_id, _target, fident = parts[0], parts[1], parts[2]
            try:
                pid = float(fident)
            except ValueError:
                continue
            if pid > 1.0:           # some builds report identity as a percentage
                pid /= 100.0
            if pid >= cutoff:
                to_remove.add(query_id)
    logger.info("Parsed %d hits -> %d unique train sequences to remove "
                "(identity >= %.2f)", n_lines, len(to_remove), cutoff)
    return to_remove


# --------------------------------------------------------------------------- #
#  STEP 3 -- write plm_train / plm_validation tables (+ index)
# --------------------------------------------------------------------------- #
def write_split_tables(conn, args, train_fa, val_fa, to_remove: Set[str]):
    """Populate train/val tables by streaming the dumped FASTA (so we never
    re-query the source for the same data twice)."""
    for tbl in (args.train_table, args.val_table):
        if table_exists(conn, tbl):
            if not args.overwrite:
                raise RuntimeError(
                    f"Table {tbl!r} already exists. Re-run with --overwrite.")
            logger.info("Dropping existing table %s", tbl)
            conn.execute(f"DROP TABLE {tbl}")
    for tbl in (args.train_table, args.val_table):
        conn.execute(
            f"CREATE TABLE {tbl} (entry_id TEXT NOT NULL, sequence TEXT NOT NULL)")
    conn.commit()

    n_train = _insert_from_fasta(conn, args.train_table, train_fa, skip=to_remove)
    n_val = _insert_from_fasta(conn, args.val_table, val_fa, skip=None)

    # Index entry_id on BOTH split tables (fast lookups during training loops).
    for tbl in (args.train_table, args.val_table):
        logger.info("Creating index on %s(entry_id)", tbl)
        conn.execute(f"CREATE INDEX idx_{tbl}_entry_id ON {tbl}(entry_id)")
    conn.commit()
    logger.info("Wrote %s=%d rows, %s=%d rows (post-homology-reduction).",
                args.train_table, n_train, args.val_table, n_val)
    return n_train, n_val


def _iter_fasta(path: str) -> Iterator[Tuple[str, str]]:
    entry_id, chunks = None, []
    with open(path) as fh:
        for line in fh:
            if line.startswith(">"):
                if entry_id is not None:
                    yield entry_id, "".join(chunks)
                entry_id = line[1:].strip().split()[0]
                chunks = []
            else:
                chunks.append(line.strip())
        if entry_id is not None:
            yield entry_id, "".join(chunks)


def _insert_from_fasta(conn, table: str, fasta: str, skip) -> int:
    skip = skip or set()
    cur = conn.cursor()
    cur.execute("BEGIN")
    n, buf = 0, []
    for entry_id, seq in _iter_fasta(fasta):
        if entry_id in skip:
            continue
        buf.append((entry_id, seq))
        if len(buf) >= 50_000:
            cur.executemany(f"INSERT INTO {table}(entry_id, sequence) VALUES (?,?)", buf)
            n += len(buf); buf.clear()
    if buf:
        cur.executemany(f"INSERT INTO {table}(entry_id, sequence) VALUES (?,?)", buf)
        n += len(buf)
    conn.commit()
    return n


# --------------------------------------------------------------------------- #
#  MAIN
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()
    if not os.path.isfile(args.db):
        sys.exit(f"DB not found: {args.db}")

    work_dir = args.tmp_dir or tempfile.mkdtemp(prefix="uniref50_splits_")
    os.makedirs(work_dir, exist_ok=True)
    logger.info("Working dir: %s", work_dir)

    conn = sqlite3.connect(args.db)
    try:
        if not table_exists(conn, args.source_table):
            sys.exit(f"Source table {args.source_table!r} not in {args.db}")

        train_fa, val_fa, n_tr, n_va = partition_and_dump(conn, args, work_dir)

        if args.skip_homology:
            logger.warning("--skip_homology set: NOT pruning train by homology.")
            to_remove: Set[str] = set()
        else:
            to_remove = run_mmseqs(args, train_fa, val_fa, work_dir)

        write_split_tables(conn, args, train_fa, val_fa, to_remove)
    finally:
        conn.close()
        if not args.keep_tmp and not args.tmp_dir:
            shutil.rmtree(work_dir, ignore_errors=True)
            logger.info("Cleaned up temporary dir %s", work_dir)
        elif not args.keep_tmp:
            # user gave an explicit tmp_dir: only remove the files we created
            for f in ("train.fasta", "val.fasta", "train_vs_val.m8"):
                try:
                    os.remove(os.path.join(work_dir, f))
                except OSError:
                    pass
            shutil.rmtree(os.path.join(work_dir, "mmseqs_tmp"), ignore_errors=True)
            logger.info("Cleaned up temporary files in %s", work_dir)

    logger.info("Done. Tables '%s' and '%s' are ready in %s.",
                args.train_table, args.val_table, args.db)


if __name__ == "__main__":
    main()
