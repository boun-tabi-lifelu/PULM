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
files (one streaming pass over the source table), MMseqs2 is invoked via its
CLI, the m8 hit table is parsed, and the temporary files (FASTA + MMseqs2
tmp/result dirs) are cleaned up at the end.

GPU acceleration (default): with a GPU-enabled MMseqs2 build the homology search
runs on a single GPU (the small validation set is built into a padded GPU
database; the huge train query is streamed through the GPU prefilter). The
device is pinned via CUDA_VISIBLE_DEVICES (--gpu_devices, default '0' = first
GPU). Pass --no_gpu to fall back to CPU `easy-search`.

Example
-------
    # GPU on the first device, all CPU threads for alignment/IO
    python scripts/prepare_uniref50_splits.py \
        --db /cta/share/users/uniprot/uniref/uniref_2024_06/uniref50_representatives.db \
        --source_table uniref50_representatives \
        --gpu_devices 0 --threads 256 --tmp_dir /scratch/$USER/mmseqs_tmp

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

DEFAULT_DB = "/cta/share/users/uniprot/uniref/uniref50_2026_02/uniref50_representatives.db"


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
                   help="Fraction held out for validation (ESM-2 used ~0.5%%).")
    p.add_argument("--val_size", type=int, default=None,
                   help="Absolute validation size; overrides --val_frac if set.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Debug: only consider the first N source rows.")
    # mmseqs
    p.add_argument("--mmseqs", default="mmseqs", help="MMseqs2 binary / path.")
    p.add_argument("--threads", type=int, default=max(1, os.cpu_count() or 1),
                   help="CPU threads (defaults to all logical CPUs).")
    # GPU acceleration (requires a GPU-enabled MMseqs2 build)
    p.add_argument("--gpu", dest="gpu", action="store_true", default=True,
                   help="Use GPU-accelerated MMseqs2 (default; needs a GPU build).")
    p.add_argument("--no_gpu", dest="gpu", action="store_false",
                   help="Force CPU MMseqs2 (easy-search) instead.")
    p.add_argument("--gpu_devices", default="0",
                   help="CUDA device(s) exposed to MMseqs2 (default: first GPU only).")
    p.add_argument("--tmp_dir", default=None,
                   help="Working dir for FASTA + MMseqs2 tmp (default: system temp).")
    p.add_argument("--identity_cutoff", type=float, default=0.5,
                   help="Drop train seqs hitting a val seq at >= this identity.")
    p.add_argument("--min_seq_id", default="0.5")
    p.add_argument("--alignment_mode", default="3")
    p.add_argument("--max_seqs", default="300")
    p.add_argument("--sensitivity", default="7", help="MMseqs2 -s (CPU prefilter).")
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


def choose_val_rowids(conn, table: str, val_frac: float, val_size_arg,
                      limit, seed: int):
    """Pick the validation rowids without materialising the whole table.
    Returns (n_considered, val_size, val_rowids:set)."""
    mn, mx, cnt = conn.execute(
        f"SELECT MIN(rowid), MAX(rowid), COUNT(*) FROM {table}").fetchone()
    if cnt == 0:
        raise RuntimeError(f"Source table {table!r} is empty.")
    n = min(cnt, limit) if limit else cnt
    contiguous = (mn == 1 and mx == cnt)
    val_size = val_size_arg if val_size_arg is not None else int(round(n * val_frac))
    val_size = max(1, min(val_size, n - 1))

    rng = np.random.default_rng(seed)
    if contiguous:                                   # rowids are 1..cnt
        val_rowids = set((rng.choice(n, val_size, replace=False) + 1).tolist())
    else:                                            # need the actual rowids
        logger.info("Non-contiguous rowids; loading the rowid list ...")
        sql = f"SELECT rowid FROM {table}" + (f" LIMIT {int(limit)}" if limit else "")
        rowids = np.fromiter((r[0] for r in conn.execute(sql)), dtype=np.int64, count=n)
        pos = rng.choice(n, val_size, replace=False)
        val_rowids = set(rowids[pos].tolist())
    return n, val_size, val_rowids


# --------------------------------------------------------------------------- #
#  STEP 1 -- PARTITION + dump FASTA  (single streaming scan over the source)
# --------------------------------------------------------------------------- #
def partition_and_dump(conn, args, work_dir: str):
    """Choose the validation rowids, dump train/val FASTA, return paths + counts."""
    n, val_size, val_rowids = choose_val_rowids(
        conn, args.source_table, args.val_frac, args.val_size, args.limit, args.seed)
    logger.info("Source rows: %d | validation: %d | train (pre-homology): %d",
                n, val_size, n - val_size)

    train_fa = os.path.join(work_dir, "train.fasta")
    val_fa = os.path.join(work_dir, "val.fasta")
    sql = f"SELECT rowid, entry_id, sequence FROM {args.source_table}"
    if args.limit:
        sql += f" LIMIT {int(args.limit)}"

    n_train = n_val = 0
    t0 = time.time()
    cur = conn.cursor()
    cur.execute(sql)
    with open(train_fa, "w", buffering=1 << 20) as ftr, \
            open(val_fa, "w", buffering=1 << 20) as fva:
        for rowid, entry_id, seq in cur:            # cursor streams row by row
            if not seq:
                continue
            if rowid in val_rowids:
                fva.write(f">{entry_id}\n{seq}\n"); n_val += 1
            else:
                ftr.write(f">{entry_id}\n{seq}\n"); n_train += 1
    logger.info("Dumped FASTA in %.1fs -> train=%d val=%d (%s, %s)",
                time.time() - t0, n_train, n_val, train_fa, val_fa)
    return train_fa, val_fa, n_train, n_val


# --------------------------------------------------------------------------- #
#  STEP 2 -- MMseqs2 homology search train(query) vs val(target)
# --------------------------------------------------------------------------- #
def _run(cmd, env):
    import subprocess
    logger.info("  $ %s", " ".join(map(str, cmd)))
    t0 = time.time()
    rc = subprocess.run(cmd, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"MMseqs2 step failed (exit {rc}): {' '.join(map(str, cmd))}")
    logger.info("    done in %.1fs", time.time() - t0)


def run_mmseqs(args, train_fa: str, val_fa: str, work_dir: str) -> Set[str]:
    """Return the set of TRAIN entry_ids that hit a VAL seq at >= cutoff identity.

    Search direction: query = TRAIN, target = VALIDATION. With --gpu the small
    validation target is built into a padded GPU database and the (huge) train
    query is streamed through the GPU prefilter on the first device only."""
    result_m8 = os.path.join(work_dir, "train_vs_val.m8")
    mmseqs_tmp = os.path.join(work_dir, "mmseqs_tmp")
    os.makedirs(mmseqs_tmp, exist_ok=True)

    # shared search parameters (per Meier et al.)
    search_params = [
        "--min-seq-id", str(args.min_seq_id),
        "--alignment-mode", str(args.alignment_mode),
        "--max-seqs", str(args.max_seqs),
        "-s", str(args.sensitivity),
        "-c", str(args.coverage),
        "--cov-mode", str(args.cov_mode),
        "--threads", str(args.threads),
    ]
    fmt = ["--format-output", "query,target,fident"]

    env = dict(os.environ)
    t0 = time.time()
    if args.gpu:
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_devices)
        logger.info("GPU-accelerated MMseqs2 (CUDA_VISIBLE_DEVICES=%s, threads=%d)",
                    args.gpu_devices, args.threads)
        query_db = os.path.join(work_dir, "queryDB")     # train
        target_db = os.path.join(work_dir, "targetDB")   # val
        target_gpu = os.path.join(work_dir, "targetDB_gpu")
        result_db = os.path.join(work_dir, "resultDB")
        _run([args.mmseqs, "createdb", train_fa, query_db], env)
        _run([args.mmseqs, "createdb", val_fa, target_db], env)
        # pad the (small) validation target for the GPU prefilter
        _run([args.mmseqs, "makepaddedseqdb", target_db, target_gpu], env)
        _run([args.mmseqs, "search", query_db, target_gpu, result_db, mmseqs_tmp,
              "--gpu", "1"] + search_params, env)
        _run([args.mmseqs, "convertalis", query_db, target_gpu, result_db, result_m8]
             + fmt + ["--threads", str(args.threads)], env)
    else:
        logger.info("CPU MMseqs2 easy-search (threads=%d)", args.threads)
        # easy-search consumes FASTA directly: query=train, target=val.
        _run([args.mmseqs, "easy-search", train_fa, val_fa, result_m8, mmseqs_tmp]
             + search_params + fmt, env)
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
