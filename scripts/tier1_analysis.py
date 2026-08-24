#!/usr/bin/env python3
"""Tier-1 analyses for the tokenizer comparison: is the from-scratch gain the
vocabulary, or just the parameters that come with it?

In the scratch arm the encoder is `d x V` embedding + one small Transformer block,
so the *nominal* model size grows 7x from AA (2.66M params) to V=51200 (19.0M, 86%
of it embedding table). That is an obvious confound for "sub-word beats AA".

But nominal size is the wrong denominator. Only embedding rows for tokens that
actually occur in a task's training split ever receive gradient, and a task like GB1
(variants of one 56-residue protein) touches a few hundred rows whether V is 6400 or
51200. The quantity that matters is therefore the EFFECTIVE embedding capacity,
`n_distinct_tokens_in_train x hidden_dim`, which this script measures.

Three subcommands, all cheap (no training, no GPU):

  tokens    Per (task, split, tokenizer): k-bar (residues/token), distinct tokens
            used, effective embedding parameters, truncation loss at --max-length,
            and the fraction of test tokens never seen in training.
              -> outputs/analysis/token_stats.csv

  paired    Paired per-task significance tests against a baseline tokenizer.
            The aggregate task-balanced means hide that every arm runs the SAME
            tasks; pairing over tasks is both the correct test for this design and
            far more powerful than comparing two aggregate means with seed SDs.
              -> outputs/analysis/paired_tests.csv, task_scores.csv

  capacity  Joins the two: score vs effective embedding parameters. If AA and the
            sub-word arms fall on one curve, capacity explains the gain. If the
            sub-word arms sit ABOVE AA at matched effective capacity, the
            segmentation is carrying information that parameters alone don't.
              -> outputs/analysis/capacity.csv

Examples
--------
    # k-bar / effective vocabulary for the scratch arm's tokenizers
    python scripts/tier1_analysis.py tokens --task everything \
        --tokenizers aa,/tok/bpe/hf_uniref50_bpe_51200.json,\
/tok/blosum62/hf_uniref50_mutbpe_0.7_3_12_0.05_51200.json,\
pc:/tok/blosum62/hf_uniref50_mutbpe_0.7_3_12_0.05_51200.json

    # every tokenizer JSON under a directory tree
    python scripts/tier1_analysis.py tokens --task everything \
        --tokenizers aa --tokenizer-dir /cta/share/users/mutbpe/tokenizers/uniref50_2024_06

    # AA vs every other tokenizer, paired over tasks, scratch arm
    python scripts/tier1_analysis.py paired --method full_ft \
        --model scratch_d320_l1_h8 --baseline AA

    python scripts/tier1_analysis.py capacity --method full_ft --model scratch_d320_l1_h8

Note: `tokens` needs `transformers`/`tokenizers` (to load the tokenizer JSONs) and
the benchmark datasets; `paired` needs only outputs/experiments.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from plm_benchmark.config import OUTPUTS_DIR  # noqa: E402
from plm_benchmark.tasks import (  # noqa: E402
    TASKS,
    load_splits,
    preprocess_sequences,
    resolve_task_splits,
    shares_train_across_splits,
)

ANALYSIS_DIR = OUTPUTS_DIR / "analysis"

# Default exclusion: eSol is reported as an error (MSE), not a score, and is left
# out of the task-balanced tables in the paper. Pass --exclude-tasks "" to keep it
# (the sign is flipped automatically for lower-is-better metrics either way).
DEFAULT_EXCLUDE = "peta_esol"


# --------------------------------------------------------------------------- #
#  TOKENIZERS
#
#  Performance notes. This is CPU/Rust work -- there is no GPU tokenizer, and
#  tokenization itself dominates, so the only real lever is doing FEWER passes:
#   * a PUMA tokenizer and its `pc:` variant segment IDENTICALLY (the collapse only
#     remaps ids), so both rows come from ONE tokenization pass plus a numpy gather
#   * CollapseTokenizer.__call__ is never used here: it is pure Python and pads each
#     batch to its longest sequence, so on Meltome (35k-token proteins) a 256-chunk
#     builds ~9M-element lists. This was the real cost of --include-pc.
#   * splits that share a train set (remote_homology, deeploc_2) scan it once
#   * --jobs runs tokenizers concurrently; the Rust encoder releases the GIL
#   * the counting is written over numpy arrays because the collapse gather needs
#     them anyway -- MEASURED against per-token set.update() it is a wash (0.9-1.2x),
#     so do not mistake it for the speedup. set.update() was already a C-level loop.
# --------------------------------------------------------------------------- #
class TokGroup:
    """One tokenizer, emitting a plain row, a parent-collapsed row, or both."""

    def __init__(self, spec_path: str, want_plain: bool, want_pc: bool, max_length: int):
        from plm_benchmark.tokenizers import resolve_tokenizer, tokenizer_label

        self.spec = spec_path
        self.tok = resolve_tokenizer(spec_path, max_length=max_length)
        self.vocab = len(self.tok)
        self.plain_label = tokenizer_label(spec_path, False) if want_plain else None
        self.pc_label = None
        self.collapse = None
        self.pc_vocab = 0
        if want_pc:
            from plm_benchmark.models.collapse import build_collapse_tokenizer_from_spec

            ct = build_collapse_tokenizer_from_spec(spec_path, max_length)
            self.collapse = np.asarray(ct.collapse, dtype=np.int64)
            self.pc_vocab = ct.reduced_size
            self.pc_label = tokenizer_label(spec_path, True)

    @property
    def labels(self) -> list[str]:
        return [x for x in (self.plain_label, self.pc_label) if x]

    def _flat(self, block: list[str]):
        """(per-sequence token counts, flat id array) for one chunk, no specials."""
        enc = self.tok(block, add_special_tokens=False, truncation=False)["input_ids"]
        lens = np.fromiter((len(x) for x in enc), dtype=np.int64, count=len(enc))
        total = int(lens.sum())
        # one C-level fill instead of len(block) array constructions
        flat = np.fromiter(chain.from_iterable(enc), dtype=np.int64, count=total)
        return lens, flat


def build_groups(specs: list[str], max_length: int) -> list[TokGroup]:
    """Collapse the --tokenizers list into one group per underlying tokenizer file,
    so `X` and `pc:X` share a single tokenization pass."""
    wanted: dict[str, list[bool]] = {}
    for spec in specs:
        path, is_pc = (spec[3:], True) if spec.startswith("pc:") else (spec, False)
        slot = wanted.setdefault(path, [False, False])
        slot[1 if is_pc else 0] = True

    groups = []
    for path, (plain, pc) in wanted.items():
        try:
            groups.append(TokGroup(path, plain, pc, max_length))
        except Exception as e:  # noqa: BLE001 - a bad path shouldn't abort the sweep
            print(f"SKIP tokenizer {path}: {e}", flush=True)
    return groups


def _sequences(df: pd.DataFrame) -> list[str]:
    """All sequences in a split; PPI rows contribute both chains."""
    if "sequence_a" in df.columns:
        return [str(s) for s in df["sequence_a"]] + [str(s) for s in df["sequence_b"]]
    return [str(s) for s in df["sequence"]]


def scan_train(group: TokGroup, seqs: list[str], chunk: int):
    """One pass: residue/token totals, per-sequence token lengths, and the set of
    ids seen -- as a boolean vocabulary mask, so marking is a vectorised scatter."""
    seen = np.zeros(group.vocab, dtype=bool)
    seen_pc = np.zeros(group.pc_vocab, dtype=bool) if group.collapse is not None else None
    n_res = n_tok = 0
    parts = []
    for i in range(0, len(seqs), chunk):
        block = seqs[i : i + chunk]
        lens, flat = group._flat(block)
        parts.append(lens)
        n_tok += int(lens.sum())
        n_res += sum(len(s) for s in block)
        if flat.size:
            seen[flat] = True
            if seen_pc is not None:
                seen_pc[group.collapse[flat]] = True
    lengths = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return n_res, n_tok, lengths, seen, seen_pc


def scan_oov(group: TokGroup, seqs: list[str], seen, seen_pc, chunk: int):
    """Fraction of test token occurrences (and of test sequences) carrying an id
    never seen in training. Sampling is sound here: both are ratios."""
    tot = 0
    out = {"tok": [0, 0], "seq": [0, 0]}  # [plain, pc] counters
    n_seq = 0
    for i in range(0, len(seqs), chunk):
        block = seqs[i : i + chunk]
        lens, flat = group._flat(block)
        if not flat.size:
            continue
        tot += int(lens.sum())
        nz = lens[lens > 0]
        n_seq += int(nz.size)
        offsets = np.concatenate(([0], np.cumsum(nz)[:-1]))
        for j, (mask, ids) in enumerate(
            [(seen, flat)] + ([(seen_pc, group.collapse[flat])] if seen_pc is not None else [])
        ):
            bad = (~mask[ids]).astype(np.int64)
            out["tok"][j] += int(bad.sum())
            out["seq"][j] += int((np.add.reduceat(bad, offsets) > 0).sum())
    rate = lambda j: (out["tok"][j] / max(1, tot), out["seq"][j] / max(1, n_seq))  # noqa: E731
    return rate(0), (rate(1) if seen_pc is not None else (0.0, 0.0))


def _rows_for(group, task, split, n_res, n_tok, lengths, oov_plain, oov_pc,
              n_test, n_distinct_plain, n_distinct_pc, args):
    """Emit the plain and/or parent-collapsed row from one shared scan.

    Both rows describe the SAME segmentation (collapse only remaps ids), so k-bar,
    lengths and truncation are shared; only the vocabulary, the distinct-id count
    and the OOV rates differ.
    """
    budget = max(1, args.max_length - 2)
    over = lengths > budget
    # Residues retained is proportional: exact per-token residue spans would need
    # offset mappings, which the collapse path cannot provide.
    kept = np.minimum(lengths, budget) / np.maximum(lengths, 1)
    shared = {
        "task": task,
        "split": split or "default",
        "n_train_sequences": len(lengths),
        "n_train_residues": n_res,
        "n_train_tokens": n_tok,
        "kbar_residues_per_token": round(n_res / max(1, n_tok), 4),
        "mean_tokens_per_seq": round(float(lengths.mean()), 2) if lengths.size else 0.0,
        "p95_tokens_per_seq": int(np.percentile(lengths, 95)) if lengths.size else 0,
        "max_tokens_per_seq": int(lengths.max()) if lengths.size else 0,
        "frac_seqs_truncated": round(float(over.mean()), 6) if lengths.size else 0.0,
        "mean_frac_residues_kept": round(float(kept.mean()), 6) if lengths.size else 1.0,
        "n_test_sampled": n_test,
    }
    rows = []
    for label, vocab, distinct, spec, oov in (
        (group.plain_label, group.vocab, n_distinct_plain, group.spec, oov_plain),
        (group.pc_label, group.pc_vocab, n_distinct_pc, f"pc:{group.spec}", oov_pc),
    ):
        if not label:
            continue
        rows.append({
            **shared,
            "tokenizer": label,
            "tokenizer_spec": spec,
            "vocab_size": vocab,
            "n_distinct_tokens_train": distinct,
            "vocab_utilization": round(distinct / max(1, vocab), 6),
            # THE capacity number: rows that can actually receive gradient.
            "effective_embed_params": distinct * args.scratch_dim,
            "nominal_embed_params": vocab * args.scratch_dim,
            "test_oov_token_rate": round(oov[0], 6),
            "test_frac_seqs_with_oov": round(oov[1], 6),
        })
    return rows


def pc_variants(specs: list[str]) -> tuple[list[str], list[tuple[str, str]]]:
    """`pc:` siblings for every spec that can have one. Returns (added, skipped).

    Parent-collapse needs a PUMA genealogy: the non-'hf_' sibling JSON that maps each
    mutational child to its parent. AA has no multi-residue units at all, and BPE has
    no genealogy -- its sibling file, if present, carries no `parent` entries, so a
    "collapsed" BPE arm would be an identity remap and a duplicate of the plain arm.
    Both are skipped, with a reason, rather than silently dropped.
    """
    added, skipped = [], []
    for spec in specs:
        if spec.startswith("pc:"):
            continue  # already a PC arm
        path = Path(spec)
        if spec.lower() in ("aa", "esm2") or path.suffix != ".json":
            skipped.append((spec, "not a tokenizer JSON (no mutational units)"))
            continue
        if path.parent.name.lower() == "bpe":
            skipped.append((spec, "BPE has no mutational genealogy"))
            continue
        family = path.parent / (path.name[3:] if path.name.startswith("hf_") else path.name)
        if not family.is_file():
            skipped.append((spec, f"no family JSON alongside it ({family.name})"))
            continue
        added.append(f"pc:{spec}")
    return added, skipped


def cmd_tokens(args: argparse.Namespace) -> None:
    specs = [s.strip() for s in (args.tokenizers or "").split(",") if s.strip()]
    if args.tokenizer_dir:
        found = sorted(str(p) for p in Path(args.tokenizer_dir).rglob("hf_*.json"))
        print(f"--tokenizer-dir: found {len(found)} tokenizer JSONs under {args.tokenizer_dir}")
        specs += found
    if not specs:
        raise SystemExit("Pass --tokenizers and/or --tokenizer-dir.")

    if args.include_pc:
        # Applies to everything in --tokenizers AND --tokenizer-dir. A PC arm costs one
        # numpy gather on the base tokenizer's pass, not a second tokenization pass.
        pc, skipped = pc_variants(specs)
        pc = [s for s in pc if s not in specs]  # don't duplicate an explicit pc: entry
        specs += pc
        print(f"--include-pc: added {len(pc)} parent-collapsed variant(s) (no extra passes)")
        for spec, why in skipped:
            print(f"  no PC variant for {spec}: {why}")

    # Rust tokenizers parallelise internally; combined with our own threads that
    # oversubscribes the machine, so pick exactly one level of parallelism.
    if args.jobs > 1:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("Loading tokenizers ...", flush=True)
    groups = build_groups(specs, args.max_length)
    if not groups:
        # Without this the per-task loop finds nothing pending and reports
        # "all arms already present", which points at the wrong problem.
        raise SystemExit(
            f"None of the {len(specs)} tokenizer spec(s) could be loaded (see the SKIP "
            f"lines above). `tokens` needs `transformers` and `tokenizers` installed.")
    n_labels = sum(len(g.labels) for g in groups)
    pairs = resolve_task_splits(args.task, args.split_method)
    tasks: dict[str, list] = {}
    for task, split in pairs:
        tasks.setdefault(task, []).append(split)
    print(f"{len(groups)} tokenization pass(es) -> {n_labels} arms | "
          f"{len(tasks)} task(s), {len(pairs)} task/split(s) | --jobs {args.jobs}\n", flush=True)

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "token_stats.csv"
    rows: list[dict] = []
    done: set = set()
    if args.resume and path.is_file():
        prev = pd.read_csv(path)
        rows = prev.to_dict("records")
        done = set(zip(prev["task"], prev["split"].astype(str), prev["tokenizer"]))
        print(f"--resume: {len(rows)} existing rows in {path}\n")

    for task, splits in tasks.items():
        spec_obj = TASKS[task]
        # remote_homology / deeploc_2 share train+valid across splits; only the test
        # file differs, so the (expensive) train scan is done once.
        shared_train = shares_train_across_splits(task)
        loaded = {}
        for split in splits:
            try:
                train, _valid, test = load_splits(spec_obj, split_method=split)
            except Exception as e:  # noqa: BLE001
                print(f"SKIP {task}/{split}: {e}", flush=True)
                continue
            train = preprocess_sequences(train, task_type=spec_obj.task_type)
            test = preprocess_sequences(test, task_type=spec_obj.task_type)
            te_seqs = _sequences(test)
            # Subsample the OOV test set ONCE, here in the parent, for two reasons:
            #  * every tokenizer must be scored on the SAME subset, or the OOV rates
            #    aren't comparable across arms;
            #  * drawing it inside the worker made the result depend on thread
            #    scheduling, so --jobs changed the numbers.
            # The seed is derived from (task, split) so it is stable across runs,
            # across --jobs, and across --resume skipping earlier tasks.
            if args.test_sample and len(te_seqs) > args.test_sample:
                key = f"{task}/{split or 'default'}".encode()
                digest = hashlib.blake2b(key, digest_size=8).digest()
                sub_rng = np.random.default_rng(args.seed + int.from_bytes(digest, "big"))
                idx = np.sort(sub_rng.choice(len(te_seqs), args.test_sample, replace=False))
                te_seqs = [te_seqs[i] for i in idx]
            loaded[split] = (_sequences(train), te_seqs)
        if not loaded:
            continue
        if shared_train and len(loaded) > 1:
            print(f"[shared-train] {task}: scanning train once for {len(loaded)} splits")

        def work(group, task=task, loaded=loaded, shared_train=shared_train):
            out, train_scan = [], None
            for split, (tr_seqs, te_seqs) in loaded.items():
                if train_scan is None or not shared_train:
                    train_scan = scan_train(group, tr_seqs, args.chunk)
                n_res, n_tok, lengths, seen, seen_pc = train_scan
                if n_tok == 0:
                    continue
                # te_seqs was already subsampled deterministically in the parent.
                oov_p, oov_c = scan_oov(group, te_seqs, seen, seen_pc, args.chunk)
                out.append((split, n_res, n_tok, lengths, oov_p, oov_c, len(te_seqs),
                            int(seen.sum()),
                            int(seen_pc.sum()) if seen_pc is not None else 0))
            return out

        pending = [g for g in groups if not all(
            (task, str(s or "default"), lab) in done for s in loaded for lab in g.labels)]
        if not pending:
            print(f"  {task}: all arms already present (--resume), skipping")
            continue

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            for group, results in zip(pending, pool.map(work, pending)):
                for split, n_res, n_tok, lengths, oov_p, oov_c, n_test, d_plain, d_pc in results:
                    made = _rows_for(group, task, split, n_res, n_tok, lengths,
                                     oov_p, oov_c, n_test, d_plain, d_pc, args)
                    rows.extend(made)
                    for r in made:
                        print(f"  {r['tokenizer']:34} {task}/{r['split']:16} "
                              f"k-bar={r['kbar_residues_per_token']:5.2f}  "
                              f"distinct={r['n_distinct_tokens_train']:6d}  "
                              f"eff_params={r['effective_embed_params'] / 1e3:8.1f}k  "
                              f"trunc={r['frac_seqs_truncated']:.1%}  "
                              f"test_oov={r['test_oov_token_rate']:.2%}", flush=True)
        # checkpoint after every task so a long sweep survives an interruption
        pd.DataFrame(rows).to_csv(path, index=False)

    if not rows:
        raise SystemExit("No rows produced.")
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f"\nWrote {len(rows)} rows -> {path}")


# --------------------------------------------------------------------------- #
#  PAIRED TESTS
# --------------------------------------------------------------------------- #
def _greater_is_better(task: str) -> bool:
    spec = TASKS.get(task)
    return True if spec is None else spec.greater_is_better


def _drop_collapsed(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Filter runs by the `collapsed` flag, which has TWO independent meanings.

    `test_nan` -> the test metric was NaN (constant predictions): a failed run.
    `val_nan`  -> only the best VALIDATION metric was NaN. The test score is still
                  a real number; all it means is that best-val checkpoint selection
                  had nothing to go on. This is common and expected for the
                  random-init scratch encoder on Spearman tasks, and dropping those
                  runs deletes most of the scratch arm.

    Default `test` therefore removes only genuinely failed runs. (Note that
    load_experiments already drops rows with a missing test_score, so most
    test_nan rows are gone before this runs.)
    """
    if mode == "none" or "collapsed" not in df.columns:
        return df
    flag = df["collapsed"].astype(str).replace({"nan": "", "None": ""}).fillna("")
    bad_test = flag.str.contains("test_nan", na=False)
    val_only = flag.str.contains("val_nan", na=False) & ~bad_test

    if mode == "any":
        keep = ~(bad_test | val_only)
        print(f"--collapsed any: dropping {int((~keep).sum())} flagged run(s) "
              f"({int(bad_test.sum())} test_nan, {int(val_only.sum())} val_nan-only).")
    else:
        keep = ~bad_test
        print(f"Collapsed flags: {int(bad_test.sum())} test_nan (dropped), "
              f"{int(val_only.sum())} val_nan-only (KEPT — the test score is valid; "
              f"use --collapsed any to drop these too).")
    return df[keep]


def task_scores(args: argparse.Namespace) -> pd.DataFrame:
    """Task-balanced score per (model, tokenizer, task).

    Mirrors the paper's aggregation: average seeds within a split, then splits
    within a task, so multi-split tasks don't dominate. Lower-is-better metrics are
    sign-flipped so "higher is better" holds everywhere.
    """
    from plm_benchmark.results import load_experiments

    df = load_experiments()
    if df.empty:
        raise SystemExit("outputs/experiments.csv is empty or missing.")

    df = df[df["method"] == args.method].copy()
    if args.model:
        df = df[df["model"] == args.model]
    if df.empty:
        raise SystemExit(f"No rows for method={args.method!r} model={args.model!r}.")

    df = _drop_collapsed(df, args.collapsed)
    if df.empty:
        raise SystemExit("Every run was filtered out. Try --collapsed none.")

    df["test_score"] = pd.to_numeric(df["test_score"], errors="coerce")
    df = df.dropna(subset=["test_score"])
    excluded = {t.strip() for t in (args.exclude_tasks or "").split(",") if t.strip()}
    if excluded:
        df = df[~df["task"].isin(excluded)]
    df["score"] = [
        s if _greater_is_better(t) else -s for t, s in zip(df["task"], df["test_score"])
    ]

    per_split = df.groupby(["model", "tokenizer", "task", "split"], as_index=False).agg(
        score=("score", "mean"), n_seeds=("score", "size")
    )
    per_task = per_split.groupby(["model", "tokenizer", "task"], as_index=False).agg(
        score=("score", "mean"), n_splits=("score", "size"), n_seeds=("n_seeds", "sum")
    )
    if per_task.empty:
        raise SystemExit("No (model, tokenizer, task) cells survived filtering.")
    print(f"{len(per_task)} (model, tokenizer, task) cells | "
          f"{per_task['model'].nunique()} model(s) | {per_task['task'].nunique()} tasks | "
          f"{per_task['tokenizer'].nunique()} tokenizers")
    return per_task


def _bootstrap_ci(diffs: np.ndarray, n_boot: int, seed: int, alpha: float = 0.05):
    """Percentile CI on the mean paired difference. With ~13 tasks the normal
    approximation behind a t-interval is shaky, so resample instead."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def cmd_paired(args: argparse.Namespace) -> None:
    from scipy import stats

    per_task = task_scores(args)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    matrix = per_task.pivot_table(index=["model", "task"], columns="tokenizer", values="score")
    matrix.to_csv(ANALYSIS_DIR / "task_scores.csv")
    print(f"Per-task score matrix -> {ANALYSIS_DIR / 'task_scores.csv'}\n")

    rows, thin = [], []
    for model, sub in per_task.groupby("model"):
        wide = sub.pivot_table(index="task", columns="tokenizer", values="score")
        if args.baseline not in wide.columns:
            print(f"SKIP model={model}: baseline {args.baseline!r} not present. "
                  f"Available tokenizers:\n  " + "\n  ".join(sorted(map(str, wide.columns))))
            continue
        for tokenizer in wide.columns:
            if tokenizer == args.baseline:
                continue
            # Pair on tasks where BOTH arms have a score; anything else would be an
            # unpaired comparison wearing a paired label.
            both = wide[[args.baseline, tokenizer]].dropna()
            if len(both) < args.min_tasks:
                thin.append((str(model), str(tokenizer), len(both)))
                continue
            base = both[args.baseline].to_numpy(float)
            other = both[tokenizer].to_numpy(float)
            diffs = other - base

            t_stat, t_p = stats.ttest_rel(other, base)
            try:
                w_stat, w_p = stats.wilcoxon(other, base)
            except ValueError:  # all differences zero
                w_stat, w_p = np.nan, np.nan
            lo, hi = _bootstrap_ci(diffs, args.n_boot, args.seed)

            rows.append(
                {
                    "model": model,
                    "method": args.method,
                    "tokenizer": tokenizer,
                    "baseline": args.baseline,
                    "n_tasks": len(both),
                    "mean_baseline": round(base.mean(), 6),
                    "mean_tokenizer": round(other.mean(), 6),
                    "mean_diff": round(float(diffs.mean()), 6),
                    "ci_lo": round(lo, 6),
                    "ci_hi": round(hi, 6),
                    "median_diff": round(float(np.median(diffs)), 6),
                    "wins": int((diffs > 0).sum()),
                    "losses": int((diffs < 0).sum()),
                    "t_stat": round(float(t_stat), 4),
                    "t_p": float(t_p),
                    "wilcoxon_stat": float(w_stat),
                    "wilcoxon_p": float(w_p),
                    # CI excluding zero is the readable version of "significant"
                    "ci_excludes_zero": bool(lo > 0 or hi < 0),
                }
            )

    if thin:
        print(f"Skipped {len(thin)} tokenizer(s) sharing fewer than --min-tasks "
              f"({args.min_tasks}) tasks with {args.baseline!r}: "
              + ", ".join(f"{t}({n})" for _m, t, n in thin[:12])
              + (" ..." if len(thin) > 12 else "") + "\n")
    if not rows:
        raise SystemExit(
            f"No comparable pairs. Every tokenizer shared fewer than {args.min_tasks} "
            f"tasks with {args.baseline!r} (see the list above). Check that the baseline "
            f"label is right and that it was run on the same tasks as the others.")
    out = pd.DataFrame(rows).sort_values(["model", "mean_diff"], ascending=[True, False])
    path = ANALYSIS_DIR / "paired_tests.csv"
    out.to_csv(path, index=False)

    show = ["model", "tokenizer", "n_tasks", "mean_diff", "ci_lo", "ci_hi",
            "wins", "losses", "t_p", "wilcoxon_p"]
    print(out[show].to_string(index=False))
    print(f"\nWrote {path}")
    print(f"\nPaired over tasks vs {args.baseline!r}. `mean_diff` > 0 means the "
          f"tokenizer beats the baseline; treat a CI spanning 0 as no difference.")


# --------------------------------------------------------------------------- #
#  CAPACITY JOIN
# --------------------------------------------------------------------------- #
def cmd_capacity(args: argparse.Namespace) -> None:
    from scipy import stats

    stats_path = ANALYSIS_DIR / "token_stats.csv"
    if not stats_path.is_file():
        raise SystemExit(f"{stats_path} not found — run `tier1_analysis.py tokens` first.")
    tok_stats = pd.read_csv(stats_path)
    per_task = task_scores(args)

    # token_stats is per (task, split); scores are task-balanced, so collapse the
    # token stats the same way (unweighted mean over splits).
    agg = (
        tok_stats.groupby(["task", "tokenizer"], as_index=False)
        .agg(
            vocab_size=("vocab_size", "first"),
            kbar=("kbar_residues_per_token", "mean"),
            n_distinct_tokens_train=("n_distinct_tokens_train", "mean"),
            effective_embed_params=("effective_embed_params", "mean"),
            nominal_embed_params=("nominal_embed_params", "first"),
            frac_seqs_truncated=("frac_seqs_truncated", "mean"),
            mean_frac_residues_kept=("mean_frac_residues_kept", "mean"),
            test_oov_token_rate=("test_oov_token_rate", "mean"),
        )
    )

    joined = per_task.merge(agg, on=["task", "tokenizer"], how="inner")
    if joined.empty:
        # Report both sides rather than just failing: the usual causes are that
        # `tokens` was run for fewer tokenizers than the benchmark has, or for a
        # different task group.
        st, se = set(agg["tokenizer"]), set(per_task["tokenizer"])
        kt, ke = set(agg["task"]), set(per_task["task"])
        raise SystemExit(
            "No overlap between token_stats.csv and experiments.csv on (task, tokenizer).\n"
            f"  token_stats tokenizers ({len(st)}): {', '.join(sorted(map(str, st))[:10])}\n"
            f"  experiments tokenizers ({len(se)}): {', '.join(sorted(map(str, se))[:10])}\n"
            f"  tasks only in token_stats: {', '.join(sorted(kt - ke)[:8]) or '(none)'}\n"
            f"  tasks only in experiments: {', '.join(sorted(ke - kt)[:8]) or '(none)'}\n"
            "Re-run `tokens` with every tokenizer the benchmark used (--tokenizer-dir "
            "--include-pc) and the same --task group."
        )
    missing = sorted(set(per_task["tokenizer"]) - set(agg["tokenizer"]))
    if missing:
        print(f"NOTE: {len(missing)} tokenizer(s) have scores but no token_stats row and "
              f"are excluded: {', '.join(map(str, missing[:10]))}"
              + (" ..." if len(missing) > 10 else "") + "\n")
    # Non-embedding parameters are constant across arms, so total size is just an
    # offset; both columns are provided so the plot can use either.
    joined["total_params_nominal"] = joined["nominal_embed_params"] + args.non_embed_params
    joined["total_params_effective"] = joined["effective_embed_params"] + args.non_embed_params
    joined["is_baseline"] = joined["tokenizer"] == args.baseline

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    path = ANALYSIS_DIR / "capacity.csv"
    joined.to_csv(path, index=False)

    print(f"Joined {len(joined)} (task, tokenizer) rows -> {path}\n")

    # Correlate WITHIN each task, then summarise across tasks. Pooling all rows
    # would confound capacity with task difficulty (an easy task scores high at any
    # capacity), which is the classic way to manufacture a spurious correlation.
    print("Score vs log10(effective embedding params), Spearman computed WITHIN each "
          "task, over the sub-word arms only:")
    rhos = []
    for task, sub in joined[~joined["is_baseline"]].groupby("task"):
        if sub["effective_embed_params"].nunique() >= 4:
            rho, _ = stats.spearmanr(np.log10(sub["effective_embed_params"] + 1), sub["score"])
            if np.isfinite(rho):
                rhos.append((task, rho))
    if rhos:
        vals = np.array([r for _, r in rhos])
        print(f"  {len(rhos)} tasks | mean rho={vals.mean():+.3f}  median={np.median(vals):+.3f}  "
              f"positive in {(vals > 0).sum()}/{len(vals)}")
        print("  " + "  ".join(f"{t}:{r:+.2f}" for t, r in rhos))
        print("  -> rho near 0 means capacity does NOT drive the score within the "
              "sub-word family, so it is unlikely to explain the AA gap either.")
    else:
        print("  (need >=4 distinct capacities per task; add more tokenizers)")

    # The actual test: at matched EFFECTIVE capacity, does the baseline sit on the
    # same curve as the sub-word arms, or below it?
    base = joined[joined["is_baseline"]]
    if not base.empty:
        print(f"\nPer-task: {args.baseline} vs the sub-word arm closest to it in "
              f"effective capacity")
        print(f"{'model':20} {'task':24} {'eff_params(base)':>17} {'score(base)':>12} "
              f"{'closest arm':>28} {'eff_params':>11} {'score':>8} {'delta':>8}")
        for _, brow in base.iterrows():
            others = joined[(joined["task"] == brow["task"])
                            & (joined["model"] == brow["model"])   # never mix models
                            & (~joined["is_baseline"])]
            if others.empty:
                continue
            i = (others["effective_embed_params"] - brow["effective_embed_params"]).abs().idxmin()
            o = others.loc[i]
            print(f"{str(brow['model'])[:20]:20} {brow['task']:24} "
                  f"{brow['effective_embed_params']:17.0f} {brow['score']:12.4f} "
                  f"{o['tokenizer']:>28} {o['effective_embed_params']:11.0f} "
                  f"{o['score']:8.4f} {o['score'] - brow['score']:+8.4f}")

    print("\nRead it as: if the baseline lies ON the sub-word capacity curve, the gain "
          "is parameters. If it lies BELOW arms of comparable effective capacity, the "
          "segmentation is carrying information parameters alone don't explain.")


# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("tokens", help="k-bar, effective vocabulary, truncation, test OOV")
    t.add_argument("--task", default="everything", help="Task, list, or group.")
    t.add_argument("--split-method", default=None, help="One split, 'all', or omit.")
    t.add_argument("--tokenizers", default=None,
                   help="Comma-separated: 'aa', a tokenizer.json, a dir, a hub id, or "
                        "'pc:<puma.json>' for parent-collapse.")
    t.add_argument("--tokenizer-dir", default=None,
                   help="Also scan this tree for hf_*.json tokenizers.")
    t.add_argument("--include-pc", action="store_true",
                   help="Also add the parent-collapsed (pc:) variant of every PUMA "
                        "tokenizer, from --tokenizers and --tokenizer-dir alike. AA and "
                        "BPE are skipped (no genealogy) with a printed reason.")
    t.add_argument("--max-length", type=int, default=1024,
                   help="Benchmark max_length, for the truncation columns.")
    t.add_argument("--scratch-dim", type=int, default=320,
                   help="Hidden size, for effective_embed_params = distinct x dim.")
    t.add_argument("--chunk", type=int, default=256, help="Sequences per encode call.")
    t.add_argument("--jobs", type=int, default=1,
                   help="Tokenizers scanned concurrently. The Rust encoder releases the "
                        "GIL, so threads scale; >1 disables the tokenizers' own internal "
                        "parallelism to avoid oversubscribing the machine. Set to about "
                        "the core count.")
    t.add_argument("--test-sample", type=int, default=20000,
                   help="Cap test sequences used for the OOV rates (0 = all). Both are "
                        "ratios, so sampling is unbiased; the count used is recorded in "
                        "n_test_sampled. Train is ALWAYS scanned in full — "
                        "n_distinct_tokens_train is a coverage statistic and sampling "
                        "would bias it downward.")
    t.add_argument("--seed", type=int, default=0,
                   help="Seed for --test-sample. The per-split seed is derived from "
                        "this plus a hash of (task, split), so the sample is identical "
                        "across runs, across --jobs, and across --resume.")
    t.add_argument("--resume", action="store_true",
                   help="Reuse rows already in token_stats.csv and only compute what is "
                        "missing. The file is rewritten after every task, so an "
                        "interrupted sweep resumes cheaply.")
    t.set_defaults(func=cmd_tokens)

    for name, fn, helptext in (
        ("paired", cmd_paired, "Paired per-task tests vs a baseline tokenizer"),
        ("capacity", cmd_capacity, "Join scores with effective capacity"),
    ):
        s = sub.add_parser(name, help=helptext)
        s.add_argument("--method", default="full_ft")
        s.add_argument("--model", default=None,
                       help="Filter to one model, e.g. scratch_d320_l1_h8.")
        s.add_argument("--baseline", default="AA", help="Tokenizer label to compare against.")
        s.add_argument("--exclude-tasks", default=DEFAULT_EXCLUDE,
                       help="Comma-separated tasks to drop (default: eSol, an error "
                            "metric). Pass '' to keep everything.")
        s.add_argument("--collapsed", default="test", choices=["test", "any", "none"],
                       help="Which flagged runs to drop. 'test' (default) drops only "
                            "test_nan (a failed run) and KEEPS val_nan-only runs, whose "
                            "test score is valid. 'any' drops both — note this removes "
                            "most of the scratch arm. 'none' keeps everything.")
        if name == "paired":
            s.add_argument("--n-boot", type=int, default=10000)
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--min-tasks", type=int, default=3,
                           help="Minimum tasks shared with the baseline to run a test.")
        else:
            s.add_argument("--non-embed-params", type=int, default=2_650_000,
                           help="Constant non-embedding parameter count of the scratch "
                                "encoder (d320 l1 h8: pos-emb 1.31M + block 1.23M + "
                                "head ~0.1M).")
        s.set_defaults(func=fn)

    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
