#!/usr/bin/env python3
"""Run many downstream fine-tuning jobs concurrently on a single GPU.

Each (model, task, seed) is an independent process invoking the plm_benchmark
CLI. A thread pool keeps --jobs of them in flight at once; the GPU is shared
(time-sliced, or better via NVIDIA MPS — see the README). Results are written to
outputs/experiments.csv, which is append-locked so concurrent writers are safe.

`--task everything` runs every task x every curated split; use `--split-method all`
for all splits of a named task, or `--split-method X` for one.

Examples:
    # Sweep two PLMs over all PETA tasks and their splits
    python -m plm_benchmark.run_benchmark \
        --models esm2_8m,esm2_35m --task peta_all --seeds 42,43,44 \
        --method full_ft --gpu 1 --jobs 8 --eval-every 5 --wandb_project pulm-ft

    # Scratch baseline sweeping multiple tokenizers over everything
    python -m plm_benchmark.run_benchmark \
        --models scratch --tokenizer aa,/path/puma.json,/path/bpe.json \
        --task everything --scratch-layers 2 --gpu 1 --jobs 8

Per-job stdout/stderr goes to outputs/logs/<job>.log; the console shows a
one-line PASS/FAIL summary per job.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Repo root (holds the plm_benchmark package); child jobs run `-m plm_benchmark.cli` from here.
ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "outputs" / "logs"


def expand_task_splits(task_arg: str, split_method):
    """Expand a task selection into (task, split) pairs (see cli.resolve_task_splits)."""
    sys.path.insert(0, str(ROOT))
    from plm_benchmark.tasks import resolve_task_splits

    return resolve_task_splits(task_arg, split_method)


def _tok_label(spec):
    sys.path.insert(0, str(ROOT))
    from plm_benchmark.tokenizers import tokenizer_label

    return tokenizer_label(spec) if spec else "def"


def build_jobs(args) -> list[dict]:
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    tokenizers = [t.strip() for t in (args.tokenizer or "").split(",") if t.strip()] or [None]
    pairs = expand_task_splits(args.task, args.split_method)  # [(task, split)]

    jobs = []
    for model in models:
        toks = tokenizers if model == "scratch" else [None]  # tokenizer sweep only for scratch
        for tok in toks:
            for task, split in pairs:
                for seed in seeds:
                    jobs.append(
                        {"model": model, "tokenizer": tok, "task": task, "split": split, "seed": seed}
                    )
    return jobs


def job_cmd(job: dict, args) -> list[str]:
    cmd = [
        sys.executable, "-m", "plm_benchmark.cli", "train",
        "--task", job["task"],
        "--method", args.method,
        "--model", job["model"],
        "--seed", job["seed"],
        "--gpu", str(args.gpu),
        "--num-workers", str(args.num_workers),
        "--eval-every", str(args.eval_every),
        "--batch", str(args.batch),
        "--val-batch", str(args.val_batch),
        "--scratch-dim", str(args.scratch_dim),
        "--scratch-layers", str(args.scratch_layers),
        "--scratch-heads", str(args.scratch_heads),
    ]
    if job["tokenizer"]:
        cmd += ["--tokenizer", job["tokenizer"]]
    if args.parent_collapse:
        cmd += ["--parent-collapse"]
    if args.embed_cache:
        cmd += ["--embed-cache", "--embed-cache-max-gb", str(args.embed_cache_max_gb)]
        if args.embed_cache_dir:
            cmd += ["--embed-cache-dir", args.embed_cache_dir]
    if job["split"]:
        cmd += ["--split-method", job["split"]]
    if args.epochs:
        cmd += ["--epochs", str(args.epochs)]
    if args.wandb_project:
        cmd += ["--wandb_project", args.wandb_project]
        if args.wandb_group:
            cmd += ["--wandb_group", args.wandb_group]
    cmd += args.extra
    return cmd


def child_env() -> dict:
    """Keep per-process CPU threads small so K concurrent jobs don't thrash."""
    env = dict(os.environ)
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def run_one(job: dict, args, env: dict) -> tuple[dict, int, float]:
    parts = [job["model"], _tok_label(job["tokenizer"]), job["task"], job["split"] or "default", f"seed{job['seed']}"]
    tag = "__".join(parts)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{tag}.log"
    start = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(job_cmd(job, args), cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    return job, proc.returncode, time.time() - start


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", required=True, help="Comma-separated model names (or 'scratch').")
    p.add_argument("--task", required=True, help="Task name/list or group (all, peta_all, everything).")
    p.add_argument("--method", default="full_ft")
    p.add_argument("--seeds", default="42,43,44")
    p.add_argument("--gpu", type=int, default=1, help="GPU id (1 = first device).")
    p.add_argument("--jobs", type=int, default=4, help="Concurrent processes on the GPU.")
    p.add_argument("--num-workers", type=int, default=0, help="Per-job dataloader workers (keep low under high -j).")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--val-batch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument(
        "--tokenizer",
        default=None,
        help="For --models scratch: one or a COMMA-SEPARATED list of tokenizers to sweep "
        "(e.g. aa,/path/puma.json,/path/bpe.json). Ignored for non-scratch models.",
    )
    p.add_argument(
        "--parent-collapse",
        action="store_true",
        help="Scratch + PUMA: fold children onto mutational parents (reduced vocab).",
    )
    p.add_argument("--embed-cache", action="store_true", help="embed_head: cache frozen-encoder outputs.")
    p.add_argument("--embed-cache-dir", default=None)
    p.add_argument("--embed-cache-max-gb", type=float, default=50.0)
    p.add_argument("--scratch-dim", type=int, default=320, help="Scratch embedding/hidden dim.")
    p.add_argument(
        "--scratch-layers",
        type=int,
        default=0,
        help="Scratch Transformer blocks: 0 = bag-of-tokens floor; 1-2 = small model with context.",
    )
    p.add_argument("--scratch-heads", type=int, default=8, help="Attention heads per scratch block (must divide --scratch-dim).")
    p.add_argument("--split-method", default=None, help="One split, 'all', or omit (groups run all curated splits).")
    p.add_argument("--wandb_project", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Extra flags passed verbatim to the CLI.")
    args = p.parse_args()

    jobs = build_jobs(args)
    env = child_env()
    print(f"Launching {len(jobs)} jobs, {args.jobs} at a time on GPU {args.gpu}. Logs: {LOG_DIR}", flush=True)

    done = failed = 0
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run_one, job, args, env) for job in jobs]
        for fut in as_completed(futures):
            job, code, secs = fut.result()
            done += 1
            status = "PASS" if code == 0 else f"FAIL({code})"
            if code != 0:
                failed += 1
            tok = _tok_label(job["tokenizer"])
            tag = f"{job['model']}/{tok}/{job['task']}/{job['split'] or 'default'}/seed{job['seed']}"
            print(f"[{done}/{len(jobs)}] {status:9} {tag:62} {secs:6.0f}s", flush=True)

    print(f"\nFinished: {len(jobs) - failed} ok, {failed} failed. See {LOG_DIR} for per-job logs.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
