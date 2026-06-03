from __future__ import annotations

import argparse
from datetime import datetime, timezone

from plm_benchmark.config import DEFAULT_MODEL, MODELS, OUTPUTS_DIR
from plm_benchmark.methods import run_embed_head, run_finetune
from plm_benchmark.results import append_experiment, build_comparison, load_experiments
from plm_benchmark.tasks import TASKS, load_splits, resolve_tasks


def cmd_train(args: argparse.Namespace) -> None:
    if args.method not in ("full_ft", "lora", "embed_head"):
        raise SystemExit("method must be one of: full_ft, lora, embed_head")

    model_cfg = MODELS[args.model]
    tasks = resolve_tasks(args.task)

    for name in tasks:
        spec = TASKS[name]
        try:
            train, valid, test = load_splits(spec)
            if args.method == "embed_head":
                epochs = args.epochs or spec.embed_head_epochs
                run_embed_head(
                    spec,
                    model_cfg.checkpoint,
                    model_cfg.name,
                    train,
                    valid,
                    test,
                    gpu=args.gpu,
                    epochs=epochs,
                    batch=args.batch,
                    seed=args.seed,
                    reembed=args.reembed,
                    max_length=args.max_length,
                    embed_batch=args.embed_batch,
                )
            else:
                epochs = args.epochs or spec.finetune_epochs
                run_finetune(
                    spec,
                    model_cfg.checkpoint,
                    model_cfg.name,
                    train,
                    valid,
                    test,
                    method=args.method,
                    gpu=args.gpu,
                    epochs=epochs,
                    batch=args.batch,
                    accum=args.accum,
                    lr=args.lr,
                    seed=args.seed,
                    fp16=not args.no_fp16,
                )
        except Exception as e:
            print(f"FAILED {name}: {e}")
            append_experiment(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "task": name,
                    "model": model_cfg.name,
                    "checkpoint": model_cfg.checkpoint,
                    "method": f"{args.method}_error",
                    "metric": "",
                    "test_score": "",
                    "val_score": "",
                    "error": str(e),
                }
            )


def cmd_compare(_: argparse.Namespace) -> None:
    df = build_comparison()
    print(df.to_string(index=False))


def cmd_list(_: argparse.Namespace) -> None:
    print("Models:", ", ".join(MODELS))
    print("Tasks:", ", ".join(TASKS))
    print("Methods: full_ft, lora, embed_head")
    if OUTPUTS_DIR.exists():
        exp = load_experiments()
        if not exp.empty:
            print("\nLatest experiments:")
            print(exp[["task", "model", "method", "test_score"]].to_string(index=False))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Run full_ft, lora, or embed_head on task(s)")
    t.add_argument("--task", required=True, help="GB1, AAV, ... or 'all'")
    t.add_argument("--method", required=True, choices=["full_ft", "lora", "embed_head"])
    t.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODELS))
    t.add_argument("--gpu", type=int, default=2, help="GPU id (1 = first device)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--no-fp16", action="store_true")
    t.add_argument("--reembed", action="store_true", help="Force recompute embeddings (embed_head)")
    t.add_argument("--max-length", type=int, default=1024)
    t.add_argument("--embed-batch", type=int, default=4)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="Build comparison table from experiments.csv")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("list", help="List models, tasks, and recent runs")
    l.set_defaults(func=cmd_list)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
