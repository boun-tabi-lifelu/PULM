from __future__ import annotations

import argparse
from datetime import datetime, timezone

from plm_benchmark.config import DEFAULT_MODEL, FINETUNE_SEEDS, OUTPUTS_DIR, get_models, resolve_model
from plm_benchmark.methods import run_embed_head, run_finetune
from plm_benchmark.results import append_experiment, build_comparison, load_experiments
from plm_benchmark.tasks import TASKS, load_splits, resolve_tasks


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.seed is not None:
        return [args.seed]
    return list(FINETUNE_SEEDS)


def cmd_train(args: argparse.Namespace) -> None:
    if args.method not in ("full_ft", "lora", "embed_head"):
        raise SystemExit("method must be one of: full_ft, lora, embed_head")

    model_cfg = resolve_model(args.model, args.checkpoint)
    tasks = resolve_tasks(args.task)
    seeds = _parse_seeds(args)

    for name in tasks:
        spec = TASKS[name]
        for seed in seeds:
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
                        seed=seed,
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
                        seed=seed,
                        fp16=not args.no_fp16,
                    )
            except Exception as e:
                print(f"FAILED {name} seed={seed}: {e}")
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
                        "seed": seed,
                        "error": str(e),
                    }
                )


def cmd_compare(_: argparse.Namespace) -> None:
    df = build_comparison()
    print(df.to_string(index=False))


def cmd_list(_: argparse.Namespace) -> None:
    models = get_models()
    hub = [k for k, v in models.items() if v.source == "hub"]
    pulm = [k for k, v in models.items() if v.source == "pulm"]
    print("Hub models:", ", ".join(hub))
    print(f"PULM models: {len(pulm)} discovered (use list-models)")
    print("Tasks:", ", ".join(TASKS))
    print("Methods: full_ft, lora, embed_head")
    if OUTPUTS_DIR.exists():
        exp = load_experiments()
        if not exp.empty:
            print("\nLatest experiments:")
            cols = [c for c in ["task", "model", "method", "seed", "test_score"] if c in exp.columns]
            print(exp[cols].tail(20).to_string(index=False))


def cmd_list_models(_: argparse.Namespace) -> None:
    models = get_models()
    pulm = [(k, v) for k, v in sorted(models.items()) if v.source == "pulm"]
    hub = [(k, v) for k, v in sorted(models.items()) if v.source == "hub"]

    print("=== Hub (Meta ESM-2) ===")
    for name, cfg in hub:
        print(f"  {name}\n    {cfg.checkpoint}")

    print(f"\n=== PULM ({len(pulm)} with weights) ===")
    for name, cfg in pulm:
        print(f"  {name}\n    {cfg.checkpoint}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="Run full_ft, lora, or embed_head on task(s)")
    t.add_argument("--task", required=True, help="GB1, AAV, ... or 'all'")
    t.add_argument("--method", required=True, choices=["full_ft", "lora", "embed_head"])
    t.add_argument("--model", default=DEFAULT_MODEL, help=f"Registry name (default: {DEFAULT_MODEL})")
    t.add_argument("--checkpoint", default=None, help="Local checkpoint dir override")
    t.add_argument("--gpu", type=int, default=2, help="GPU id (1 = first device)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--seed", type=int, default=None, help="Single seed (overrides --seeds)")
    t.add_argument(
        "--seeds",
        default=None,
        help=f"Comma-separated seeds (default: {','.join(map(str, FINETUNE_SEEDS))})",
    )
    t.add_argument("--no-fp16", action="store_true")
    t.add_argument("--reembed", action="store_true", help="Force recompute embeddings (embed_head)")
    t.add_argument("--max-length", type=int, default=1024)
    t.add_argument("--embed-batch", type=int, default=4)
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="Build comparison table from experiments.csv")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("list", help="List models, tasks, and recent runs")
    l.set_defaults(func=cmd_list)

    m = sub.add_parser("list-models", help="List all hub + PULM checkpoints")
    m.set_defaults(func=cmd_list_models)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
