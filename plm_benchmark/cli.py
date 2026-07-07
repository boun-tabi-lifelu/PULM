from __future__ import annotations

import argparse
from datetime import datetime, timezone

from plm_benchmark.config import DEFAULT_MODEL, FINETUNE_SEEDS, OUTPUTS_DIR, get_models, resolve_model
from plm_benchmark.methods import WandbConfig, run_downstream
from plm_benchmark.results import append_experiment, build_comparison, load_experiments
from plm_benchmark.tasks import TASKS, load_splits, resolve_tasks


def _parse_seeds(args: argparse.Namespace) -> list[int]:
    if args.seeds:
        return [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    if args.seed is not None:
        return [args.seed]
    return list(FINETUNE_SEEDS)


def _wandb_cfg(args: argparse.Namespace) -> WandbConfig:
    return WandbConfig(
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        entity=args.wandb_entity,
        group=args.wandb_group,
        mode=args.wandb_mode,
        run_id=args.wandb_run_id,
        resume=args.wandb_resume,
    )


def cmd_train(args: argparse.Namespace) -> None:
    model_cfg = resolve_model(args.model, args.checkpoint, tokenizer=args.tokenizer)
    tasks = resolve_tasks(args.task)
    seeds = _parse_seeds(args)
    wandb_cfg = _wandb_cfg(args)

    for name in tasks:
        spec = TASKS[name]
        split = args.split_method or spec.default_split
        for seed in seeds:
            try:
                train, valid, test = load_splits(spec, split_method=args.split_method)
                run_downstream(
                    spec,
                    model_cfg,
                    method=args.method,
                    split=split,
                    gpu=args.gpu,
                    epochs=args.epochs,
                    batch=args.batch,
                    accum=args.accum,
                    lr=args.lr,
                    seed=seed,
                    fp16=not args.no_fp16,
                    max_length=args.max_length,
                    val_batch=args.val_batch,
                    eval_every=args.eval_every,
                    patience=args.patience,
                    tokenizer_spec=args.tokenizer,
                    scratch_dim=args.scratch_dim,
                    num_workers=args.num_workers,
                    wandb_cfg=wandb_cfg,
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    train_df=train,
                    valid_df=valid,
                    test_df=test,
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
                        "split": split or "default",
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
    print("Scratch baseline: --model scratch --tokenizer <esm2 | tokenizer.json | dir | hub-id>")
    print("Rost tasks:", ", ".join(t for t in TASKS if not t.startswith("peta_")))
    print("PETA tasks:", ", ".join(t for t in TASKS if t.startswith("peta_")))
    print("Task groups: all (Rost only), peta_all, everything")
    print("Methods: full_ft, full_ft_peta20, lora, embed_head (frozen encoder) — one pipeline for all tasks")
    if OUTPUTS_DIR.exists():
        exp = load_experiments()
        if not exp.empty:
            print("\nLatest experiments:")
            cols = [c for c in ["task", "model", "method", "split", "seed", "test_score"] if c in exp.columns]
            print(exp[cols].tail(20).to_string(index=False))


def cmd_list_tasks(_: argparse.Namespace) -> None:
    from plm_benchmark.peta_data import PETA_DEFAULT_SPLIT, PETA_SPLIT_OPTIONS

    print("=== Rost (Schmirler / Nat Commun 2024) ===")
    for name, spec in sorted((k, v) for k, v in TASKS.items() if v.data_source == "rost"):
        print(f"  {name:10} {spec.task_type:14} metric={spec.metric} epochs={spec.finetune_epochs}")

    print("\n=== PETA (ProteinPretraining) ===")
    for name, spec in sorted((k, v) for k, v in TASKS.items() if v.data_source == "peta"):
        split = ""
        if spec.peta_key and spec.peta_key in PETA_SPLIT_OPTIONS:
            default = spec.default_split or PETA_DEFAULT_SPLIT.get(spec.peta_key, "")
            split = f" splits={PETA_SPLIT_OPTIONS[spec.peta_key]} default={default}"
        status = " [PPI: attention1d pool + sum]" if spec.task_type == "ppi" else ""
        print(
            f"  {name:22} {spec.task_type:14} labels={spec.num_labels:4} "
            f"metric={spec.metric} epochs={spec.finetune_epochs}{split}{status}"
        )

    print("\nAll tasks share one PyTorch pipeline: attention1d pooling + linear head.")
    print("  full_ft -> train encoder; embed_head -> freeze encoder (train head only); lora -> adapters.")
    print("  Rost recipe: lr=2e-5, no early stopping. PETA recipe: lr=1e-3, wd=0.001, patience=20.")
    print("\nData: Rost -> training data/   PETA -> training data/PETA/ft_datasets/  (see docs/PETA.md)")


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

    t = sub.add_parser("train", help="Run any task through the unified pipeline")
    t.add_argument("--task", required=True, help="GB1, peta_gb1, all, peta_all, everything, ...")
    t.add_argument(
        "--method",
        required=True,
        choices=["full_ft", "full_ft_peta20", "lora", "embed_head"],
        help="full_ft: train encoder; embed_head: freeze encoder; lora: adapters; "
        "full_ft_peta20: full_ft capped at 20 epochs",
    )
    t.add_argument("--patience", type=int, default=None, help="Early-stopping patience (0 disables; overrides task recipe)")
    t.add_argument("--split-method", default=None, help="PETA split (e.g. one_vs_rest for peta_gb1)")
    t.add_argument("--model", default=DEFAULT_MODEL, help=f"Registry name or 'scratch' (default: {DEFAULT_MODEL})")
    t.add_argument("--checkpoint", default=None, help="Local checkpoint dir override")
    t.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer for --model scratch (required): 'esm2', a tokenizer.json path, "
        "a saved-tokenizer dir, or a hub id. Optional override for other models.",
    )
    t.add_argument("--gpu", type=int, default=1, help="GPU id (1 = first device)")
    t.add_argument("--epochs", type=int, default=None)
    t.add_argument("--batch", type=int, default=64)
    t.add_argument("--val-batch", type=int, default=64)
    t.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate + checkpoint every N epochs (1=every epoch). >1 speeds up long runs; "
        "early-stopping patience then counts eval events, not epochs.",
    )
    t.add_argument("--accum", type=int, default=1)
    t.add_argument("--lr", type=float, default=None)
    t.add_argument("--seed", type=int, default=None, help="Single seed (overrides --seeds)")
    t.add_argument(
        "--seeds",
        default=None,
        help=f"Comma-separated seeds (default: {','.join(map(str, FINETUNE_SEEDS))})",
    )
    t.add_argument("--no-fp16", action="store_true")
    t.add_argument("--max-length", type=int, default=1024)
    t.add_argument("--scratch-dim", type=int, default=320, help="Embedding dim for scratch_ baselines")
    t.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    t.add_argument("--resume-from-checkpoint", default=None, help="Path to a Trainer checkpoint to resume")

    # wandb
    t.add_argument("--wandb_project", default=None, help="Enables wandb when set.")
    t.add_argument("--wandb_run_name", default=None)
    t.add_argument("--wandb_entity", default=None)
    t.add_argument("--wandb_group", default=None, help="Group runs (e.g. by model size).")
    t.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    t.add_argument(
        "--wandb_run_id",
        default=None,
        help="Continue THIS existing wandb run id (use with --resume-from-checkpoint).",
    )
    t.add_argument(
        "--wandb_resume",
        default="allow",
        choices=["allow", "must", "never", "auto"],
        help="wandb resume mode when --wandb_run_id is set.",
    )
    t.set_defaults(func=cmd_train)

    c = sub.add_parser("compare", help="Build comparison table from experiments.csv")
    c.set_defaults(func=cmd_compare)

    l = sub.add_parser("list", help="List models, tasks, and recent runs")
    l.set_defaults(func=cmd_list)

    lt = sub.add_parser("list-tasks", help="List Rost and PETA tasks with metrics")
    lt.set_defaults(func=cmd_list_tasks)

    m = sub.add_parser("list-models", help="List all hub + PULM checkpoints")
    m.set_defaults(func=cmd_list_models)

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
