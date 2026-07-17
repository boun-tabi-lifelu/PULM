from __future__ import annotations

import argparse
from datetime import datetime, timezone

from plm_benchmark.config import (
    DEFAULT_MODEL,
    EARLY_STOPPING_PATIENCE,
    FINETUNE_SEEDS,
    LR_FULL_FT,
    LR_HEAD,
    LR_LORA,
    MAX_EPOCHS,
    OUTPUTS_DIR,
    WEIGHT_DECAY,
    get_models,
    resolve_model,
)
from plm_benchmark.methods import WandbConfig, run_downstream
from plm_benchmark.results import append_experiment, build_comparison, load_experiments
from plm_benchmark.tasks import TASKS, load_splits, resolve_task_splits, shares_train_across_splits


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


def _group_runs(pairs: list[tuple[str, str | None]]) -> list[tuple[str, list[str | None]]]:
    """Collapse splits that share train+valid into ONE training run with several test
    sets (remote_homology, deeploc_2): 3x / 2x cheaper and it is the same model anyway."""
    out: list[tuple[str, list[str | None]]] = []
    shared: dict[str, list[str | None]] = {}
    for task, split in pairs:
        if shares_train_across_splits(task):
            if task not in shared:
                shared[task] = []
                out.append((task, shared[task]))  # filled by the appends below
            shared[task].append(split)
        else:
            out.append((task, [split]))
    return out


def cmd_train(args: argparse.Namespace) -> None:
    model_cfg = resolve_model(
        args.model, args.checkpoint, tokenizer=args.tokenizer, parent_collapse=args.parent_collapse
    )
    if args.parent_collapse and model_cfg.backend != "scratch":
        raise SystemExit(
            "--parent-collapse only applies to --model scratch. Pretrained parent-collapsed "
            "(_PC) checkpoints are detected and handled automatically."
        )
    task_splits = resolve_task_splits(args.task, args.split_method)
    seeds = _parse_seeds(args)
    wandb_cfg = _wandb_cfg(args)

    for name, splits in _group_runs(task_splits):
        spec = TASKS[name]
        split = splits[0]
        for seed in seeds:
            try:
                # Shared-train tasks: train/valid come from any split; only test differs.
                train, valid, first_test = load_splits(spec, split_method=split)
                test_sets = {split: first_test}
                for extra in splits[1:]:
                    test_sets[extra] = load_splits(spec, split_method=extra)[2]
                if len(test_sets) > 1:
                    print(
                        f"[shared-train] {name}: one training run scoring {len(test_sets)} test "
                        f"sets {list(test_sets)}",
                        flush=True,
                    )
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
                    parent_collapse=args.parent_collapse,
                    scratch_dim=args.scratch_dim,
                    scratch_layers=args.scratch_layers,
                    scratch_heads=args.scratch_heads,
                    embed_cache=args.embed_cache,
                    embed_cache_dir=args.embed_cache_dir,
                    embed_cache_max_gb=args.embed_cache_max_gb,
                    num_workers=args.num_workers,
                    wandb_cfg=wandb_cfg,
                    resume_from_checkpoint=args.resume_from_checkpoint,
                    train_df=train,
                    valid_df=valid,
                    test_sets=test_sets,
                )
            except Exception as e:
                print(f"FAILED {name} split={'+'.join(str(x or 'default') for x in splits)} seed={seed}: {e}")
                now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                append_experiment(
                    {
                        "start_datetime": now,
                        "end_datetime": now,
                        "task": name,
                        "model": model_cfg.name,
                        "full_name": model_cfg.name,
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
    print("Scratch baseline: --model scratch --tokenizer <aa | tokenizer.json | dir | hub-id>")
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
    from plm_benchmark.peta_data import PETA_RUN_SPLITS, PETA_SPLIT_OPTIONS

    print("=== Rost (Schmirler / Nat Commun 2024) ===")
    for name, spec in sorted((k, v) for k, v in TASKS.items() if v.data_source == "rost"):
        print(f"  {name:10} {spec.task_type:14} metric={spec.metric}")

    print("\n=== PETA (ProteinPretraining) ===  (runs=splits under everything/--split-method all)")
    for name, spec in sorted((k, v) for k, v in TASKS.items() if v.data_source == "peta"):
        split = ""
        if spec.peta_key and spec.peta_key in PETA_SPLIT_OPTIONS:
            runs = PETA_RUN_SPLITS.get(spec.peta_key, [spec.default_split])
            split = f" runs={runs} (all options: {PETA_SPLIT_OPTIONS[spec.peta_key]})"
        status = " [PPI: attention1d pool + sum]" if spec.task_type == "ppi" else ""
        print(
            f"  {name:22} {spec.task_type:14} labels={spec.num_labels:4} "
            f"metric={spec.metric}{split}{status}"
        )

    print("\nSplits: `everything`/`peta_all` run every task's curated splits; --split-method all")
    print("  runs all splits of a named task; --split-method X runs one; omit for the default only.")
    print("\nAll tasks share one PyTorch pipeline: attention1d pooling + linear head.")
    print("  full_ft -> train encoder; embed_head -> freeze encoder (train head only); lora -> adapters.")
    print(
        f"  Unified recipe (all tasks): max_epochs={MAX_EPOCHS}, patience={EARLY_STOPPING_PATIENCE}, "
        f"weight_decay={WEIGHT_DECAY}."
    )
    print(
        f"  LR by regime: full_ft={LR_FULL_FT}, embed_head/scratch={LR_HEAD}, lora={LR_LORA}."
    )
    print("\nData: Rost -> data/training data/   PETA -> data/ft_datasets/  (see plm_benchmark/README.md)")


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
    t.add_argument(
        "--patience",
        type=int,
        default=None,
        help=f"Early-stopping patience in eval events (0 disables; default {EARLY_STOPPING_PATIENCE})",
    )
    t.add_argument(
        "--split-method",
        default=None,
        help="PETA split for the named task(s): a specific split (e.g. one_vs_rest), or 'all' "
        "to run every curated split. Ignored for groups (all/peta_all/everything), which "
        "always run every curated split. Omit to run the default split only.",
    )
    t.add_argument("--model", default=DEFAULT_MODEL, help=f"Registry name or 'scratch' (default: {DEFAULT_MODEL})")
    t.add_argument("--checkpoint", default=None, help="Local checkpoint dir override")
    t.add_argument(
        "--tokenizer",
        default=None,
        help="Tokenizer for --model scratch (required): 'aa' (amino-acid/ESM-2 tokenizer), "
        "a tokenizer.json path, a saved-tokenizer dir, or a hub id. Optional override for other models.",
    )
    t.add_argument(
        "--parent-collapse",
        action="store_true",
        help="Scratch + PUMA only: fold children onto mutational parents (reduced vocab), "
        "tokenizing with the full vocab and remapping. Needs the family JSON sibling of the "
        "tokenizer .json (same dir, 'hf_'-stripped name). Mirrors plm_train's collapse.",
    )
    t.add_argument("--gpu", type=int, default=1, help="GPU id (1 = first device)")
    t.add_argument("--epochs", type=int, default=None, help=f"Override max epochs (default {MAX_EPOCHS} for all tasks)")
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
    t.add_argument("--scratch-dim", type=int, default=320, help="Embedding/hidden dim for the scratch baseline")
    t.add_argument(
        "--scratch-layers",
        type=int,
        default=0,
        help="Transformer blocks in the scratch encoder. 0 = bag-of-tokens (embedding + "
        "LayerNorm + pooling; no token↔token context — a clean floor). 1-2 = a small "
        "non-pretrained model with context (fairer baseline; token interaction).",
    )
    t.add_argument(
        "--scratch-heads",
        type=int,
        default=8,
        help="Attention heads per scratch Transformer block (used when --scratch-layers>0; "
        "must divide --scratch-dim).",
    )
    t.add_argument(
        "--embed-cache",
        action="store_true",
        help="embed_head fast path: cache the frozen encoder's per-residue outputs once and "
        "train the attention1d head off the cache (numerically identical, ~epochs-fold faster).",
    )
    t.add_argument("--embed-cache-dir", default=None, help="Cache location (default: outputs/embeddings)")
    t.add_argument(
        "--embed-cache-max-gb",
        type=float,
        default=50.0,
        help="Skip the cache and encode on the fly if the estimate exceeds this.",
    )
    t.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    t.add_argument("--resume-from-checkpoint", default=None, help="Path to a Trainer checkpoint to resume")

    # wandb
    t.add_argument("--wandb_project", default=None, help="Enables wandb when set.")
    t.add_argument("--wandb_run_name", default=None)
    t.add_argument("--wandb_entity", default=None)
    t.add_argument("--wandb_group", default=None, help="Group runs (default: {model}/{tokenizer}).")
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
