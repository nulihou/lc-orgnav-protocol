#!/usr/bin/env python3
"""Train and evaluate LC-ORGNav architecture variants without CARLA.

The main use is the No-FiLM ablation requested during review. For that
variant, FiLM linear layers are fixed to emit identity modulation:

  gamma_node=1, beta_node=0, gamma_edge=1, beta_edge=0

The scoring head still receives the query embedding, so this removes FiLM
conditioning in message passing without turning the model into a no-query
baseline. The `full` variant trains the same architecture without this
intervention and is useful when a matched full-vs-ablation comparison is needed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.config import TrainConfig
from lc_org_nav.trainer_mixed import MixedBatchTrainer
try:
    from scripts.analysis.eval_lcorg_protocol_no_carla import evaluate as evaluate_protocol
except ModuleNotFoundError:
    from scripts.revision.eval_lcorg_protocol_no_carla import evaluate as evaluate_protocol


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_limited_list(source: Path, dest: Path, limit: int, seed: int) -> Path:
    with source.open("r", encoding="utf-8-sig") as f:
        items = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
    if limit and limit > 0 and limit < len(items):
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:limit]
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text("\n".join(items) + "\n", encoding="utf-8")
    return dest


def apply_no_film_identity(model: torch.nn.Module) -> None:
    """Freeze FiLM layers as an identity transform."""
    hidden_dim = int(getattr(model, "hidden_dim"))
    for layer in getattr(model, "film_layers"):
        with torch.no_grad():
            layer.weight.zero_()
            layer.bias.zero_()
            layer.bias[:hidden_dim].fill_(1.0)
            layer.bias[2 * hidden_dim : 3 * hidden_dim].fill_(1.0)
        for param in layer.parameters():
            param.requires_grad = False


def find_best_checkpoint(save_dir: Path, model_name: str) -> Path:
    preferred = [
        save_dir / "best_lc_org_mixed_batch.pt",
        save_dir / "best_lc_org_mtl_mixed.pt",
        save_dir / "best_lc_org_mtl.pt",
        save_dir / "best_lc_org_nomask_mixed.pt",
        save_dir / "best_lc_org_nomask.pt",
    ]
    for path in preferred:
        if path.exists():
            return path
    epoch_ckpts = sorted(list(save_dir.glob(f"epoch_*_{model_name}")) + list(save_dir.glob("epoch_*_model.pt")))
    if epoch_ckpts:
        return epoch_ckpts[-1]
    raise FileNotFoundError(f"No checkpoint found in {save_dir}")


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def format_acc(correct: int, total: int) -> str:
    acc = 100.0 * correct / total if total else 0.0
    lo, hi = wilson_ci(correct, total)
    return f"{correct}/{total} ({acc:.1f}%, 95% CI {lo * 100:.1f}-{hi * 100:.1f})"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["full", "no_film"], default="no_film")
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_work/experiments_20260520/core_ablations/no_film"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--lambda-rel", type=float, default=0.5)
    parser.add_argument("--lambda-geom", type=float, default=0.1)
    parser.add_argument("--lambda-rank", type=float, default=0.1)
    parser.add_argument("--rank-margin", type=float, default=1.0)
    parser.add_argument("--use-rel-gate", action="store_true")
    parser.add_argument("--use-edge-context-head", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--max-scen-id", type=int, default=29)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_list = args.train_list
    val_list = args.val_list
    if args.limit_train > 0:
        train_list = write_limited_list(
            args.train_list,
            args.output_dir / f"train_limit{args.limit_train}_seed{args.seed}.txt",
            args.limit_train,
            args.seed,
        )
    if args.limit_val > 0:
        val_list = write_limited_list(
            args.val_list,
            args.output_dir / f"val_limit{args.limit_val}_seed{args.seed}.txt",
            args.limit_val,
            args.seed + 1,
        )

    save_dir = args.output_dir / "checkpoints"
    cfg = TrainConfig(
        train_list=str(train_list),
        val_list=str(val_list),
        save_dir=str(save_dir),
        model_name=f"lcorg_{args.variant}.pt",
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        lambda_rel=args.lambda_rel,
        lambda_geom=args.lambda_geom,
        lambda_rank=args.lambda_rank,
        rank_margin=args.rank_margin,
        use_relation_mask=False,
        log_interval=50,
        val_interval=1,
        use_rel_gate=bool(args.use_rel_gate),
        use_edge_context_head=bool(args.use_edge_context_head),
    )

    trainer = MixedBatchTrainer(cfg)
    if args.variant == "no_film":
        apply_no_film_identity(trainer.model)

    metadata = {
        "variant": args.variant,
        "seed": args.seed,
        "train_list": str(train_list),
        "val_list": str(val_list),
        "protocol_dir": str(args.protocol_dir),
        "config": asdict(cfg),
        "film_identity_fixed": args.variant == "no_film",
        "use_rel_gate": bool(args.use_rel_gate),
        "use_edge_context_head": bool(args.use_edge_context_head),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    trainer.train()

    best_ckpt = find_best_checkpoint(save_dir, cfg.model_name)
    stable_ckpt = args.output_dir / f"{args.variant}_best.pt"
    shutil.copy(best_ckpt, stable_ckpt)

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    protocol_jsonl = args.output_dir / f"{args.variant}_protocol.jsonl"
    protocol_summary = evaluate_protocol(
        stable_ckpt,
        args.protocol_dir,
        protocol_jsonl,
        torch.device(device_name),
        args.max_scen_id,
    )
    (args.output_dir / f"{args.variant}_protocol_summary.json").write_text(
        json.dumps(protocol_summary, indent=2),
        encoding="utf-8",
    )

    correct = int(protocol_summary["correct"])
    total = int(protocol_summary["total"])
    lines = [
        f"# LC-ORGNav core ablation: {args.variant}",
        "",
        f"- Train list: `{train_list}`",
        f"- Val list: `{val_list}`",
        f"- Epochs: {args.epochs}",
        f"- Batch size: {args.batch_size}",
        f"- Device: `{device_name}`",
        f"- Checkpoint: `{stable_ckpt}`",
        f"- Protocol JSONL: `{protocol_jsonl}`",
        "",
        "| Variant | ProtoAcc@1 | Notes |",
        "|---|---:|---|",
        (
            f"| {args.variant} | {format_acc(correct, total)} | "
            + (
                "FiLM fixed to identity; query still concatenated in goal head"
                if args.variant == "no_film"
                else (
                    "Matched full-FiLM architecture with neural relation gate and edge-context residual scorer"
                    if args.use_rel_gate and args.use_edge_context_head
                    else (
                        "Matched full-FiLM architecture with edge-context residual scorer"
                        if args.use_edge_context_head
                        else (
                            "Matched full-FiLM architecture with neural relation gate"
                            if args.use_rel_gate
                            else "Matched full-FiLM architecture under the same train/val protocol"
                        )
                    )
                )
            )
            + " |"
        ),
        "",
    ]
    table_path = args.output_dir / f"{args.variant}_table.md"
    table_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {table_path}", flush=True)


if __name__ == "__main__":
    main()
