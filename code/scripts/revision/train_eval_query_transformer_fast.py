#!/usr/bin/env python3
"""Batch-train the query-token Transformer baseline without CARLA."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.baselines.attention_pool import QueryTokenTransformerNet


class GoalSelectionDataset(Dataset):
    def __init__(self, list_file: str | Path):
        with Path(list_file).open("r", encoding="utf-8-sig") as f:
            self.paths = [line.strip() for line in f if line.strip() and not line.lstrip().startswith("#")]
        if not self.paths:
            raise RuntimeError(f"No samples found in {list_file}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        data = np.load(path, allow_pickle=False)

        def tensor_float(key: str) -> torch.Tensor:
            arr = data[key]
            if arr.ndim >= 1 and arr.shape[0] == 1 and key in {"node_feats", "lang_feat", "candidate_mask"}:
                arr = arr[0]
            return torch.from_numpy(arr.astype(np.float32))

        cand = data["candidate_mask"]
        if cand.ndim == 2 and cand.shape[0] == 1:
            cand = cand[0]
        return {
            "node_feats": tensor_float("node_feats"),
            "lang_feat": tensor_float("lang_feat"),
            "candidate_mask": torch.from_numpy(cand.astype(np.bool_)),
            "target_index": int(data["target_index"]),
            "path": path,
        }


def collate_padded(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    batch_size = len(batch)
    max_nodes = max(int(item["node_feats"].shape[0]) for item in batch)
    node_dim = int(batch[0]["node_feats"].shape[1])
    lang_dim = int(batch[0]["lang_feat"].shape[0])
    node_feats = torch.zeros(batch_size, max_nodes, node_dim, dtype=torch.float32)
    lang_feat = torch.zeros(batch_size, lang_dim, dtype=torch.float32)
    candidate_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    node_valid_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    targets = torch.zeros(batch_size, dtype=torch.long)
    paths: List[str] = []
    for i, item in enumerate(batch):
        n = int(item["node_feats"].shape[0])
        node_feats[i, :n] = item["node_feats"]
        lang_feat[i] = item["lang_feat"]
        candidate_mask[i, :n] = item["candidate_mask"].to(torch.bool)
        node_valid_mask[i, :n] = True
        targets[i] = int(item["target_index"])
        paths.append(str(item["path"]))
    return {
        "node_feats": node_feats,
        "lang_feat": lang_feat,
        "candidate_mask": candidate_mask,
        "node_valid_mask": node_valid_mask,
        "target_index": targets,
        "path": paths,
    }


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return moved


def make_model(sample: Dict[str, Any], args: argparse.Namespace) -> QueryTokenTransformerNet:
    return QueryTokenTransformerNet(
        node_feat_dim=int(sample["node_feats"].shape[-1]),
        edge_feat_dim=0,
        lang_feat_dim=int(sample["lang_feat"].shape[-1]),
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_rel_types=5,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )


def masked_ce_loss(logits: torch.Tensor, candidate_mask: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    masked_logits = logits.masked_fill(~candidate_mask.to(torch.bool), -1e9)
    return F.cross_entropy(masked_logits, targets)


def predict_batch(logits: torch.Tensor, candidate_mask: torch.Tensor, targets: torch.Tensor) -> Tuple[int, int, List[int]]:
    masked_logits = logits.masked_fill(~candidate_mask.to(torch.bool), -1e9)
    pred = torch.argmax(masked_logits, dim=1)
    correct = int((pred == targets).sum().item())
    total = int(targets.numel())
    return correct, total, [int(x) for x in pred.detach().cpu().tolist()]


def evaluate_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            out = model(
                node_feats=batch["node_feats"],
                edge_index=torch.empty(0, device=device),
                edge_attr=torch.empty(0, device=device),
                lang_feat=batch["lang_feat"],
                candidate_mask=batch["candidate_mask"],
                node_valid_mask=batch["node_valid_mask"],
            )
            c, n, _ = predict_batch(out["goal_logits"], batch["candidate_mask"], batch["target_index"])
            correct += c
            total += n
    return {"correct": correct, "total": total, "accuracy": 100.0 * correct / total if total else 0.0}


def load_protocol_batch(protocol_dir: Path, max_scen_id: int) -> Tuple[Dict[str, Any], List[int]]:
    items: List[Dict[str, Any]] = []
    scen_ids: List[int] = []
    for scen_id in range(max_scen_id + 1):
        path = protocol_dir / f"scen_{scen_id:03d}.npz"
        if not path.exists():
            continue
        data = np.load(path, allow_pickle=False)
        items.append(
            {
                "node_feats": torch.from_numpy(data["node_feats"].astype(np.float32)),
                "lang_feat": torch.from_numpy(data["lang_feat"].astype(np.float32)),
                "candidate_mask": torch.from_numpy(data["candidate_mask"].astype(np.bool_)),
                "target_index": int(data["target_idx_gt"]),
                "path": str(path),
            }
        )
        scen_ids.append(scen_id)
    return collate_padded(items), scen_ids


def evaluate_protocol(
    model: torch.nn.Module,
    protocol_dir: Path,
    device: torch.device,
    max_scen_id: int,
    out_jsonl: Path,
) -> Dict[str, Any]:
    raw_batch, scen_ids = load_protocol_batch(protocol_dir, max_scen_id)
    batch = move_batch(raw_batch, device)
    model.eval()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        out = model(
            node_feats=batch["node_feats"],
            edge_index=torch.empty(0, device=device),
            edge_attr=torch.empty(0, device=device),
            lang_feat=batch["lang_feat"],
            candidate_mask=batch["candidate_mask"],
            node_valid_mask=batch["node_valid_mask"],
        )
        correct, total, preds = predict_batch(out["goal_logits"], batch["candidate_mask"], batch["target_index"])
    with out_jsonl.open("w", encoding="utf-8") as f:
        for row_idx, scen_id in enumerate(scen_ids):
            target = int(batch["target_index"][row_idx].detach().cpu().item())
            chosen = int(preds[row_idx])
            cand_count = int(batch["candidate_mask"][row_idx].detach().cpu().sum().item())
            f.write(
                json.dumps(
                    {
                        "scen_id": scen_id,
                        "num_candidates": cand_count,
                        "target_idx_gt": target,
                        "chosen_idx": chosen,
                        "gs_correct": chosen == target,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "jsonl": str(out_jsonl),
    }


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


def maybe_subset(ds: Dataset, limit: int, seed: int) -> Dataset | Subset:
    if limit <= 0 or limit >= len(ds):
        return ds
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return Subset(ds, indices[:limit])


def train(args: argparse.Namespace) -> Dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    train_ds = maybe_subset(GoalSelectionDataset(args.train_list), args.limit_train, args.seed)
    val_ds = maybe_subset(GoalSelectionDataset(args.val_list), args.limit_val, args.seed + 1)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_padded)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_padded)

    sample = next(iter(train_loader))
    model = make_model(sample, args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_val = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "query_transformer_best.pt"

    print(f"[Config] device={device} train={len(train_ds)} val={len(val_ds)}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_correct = 0
        for raw_batch in train_loader:
            batch = move_batch(raw_batch, device)
            optimizer.zero_grad()
            out = model(
                node_feats=batch["node_feats"],
                edge_index=torch.empty(0, device=device),
                edge_attr=torch.empty(0, device=device),
                lang_feat=batch["lang_feat"],
                candidate_mask=batch["candidate_mask"],
                node_valid_mask=batch["node_valid_mask"],
            )
            loss = masked_ce_loss(out["goal_logits"], batch["candidate_mask"], batch["target_index"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            c, n, _ = predict_batch(out["goal_logits"], batch["candidate_mask"], batch["target_index"])
            running_loss += float(loss.item()) * n
            running_total += n
            running_correct += c
        scheduler.step()

        val_stats = evaluate_dataset(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": running_loss / running_total if running_total else 0.0,
            "train_accuracy": 100.0 * running_correct / running_total if running_total else 0.0,
            "val_correct": val_stats["correct"],
            "val_total": val_stats["total"],
            "val_accuracy": val_stats["accuracy"],
        }
        history.append(row)
        print(
            f"[query_transformer] epoch {epoch:03d} "
            f"loss={row['train_loss']:.4f} train={row['train_accuracy']:.2f}% "
            f"val={val_stats['accuracy']:.2f}% ({val_stats['correct']}/{val_stats['total']})",
            flush=True,
        )
        if val_stats["accuracy"] > best_val:
            best_val = float(val_stats["accuracy"])
            best_epoch = epoch
            torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
    protocol_stats = evaluate_protocol(
        model,
        args.protocol_dir,
        device,
        args.max_scen_id,
        args.output_dir / "query_transformer_protocol.jsonl",
    )
    history_path = args.output_dir / "query_transformer_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    result = {
        "name": "query_transformer",
        "description": "Query-token Transformer baseline; no edge features or graph message passing.",
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "checkpoint": str(best_path),
        "history": str(history_path),
        "protocol": protocol_stats,
    }
    summary = {
        "seed": args.seed,
        "device": str(device),
        "train_list": str(args.train_list),
        "val_list": str(args.val_list),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": args.epochs,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
        "results": [result],
    }
    summary_path = args.output_dir / "query_transformer_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    proto = protocol_stats
    md_lines = [
        "# Query-token Transformer baseline",
        "",
        "This baseline is query-aware and attention-based, but non-graph: it ignores",
        "edge attributes and uses no graph message passing.",
        "",
        f"- Train list: `{args.train_list.as_posix()}`",
        f"- Val list: `{args.val_list.as_posix()}`",
        f"- Train / val size: {len(train_ds)} / {len(val_ds)}",
        f"- Epochs: {args.epochs}",
        f"- Device: `{device}`",
        f"- Hidden/layers/heads: {args.hidden_dim}/{args.num_layers}/{args.num_heads}",
        f"- Checkpoint: `{best_path}`",
        f"- Protocol JSONL: `{proto['jsonl']}`",
        "",
        "| Model | Best val ProtoAcc@1 | Protocol ProtoAcc@1 |",
        "|---|---:|---:|",
        f"| Query-token Transformer | {best_val:.1f}% | {format_acc(int(proto['correct']), int(proto['total']))} |",
    ]
    md_path = args.output_dir / "query_transformer_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[OK] wrote {summary_path}", flush=True)
    print(f"[OK] wrote {md_path}", flush=True)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_work/experiments_20260521/query_transformer"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-scen-id", type=int, default=29)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
