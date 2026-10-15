#!/usr/bin/env python3
"""Train query-aware non-graph baselines and evaluate protocol graphs.

This script is intentionally independent from CARLA.  It trains lightweight
learnable selectors on saved graph samples and evaluates the resulting model on
the fixed protocol graphs used by the main GS/Random/Oracle comparison.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.baselines.end_to_end import EndToEndNet
from lc_org_nav.baselines.clip_style import CLIPStyleNet
from lc_org_nav.baselines.attention_pool import QueryTokenTransformerNet


class FastGoalSelectionDataset(Dataset):
    """NPZ dataset that reads only fields needed for selection training."""

    def __init__(self, list_file: str | Path):
        path = Path(list_file)
        with path.open("r", encoding="utf-8-sig") as f:
            self.paths = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        if not self.paths:
            raise RuntimeError(f"No samples found in {path}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.paths[idx]
        if not os.path.exists(path):
            raise FileNotFoundError(f"Sample not found: {path}")
        data = np.load(path, allow_pickle=False)

        def arr_float(key: str) -> torch.Tensor:
            arr = data[key]
            if arr.ndim >= 1 and arr.shape[0] == 1 and key in {"node_feats", "edge_attr", "edge_index", "lang_feat", "candidate_mask", "geom"}:
                arr = arr[0]
            return torch.from_numpy(arr.astype(np.float32))

        edge_index = data["edge_index"]
        if edge_index.ndim == 3 and edge_index.shape[0] == 1:
            edge_index = edge_index[0]
        cand = data["candidate_mask"]
        if cand.ndim == 2 and cand.shape[0] == 1:
            cand = cand[0]
        return {
            "node_feats": arr_float("node_feats"),
            "edge_index": torch.from_numpy(edge_index.astype(np.int64)),
            "edge_attr": arr_float("edge_attr"),
            "lang_feat": arr_float("lang_feat"),
            "candidate_mask": torch.from_numpy(cand.astype(np.bool_)),
            "target_index": int(data["target_index"]),
        }


def collate_list(batch: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return list(batch)


def move_sample(sample: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def make_model(name: str, sample0: Dict[str, Any], hidden_dim: int, num_layers: int) -> torch.nn.Module:
    kwargs = dict(
        node_feat_dim=int(sample0["node_feats"].shape[1]),
        edge_feat_dim=int(sample0["edge_attr"].shape[1]),
        lang_feat_dim=int(sample0["lang_feat"].shape[0]),
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        num_rel_types=5,
    )
    if name == "mlp_pool":
        return EndToEndNet(**kwargs)
    if name == "clip_style":
        return CLIPStyleNet(**kwargs)
    if name == "query_transformer":
        return QueryTokenTransformerNet(**kwargs)
    raise ValueError(f"Unknown baseline: {name}")


def selection_loss(outputs: Dict[str, torch.Tensor], sample: Dict[str, Any]) -> torch.Tensor | None:
    logits = outputs["goal_logits"].view(-1)
    cand_mask = sample["candidate_mask"].to(torch.bool).view(-1)
    target_index = int(sample["target_index"])
    cand_indices = torch.nonzero(cand_mask, as_tuple=False).view(-1)
    if cand_indices.numel() == 0:
        return None
    target_matches = torch.nonzero(cand_indices == target_index, as_tuple=False).view(-1)
    if target_matches.numel() == 0:
        return None
    local_target = target_matches[0].view(1)
    return F.cross_entropy(logits[cand_indices].unsqueeze(0), local_target)


def predict_one(model: torch.nn.Module, sample: Dict[str, Any]) -> Tuple[int, int, bool]:
    with torch.no_grad():
        outputs = model(
            node_feats=sample["node_feats"],
            edge_index=sample["edge_index"],
            edge_attr=sample["edge_attr"],
            lang_feat=sample["lang_feat"],
            candidate_mask=sample["candidate_mask"],
        )
        logits = outputs["goal_logits"].view(-1)
        cand_mask_np = sample["candidate_mask"].detach().cpu().numpy().astype(bool)
        cand_indices = np.nonzero(cand_mask_np)[0]
        if len(cand_indices) == 0:
            return -1, int(sample["target_index"]), False
        cand_tensor = torch.from_numpy(cand_indices).long().to(logits.device)
        chosen_local = int(torch.argmax(logits[cand_tensor]).item())
        chosen_idx = int(cand_indices[chosen_local])
        target_idx = int(sample["target_index"])
        return chosen_idx, target_idx, chosen_idx == target_idx


def evaluate_dataset(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            for raw_sample in batch:
                sample = move_sample(raw_sample, device)
                _, _, ok = predict_one(model, sample)
                total += 1
                correct += int(ok)
    return {
        "total": total,
        "correct": correct,
        "accuracy": (100.0 * correct / total) if total else 0.0,
    }


def load_protocol_sample(path: Path, device: torch.device) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    return {
        "node_feats": torch.from_numpy(data["node_feats"]).float().to(device),
        "edge_index": torch.from_numpy(data["edge_index"]).long().to(device),
        "edge_attr": torch.from_numpy(data["edge_attr"]).float().to(device),
        "lang_feat": torch.from_numpy(data["lang_feat"]).float().to(device),
        "candidate_mask": torch.from_numpy(data["candidate_mask"].astype(bool)).to(device),
        "target_index": int(data["target_idx_gt"]),
    }


def evaluate_protocol(
    model: torch.nn.Module,
    protocol_dir: Path,
    device: torch.device,
    max_scen_id: int,
    out_jsonl: Path,
) -> Dict[str, Any]:
    model.eval()
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    with out_jsonl.open("w", encoding="utf-8") as f:
        for scen_id in range(max_scen_id + 1):
            path = protocol_dir / f"scen_{scen_id:03d}.npz"
            if not path.exists():
                continue
            sample = load_protocol_sample(path, device)
            chosen_idx, target_idx, ok = predict_one(model, sample)
            cand_count = int(sample["candidate_mask"].detach().cpu().numpy().sum())
            rec = {
                "scen_id": scen_id,
                "num_candidates": cand_count,
                "target_idx_gt": target_idx,
                "chosen_idx": chosen_idx,
                "gs_correct": bool(ok),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records.append(rec)
    total = len(records)
    correct = sum(1 for item in records if item["gs_correct"])
    return {
        "total": total,
        "correct": correct,
        "accuracy": (100.0 * correct / total) if total else 0.0,
        "jsonl": str(out_jsonl),
    }


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def format_acc(correct: int, total: int) -> str:
    acc = 100.0 * correct / total if total else 0.0
    lo, hi = wilson_ci(correct, total)
    return f"{correct}/{total} ({acc:.1f}%, 95% CI {lo * 100:.1f}-{hi * 100:.1f})"


def train_one(
    name: str,
    train_ds: Dataset,
    val_ds: Dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    sample0 = train_ds[0]
    model = make_model(name, sample0, args.hidden_dim, args.num_layers).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_list,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_list,
        num_workers=0,
    )

    best_val = -1.0
    best_epoch = 0
    model_dir = args.output_dir / "checkpoints"
    model_dir.mkdir(parents=True, exist_ok=True)
    best_path = model_dir / f"{name}_best.pt"
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        skipped = 0
        for batch in train_loader:
            optimizer.zero_grad()
            batch_losses: List[torch.Tensor] = []
            for raw_sample in batch:
                sample = move_sample(raw_sample, device)
                outputs = model(
                    node_feats=sample["node_feats"],
                    edge_index=sample["edge_index"],
                    edge_attr=sample["edge_attr"],
                    lang_feat=sample["lang_feat"],
                    candidate_mask=sample["candidate_mask"],
                )
                loss = selection_loss(outputs, sample)
                if loss is None:
                    skipped += 1
                    continue
                batch_losses.append(loss)
            if not batch_losses:
                continue
            loss_batch = torch.stack(batch_losses).mean()
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running_loss += float(loss_batch.item()) * len(batch_losses)
            running_count += len(batch_losses)

        val_stats = evaluate_dataset(model, val_loader, device)
        avg_loss = running_loss / running_count if running_count else 0.0
        row = {
            "epoch": epoch,
            "train_loss": avg_loss,
            "train_samples": running_count,
            "skipped_samples": skipped,
            "val_correct": val_stats["correct"],
            "val_total": val_stats["total"],
            "val_accuracy": val_stats["accuracy"],
        }
        history.append(row)
        print(
            f"[{name}] epoch {epoch:03d} "
            f"loss={avg_loss:.4f} val={val_stats['accuracy']:.2f}% "
            f"({val_stats['correct']}/{val_stats['total']})",
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
        args.output_dir / f"{name}_protocol.jsonl",
    )
    history_path = args.output_dir / f"{name}_history.json"
    history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {
        "name": name,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "checkpoint": str(best_path),
        "history": str(history_path),
        "protocol": protocol_stats,
    }


def maybe_subset(ds: Dataset, limit: int | None, seed: int) -> Dataset | Subset:
    if limit is None or limit <= 0 or limit >= len(ds):
        return ds
    rng = random.Random(seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    return Subset(ds, indices[:limit])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_work/experiments_20260520/query_baselines"))
    parser.add_argument("--models", nargs="+", default=["mlp_pool"], choices=["mlp_pool", "clip_style", "query_transformer"])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-4)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    train_ds_base = FastGoalSelectionDataset(args.train_list)
    val_ds_base = FastGoalSelectionDataset(args.val_list)
    train_ds = maybe_subset(train_ds_base, args.limit_train, args.seed)
    val_ds = maybe_subset(val_ds_base, args.limit_val, args.seed + 1)

    print(f"[Config] device={device}", flush=True)
    print(f"[Config] train={len(train_ds)} val={len(val_ds)}", flush=True)
    print(f"[Config] protocol_dir={args.protocol_dir}", flush=True)

    results: List[Dict[str, Any]] = []
    for name in args.models:
        results.append(train_one(name, train_ds, val_ds, args, device))

    summary = {
        "seed": args.seed,
        "device": str(device),
        "train_list": str(args.train_list),
        "val_list": str(args.val_list),
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "epochs": args.epochs,
        "results": results,
    }
    summary_path = args.output_dir / "query_baseline_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# Query-aware learnable baseline",
        "",
        f"- Train list: `{args.train_list.as_posix()}`",
        f"- Val list: `{args.val_list.as_posix()}`",
        f"- Train / val size: {len(train_ds)} / {len(val_ds)}",
        f"- Epochs: {args.epochs}",
        f"- Device: `{device}`",
        "",
        "| Model | Best val ProtoAcc@1 | Protocol ProtoAcc@1 | Checkpoint |",
        "|---|---:|---:|---|",
    ]
    for item in results:
        p = item["protocol"]
        md_lines.append(
            "| {name} | {val:.1f}% | {proto} | `{ckpt}` |".format(
                name=item["name"],
                val=item["best_val_accuracy"],
                proto=format_acc(int(p["correct"]), int(p["total"])),
                ckpt=item["checkpoint"],
            )
        )
    md_path = args.output_dir / "query_baseline_table.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {summary_path}", flush=True)
    print(f"[OK] wrote {md_path}", flush=True)


if __name__ == "__main__":
    main()
