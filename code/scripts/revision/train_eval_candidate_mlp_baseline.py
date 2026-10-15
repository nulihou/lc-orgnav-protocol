#!/usr/bin/env python3
"""Fast query-aware MLP+pooling baseline for protocol target selection.

The baseline scores each candidate independently from:
  [candidate node features, mean-pooled scene features, max-pooled scene features, query embedding]

It is intentionally non-graph: no edge features and no message passing are used.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class CandidateMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def read_list(path: Path, limit: int = 0, seed: int = 42) -> List[Path]:
    with path.open("r", encoding="utf-8-sig") as f:
        items = [Path(line.strip()) for line in f if line.strip() and not line.startswith("#")]
    if limit and limit > 0 and limit < len(items):
        rng = random.Random(seed)
        rng.shuffle(items)
        items = items[:limit]
    return items


def _squeeze_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def candidate_features_from_npz(path: Path, protocol: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    lang_feat = _squeeze_first(data["lang_feat"]).astype(np.float32)
    candidate_mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    target_key = "target_idx_gt" if protocol else "target_index"
    target_index = int(data[target_key])

    cand_indices = np.nonzero(candidate_mask)[0]
    if len(cand_indices) == 0:
        return np.zeros((0, node_feats.shape[1] * 3 + lang_feat.shape[0]), dtype=np.float32), cand_indices, np.zeros((0,), dtype=np.float32)

    mean_pool = node_feats.mean(axis=0)
    max_pool = node_feats.max(axis=0)
    rows: List[np.ndarray] = []
    labels: List[float] = []
    for idx in cand_indices:
        rows.append(np.concatenate([node_feats[idx], mean_pool, max_pool, lang_feat], axis=0))
        labels.append(1.0 if int(idx) == target_index else 0.0)
    return np.stack(rows).astype(np.float32), cand_indices.astype(np.int64), np.asarray(labels, dtype=np.float32)


def build_training_matrix(paths: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray]:
    xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    skipped = 0
    for path in paths:
        try:
            x, _, y = candidate_features_from_npz(path, protocol=False)
        except Exception:
            skipped += 1
            continue
        if len(y) == 0:
            skipped += 1
            continue
        xs.append(x)
        ys.append(y)
    if not xs:
        raise RuntimeError("No usable candidate samples found")
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    print(f"[Data] candidate rows={len(y_all)} positives={int(y_all.sum())} skipped_scenes={skipped}", flush=True)
    return x_all, y_all


def evaluate_paths(model: CandidateMLP, paths: Sequence[Path], device: torch.device, protocol: bool = False) -> Dict[str, Any]:
    model.eval()
    total = 0
    correct = 0
    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for i, path in enumerate(paths):
            try:
                x, cand_indices, y = candidate_features_from_npz(path, protocol=protocol)
            except Exception as exc:
                records.append({"path": str(path), "valid": False, "error": str(exc)})
                continue
            if len(y) == 0:
                records.append({"path": str(path), "valid": False, "error": "no candidates"})
                continue
            logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
            best_local = int(np.argmax(logits))
            chosen_idx = int(cand_indices[best_local])
            target_idx = int(cand_indices[int(np.argmax(y))])
            ok = chosen_idx == target_idx
            records.append({
                "item_id": i,
                "path": str(path),
                "valid": True,
                "chosen_idx": chosen_idx,
                "target_idx_gt": target_idx,
                "gs_correct": bool(ok),
                "num_candidates": int(len(cand_indices)),
            })
            total += 1
            correct += int(ok)
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "records": records,
    }


def protocol_paths(protocol_dir: Path, max_scen_id: int) -> List[Path]:
    return [protocol_dir / f"scen_{sid:03d}.npz" for sid in range(max_scen_id + 1) if (protocol_dir / f"scen_{sid:03d}.npz").exists()]


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def write_protocol_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-list", type=Path, required=True)
    parser.add_argument("--val-list", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("scripts/online_multi/protocol_graphs"))
    parser.add_argument("--output-dir", type=Path, default=Path("revision_work/experiments_20260520/query_baselines"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    parser.add_argument("--max-scen-id", type=int, default=29)
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

    train_paths = read_list(args.train_list, args.limit_train, args.seed)
    val_paths = read_list(args.val_list, args.limit_val, args.seed + 1)
    print(f"[Config] train_scenes={len(train_paths)} val_scenes={len(val_paths)} device={device}", flush=True)

    x_train, y_train = build_training_matrix(train_paths)
    input_dim = int(x_train.shape[1])
    model = CandidateMLP(input_dim=input_dim, hidden_dim=args.hidden_dim).to(device)

    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    best_val = -1.0
    best_epoch = 0
    history: List[Dict[str, Any]] = []
    ckpt_path = args.output_dir / "candidate_mlp_best.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(yb)
            total_rows += len(yb)
        val_stats = evaluate_paths(model, val_paths, device, protocol=False)
        avg_loss = total_loss / max(total_rows, 1)
        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "val_correct": val_stats["correct"],
            "val_total": val_stats["total"],
            "val_accuracy": val_stats["accuracy"],
        })
        print(
            f"[candidate_mlp] epoch {epoch:03d} "
            f"loss={avg_loss:.4f} val={val_stats['accuracy']:.2f}% "
            f"({val_stats['correct']}/{val_stats['total']})",
            flush=True,
        )
        if val_stats["accuracy"] > best_val:
            best_val = float(val_stats["accuracy"])
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
    proto_stats = evaluate_paths(model, protocol_paths(args.protocol_dir, args.max_scen_id), device, protocol=True)
    protocol_jsonl = args.output_dir / "candidate_mlp_protocol.jsonl"
    write_protocol_jsonl(protocol_jsonl, proto_stats["records"])

    lo, hi = wilson_ci(int(proto_stats["correct"]), int(proto_stats["total"]))
    summary = {
        "model": "candidate_mlp_pooling",
        "train_list": str(args.train_list),
        "val_list": str(args.val_list),
        "train_scenes": len(train_paths),
        "val_scenes": len(val_paths),
        "candidate_rows": int(len(y_train)),
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val,
        "checkpoint": str(ckpt_path),
        "protocol_jsonl": str(protocol_jsonl),
        "protocol": {
            "total": int(proto_stats["total"]),
            "correct": int(proto_stats["correct"]),
            "accuracy": float(proto_stats["accuracy"]),
            "ci95": [100.0 * lo, 100.0 * hi],
        },
    }
    (args.output_dir / "candidate_mlp_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.output_dir / "candidate_mlp_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    md = [
        "# Query-aware MLP+pooling baseline",
        "",
        f"- Train / val scenes: {len(train_paths)} / {len(val_paths)}",
        f"- Candidate rows: {len(y_train)}",
        f"- Best validation ProtoAcc@1: {best_val:.1f}% at epoch {best_epoch}",
        f"- Protocol ProtoAcc@1: {proto_stats['correct']}/{proto_stats['total']} ({proto_stats['accuracy']:.1f}%, 95% CI {lo * 100:.1f}-{hi * 100:.1f})",
        f"- Checkpoint: `{ckpt_path}`",
        f"- Protocol JSONL: `{protocol_jsonl}`",
        "",
        "This baseline is query-aware but non-graph: it uses no edge features and no message passing.",
    ]
    (args.output_dir / "candidate_mlp_table.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[OK] wrote {args.output_dir / 'candidate_mlp_table.md'}", flush=True)


if __name__ == "__main__":
    main()
