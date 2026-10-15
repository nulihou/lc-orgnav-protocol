#!/usr/bin/env python3
"""Evaluate saved selectors on a large offline graph list without CARLA.

This script is intended for revision-time robustness checks. It evaluates
static goal selection on saved NPZ graphs and reports exact binomial summaries,
candidate-count breakdowns, and simple non-learned baselines.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.model import LCORGNet
try:
    from scripts.analysis.train_eval_candidate_mlp_baseline import (
        CandidateMLP,
        candidate_features_from_npz,
    )
except ModuleNotFoundError:
    from scripts.revision.train_eval_candidate_mlp_baseline import (
        CandidateMLP,
        candidate_features_from_npz,
    )


def read_paths(path: Path, limit: int = 0) -> List[Path]:
    with path.open("r", encoding="utf-8-sig") as f:
        items = [Path(line.strip()) for line in f if line.strip() and not line.lstrip().startswith("#")]
    if limit and limit > 0:
        items = items[:limit]
    return items


def _squeeze_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def load_graph(path: Path, device: torch.device) -> Dict[str, Any]:
    data = np.load(path, allow_pickle=False)
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    edge_index = _squeeze_first(data["edge_index"]).astype(np.int64)
    edge_attr = _squeeze_first(data["edge_attr"]).astype(np.float32)
    lang_feat = _squeeze_first(data["lang_feat"]).astype(np.float32)
    candidate_mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    return {
        "path": str(path),
        "node_feats_np": node_feats,
        "candidate_mask_np": candidate_mask,
        "node_feats": torch.from_numpy(node_feats).float().to(device),
        "edge_index": torch.from_numpy(edge_index).long().to(device),
        "edge_attr": torch.from_numpy(edge_attr).float().to(device),
        "lang_feat": torch.from_numpy(lang_feat).float().to(device),
        "candidate_mask": torch.from_numpy(candidate_mask).bool().to(device),
        "target_index": int(data["target_index"]),
        "rel_type": int(data["rel_type"]) if "rel_type" in data.files else -1,
        "num_candidates": int(candidate_mask.sum()),
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


def normalize_state_dict(state: Any) -> Dict[str, torch.Tensor]:
    if isinstance(state, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError("Checkpoint does not contain a state dict")
    return state


def load_lcorg_model(checkpoint: Path, sample_path: Path, device: torch.device) -> LCORGNet:
    data = np.load(sample_path, allow_pickle=False)
    node_feats = _squeeze_first(data["node_feats"])
    edge_attr = _squeeze_first(data["edge_attr"])
    lang_feat = _squeeze_first(data["lang_feat"])
    raw_state = torch.load(checkpoint, map_location=device, weights_only=False)
    state = normalize_state_dict(raw_state)
    use_rel_gate = any(("rel_proj" in key or "gate_mlp" in key) for key in state.keys())
    use_edge_context_head = any(
        (
            "raw_node_proj" in key
            or "edge_context_proj" in key
            or "graph_pool_proj" in key
            or "edge_goal_head" in key
        )
        for key in state.keys()
    )
    model = LCORGNet(
        node_feat_dim=int(node_feats.shape[1]),
        edge_feat_dim=int(edge_attr.shape[1]),
        lang_feat_dim=int(lang_feat.shape[0]),
        hidden_dim=128,
        num_layers=3,
        num_rel_types=5,
        use_rel_gate=use_rel_gate,
        use_edge_context_head=use_edge_context_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_lcorg(model: LCORGNet, sample: Dict[str, Any]) -> Tuple[int, List[int]]:
    with torch.no_grad():
        outputs = model.forward_single(
            node_feats=sample["node_feats"],
            edge_index=sample["edge_index"],
            edge_attr=sample["edge_attr"],
            lang_feat=sample["lang_feat"],
            candidate_mask=sample["candidate_mask"],
        )
        logits = outputs["goal_logits"].view(-1)
        cand_indices = np.nonzero(sample["candidate_mask_np"])[0]
        cand_tensor = torch.from_numpy(cand_indices).long().to(logits.device)
        order_local = torch.argsort(logits[cand_tensor], descending=True).detach().cpu().numpy()
    ordered = [int(cand_indices[int(i)]) for i in order_local]
    return (ordered[0] if ordered else -1), ordered


def predict_nearest(sample: Dict[str, Any]) -> int:
    cand_indices = np.nonzero(sample["candidate_mask_np"])[0]
    if len(cand_indices) == 0:
        return -1
    distances = sample["node_feats_np"][cand_indices, 7]
    return int(cand_indices[int(np.argmin(distances))])


def summarize_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in records if r.get("valid", True)]
    total = len(valid)
    correct = sum(1 for r in valid if r["correct"])
    top3 = sum(1 for r in valid if r.get("top3_correct", False))
    by_k: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    by_rel: Dict[str, Dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    for rec in valid:
        k_key = str(rec["num_candidates"])
        rel_key = str(rec.get("rel_type", -1))
        by_k[k_key]["total"] += 1
        by_k[k_key]["correct"] += int(rec["correct"])
        by_rel[rel_key]["total"] += 1
        by_rel[rel_key]["correct"] += int(rec["correct"])
    lo, hi = wilson_ci(correct, total)
    top3_lo, top3_hi = wilson_ci(top3, total)
    return {
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "ci95": [100.0 * lo, 100.0 * hi],
        "top3_correct": top3,
        "top3_accuracy": 100.0 * top3 / total if total else 0.0,
        "top3_ci95": [100.0 * top3_lo, 100.0 * top3_hi],
        "by_k": dict(sorted(by_k.items(), key=lambda kv: int(kv[0]))),
        "by_rel_type": dict(sorted(by_rel.items(), key=lambda kv: int(kv[0]))),
    }


def eval_lcorg(
    name: str,
    model: LCORGNet,
    paths: Sequence[Path],
    device: torch.device,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item_id, path in enumerate(paths):
        try:
            sample = load_graph(path, device)
            chosen, ordered = predict_lcorg(model, sample)
            target = int(sample["target_index"])
            rec = {
                "item_id": item_id,
                "path": str(path),
                "model": name,
                "valid": True,
                "num_candidates": int(sample["num_candidates"]),
                "rel_type": int(sample["rel_type"]),
                "target_idx": target,
                "chosen_idx": chosen,
                "top3": ordered[:3],
                "correct": bool(chosen == target),
                "top3_correct": bool(target in ordered[: min(3, len(ordered))]),
            }
        except Exception as exc:
            rec = {
                "item_id": item_id,
                "path": str(path),
                "model": name,
                "valid": False,
                "error": str(exc),
                "correct": False,
                "top3_correct": False,
            }
        records.append(rec)
    return summarize_records(records), records


def eval_candidate_mlp(
    checkpoint: Path,
    paths: Sequence[Path],
    device: torch.device,
    hidden_dim: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    first_x, _, _ = candidate_features_from_npz(paths[0], protocol=False)
    model = CandidateMLP(input_dim=int(first_x.shape[1]), hidden_dim=hidden_dim).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.eval()
    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for item_id, path in enumerate(paths):
            try:
                x, cand_indices, y = candidate_features_from_npz(path, protocol=False)
                if len(y) == 0:
                    raise RuntimeError("no candidates")
                logits = model(torch.from_numpy(x).to(device)).detach().cpu().numpy()
                order_local = np.argsort(-logits)
                ordered = [int(cand_indices[int(i)]) for i in order_local]
                chosen = ordered[0] if ordered else -1
                target = int(cand_indices[int(np.argmax(y))])
                data = np.load(path, allow_pickle=False)
                rel_type = int(data["rel_type"]) if "rel_type" in data.files else -1
                rec = {
                    "item_id": item_id,
                    "path": str(path),
                    "model": "MLP+pooling",
                    "valid": True,
                    "num_candidates": int(len(cand_indices)),
                    "rel_type": rel_type,
                    "target_idx": target,
                    "chosen_idx": chosen,
                    "top3": ordered[:3],
                    "correct": bool(chosen == target),
                    "top3_correct": bool(target in ordered[: min(3, len(ordered))]),
                }
            except Exception as exc:
                rec = {
                    "item_id": item_id,
                    "path": str(path),
                    "model": "MLP+pooling",
                    "valid": False,
                    "error": str(exc),
                    "correct": False,
                    "top3_correct": False,
                }
            records.append(rec)
    return summarize_records(records), records


def eval_nearest(paths: Sequence[Path], device: torch.device) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    records: List[Dict[str, Any]] = []
    for item_id, path in enumerate(paths):
        try:
            sample = load_graph(path, device)
            chosen = predict_nearest(sample)
            target = int(sample["target_index"])
            rec = {
                "item_id": item_id,
                "path": str(path),
                "model": "Nearest",
                "valid": True,
                "num_candidates": int(sample["num_candidates"]),
                "rel_type": int(sample["rel_type"]),
                "target_idx": target,
                "chosen_idx": chosen,
                "correct": bool(chosen == target),
                "top3_correct": False,
            }
        except Exception as exc:
            rec = {
                "item_id": item_id,
                "path": str(path),
                "model": "Nearest",
                "valid": False,
                "error": str(exc),
                "correct": False,
                "top3_correct": False,
            }
        records.append(rec)
    return summarize_records(records), records


def random_expected(paths: Sequence[Path], device: torch.device) -> Dict[str, Any]:
    total = 0
    expected_correct = 0.0
    by_k: Dict[str, Dict[str, float]] = defaultdict(lambda: {"total": 0, "expected_correct": 0.0})
    for path in paths:
        sample = load_graph(path, device)
        k = int(sample["num_candidates"])
        if k <= 0:
            continue
        total += 1
        expected_correct += 1.0 / k
        by_k[str(k)]["total"] += 1
        by_k[str(k)]["expected_correct"] += 1.0 / k
    acc = 100.0 * expected_correct / total if total else 0.0
    return {
        "total": total,
        "expected_correct": expected_correct,
        "expected_accuracy": acc,
        "by_k": dict(sorted(by_k.items(), key=lambda kv: int(kv[0]))),
    }


def write_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def model_summary_line(name: str, stats: Dict[str, Any]) -> str:
    return "| {name} | {top1} | {top3} |".format(
        name=name,
        top1=format_acc(int(stats["correct"]), int(stats["total"])),
        top3=format_acc(int(stats["top3_correct"]), int(stats["total"])),
    )


def write_markdown(path: Path, args: argparse.Namespace, summary: Dict[str, Any]) -> None:
    lines = [
        "# Large offline validation selection check",
        "",
        f"- Graph list: `{args.graph_list}`",
        f"- Number of graphs: {summary['num_graphs']}",
        f"- Device: `{summary['device']}`",
        f"- Full checkpoint: `{args.full_checkpoint}`",
        f"- No-FiLM checkpoint: `{args.no_film_checkpoint}`",
        f"- MLP+pooling checkpoint: `{args.candidate_mlp_checkpoint}`",
        "",
        "| Method | ProtoAcc@1 | ProtoAcc@3 |",
        "|---|---:|---:|",
        model_summary_line("LCORG full", summary["models"]["LCORG full"]),
        model_summary_line("w/o FiLM", summary["models"]["w/o FiLM"]),
        model_summary_line("MLP+pooling", summary["models"]["MLP+pooling"]),
        model_summary_line("Nearest heuristic", summary["models"]["Nearest heuristic"]),
        "| Random expected | {acc:.1f}% expected | -- |".format(
            acc=summary["models"]["Random expected"]["expected_accuracy"]
        ),
        "",
        "## Candidate-count distribution",
        "",
        "| K | Graphs | Random expected | LCORG full | w/o FiLM | MLP+pooling | Nearest |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    k_keys = sorted(summary["candidate_count_distribution"].keys(), key=int)
    for k in k_keys:
        n = summary["candidate_count_distribution"][k]
        row = [k, str(n), f"{100.0 / int(k):.1f}%"]
        for model_name in ("LCORG full", "w/o FiLM", "MLP+pooling", "Nearest heuristic"):
            cell = summary["models"][model_name]["by_k"].get(k, {"total": 0, "correct": 0})
            total = int(cell["total"])
            correct = int(cell["correct"])
            row.append(f"{correct}/{total} ({100.0 * correct / total:.1f}%)" if total else "--")
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-list", type=Path, required=True)
    parser.add_argument("--full-checkpoint", type=Path, required=True)
    parser.add_argument("--no-film-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-mlp-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--candidate-mlp-hidden-dim", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    paths = read_paths(args.graph_list, args.limit)
    if not paths:
        raise RuntimeError(f"No graph paths found in {args.graph_list}")

    full_model = load_lcorg_model(args.full_checkpoint, paths[0], device)
    no_film_model = load_lcorg_model(args.no_film_checkpoint, paths[0], device)

    model_stats: Dict[str, Any] = {}
    all_records: List[Dict[str, Any]] = []

    full_stats, full_records = eval_lcorg("LCORG full", full_model, paths, device)
    model_stats["LCORG full"] = full_stats
    all_records.extend(full_records)

    no_film_stats, no_film_records = eval_lcorg("w/o FiLM", no_film_model, paths, device)
    model_stats["w/o FiLM"] = no_film_stats
    all_records.extend(no_film_records)

    mlp_stats, mlp_records = eval_candidate_mlp(
        args.candidate_mlp_checkpoint,
        paths,
        device,
        args.candidate_mlp_hidden_dim,
    )
    model_stats["MLP+pooling"] = mlp_stats
    all_records.extend(mlp_records)

    nearest_stats, nearest_records = eval_nearest(paths, device)
    model_stats["Nearest heuristic"] = nearest_stats
    all_records.extend(nearest_records)

    model_stats["Random expected"] = random_expected(paths, device)

    candidate_distribution: Dict[str, int] = defaultdict(int)
    for path in paths:
        sample = load_graph(path, device)
        candidate_distribution[str(sample["num_candidates"])] += 1

    summary = {
        "graph_list": str(args.graph_list),
        "num_graphs": len(paths),
        "device": str(device),
        "models": model_stats,
        "candidate_count_distribution": dict(sorted(candidate_distribution.items(), key=lambda kv: int(kv[0]))),
    }

    summary_path = args.output_dir / "large_offline_selection_summary.json"
    records_path = args.output_dir / "large_offline_selection_records.jsonl"
    table_path = args.output_dir / "large_offline_selection_table.md"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_jsonl(records_path, all_records)
    write_markdown(table_path, args, summary)

    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {summary_path}")
    print(f"[OK] wrote {records_path}")
    print(f"[OK] wrote {table_path}")


if __name__ == "__main__":
    main()
