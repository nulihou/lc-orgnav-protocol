#!/usr/bin/env python3
"""Evaluate an LC-ORGNav checkpoint ensemble on graph npz files.

The ensemble averages goal logits from graph-selector checkpoints and then
selects the highest-scoring candidate. It is intended for revision diagnostics:
the inputs remain graph selectors, with no MLP baseline scores used.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.revision.eval_lcorg_protocol_no_carla import load_model


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def load_paths_from_list(path: Path) -> List[Path]:
    with path.open("r", encoding="utf-8-sig") as f:
        return [Path(line.strip()) for line in f if line.strip() and not line.lstrip().startswith("#")]


def load_paths_from_protocol_dir(protocol_dir: Path, max_scen_id: int) -> List[Path]:
    return [
        protocol_dir / f"scen_{scen_id:03d}.npz"
        for scen_id in range(max_scen_id + 1)
        if (protocol_dir / f"scen_{scen_id:03d}.npz").exists()
    ]


def target_index_from_npz(data: np.lib.npyio.NpzFile, protocol: bool) -> int:
    if protocol and "target_idx_gt" in data:
        return int(data["target_idx_gt"])
    if "target_index" in data:
        return int(data["target_index"])
    if "target_idx_gt" in data:
        return int(data["target_idx_gt"])
    raise KeyError("missing target_index/target_idx_gt")


def evaluate_paths(
    checkpoints: Sequence[Path],
    graph_paths: Sequence[Path],
    output: Path,
    device: torch.device,
    protocol: bool,
    combine: str,
    weights: Optional[Sequence[float]],
) -> Dict[str, Any]:
    if not graph_paths:
        raise RuntimeError("No graph paths to evaluate")

    reference_dir = graph_paths[0].parent
    models = [load_model(checkpoint, reference_dir, device) for checkpoint in checkpoints]
    if weights is not None and len(weights) != len(models):
        raise ValueError("--weights length must match --checkpoint length")
    if weights is None:
        weights_tensor = torch.ones(len(models), dtype=torch.float32, device=device) / max(len(models), 1)
    else:
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
        weights_tensor = weights_tensor / weights_tensor.sum().clamp_min(1e-8)
    output.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with output.open("w", encoding="utf-8") as f:
        for item_id, path in enumerate(graph_paths):
            data = np.load(path, allow_pickle=False)
            node_feats = torch.from_numpy(data["node_feats"]).float().to(device)
            edge_index = torch.from_numpy(data["edge_index"]).long().to(device)
            edge_attr = torch.from_numpy(data["edge_attr"]).float().to(device)
            lang_feat = torch.from_numpy(data["lang_feat"]).float().to(device)
            cand_mask_np = data["candidate_mask"].astype(bool)
            candidate_mask = torch.from_numpy(cand_mask_np).bool().to(device)
            target_idx = target_index_from_npz(data, protocol=protocol)

            model_logits: List[torch.Tensor] = []
            model_choices: List[Dict[str, Any]] = []
            with torch.no_grad():
                for model in models:
                    outputs = model.forward_single(
                        node_feats=node_feats,
                        edge_index=edge_index,
                        edge_attr=edge_attr,
                        lang_feat=lang_feat,
                        candidate_mask=candidate_mask,
                    )
                    logits = outputs["goal_logits"].view(-1)
                    model_logits.append(logits)

            cand_indices = np.nonzero(cand_mask_np)[0]
            if len(cand_indices) == 0:
                chosen_idx = -1
            else:
                cand_tensor = torch.from_numpy(cand_indices).long().to(device)
                if combine == "mean":
                    logits_stack = torch.stack(model_logits, dim=0)
                    logits_sum = (weights_tensor.view(-1, 1) * logits_stack).sum(dim=0)
                    chosen_idx = int(cand_indices[int(torch.argmax(logits_sum[cand_tensor]).item())])
                elif combine in {"max_margin", "max_prob"}:
                    for model_id, logits in enumerate(model_logits):
                        cand_logits = logits[cand_tensor]
                        probs = torch.softmax(cand_logits, dim=0)
                        top = torch.topk(probs, k=min(2, probs.numel()))
                        top_prob = float(top.values[0].item())
                        margin = float((top.values[0] - top.values[1]).item()) if probs.numel() > 1 else top_prob
                        chosen_local = int(top.indices[0].item())
                        model_choices.append(
                            {
                                "model_id": model_id,
                                "chosen_idx": int(cand_indices[chosen_local]),
                                "top_prob": top_prob,
                                "margin": margin,
                            }
                        )
                    score_key = "margin" if combine == "max_margin" else "top_prob"
                    selected = max(model_choices, key=lambda item: item[score_key])
                    chosen_idx = int(selected["chosen_idx"])
                else:
                    raise ValueError(f"unknown combine mode: {combine}")

            scen_id = int(path.stem.split("_")[-1]) if path.stem.startswith("scen_") else item_id
            rec = {
                "item_id": item_id,
                "scen_id": scen_id,
                "path": str(path),
                "num_candidates": int(cand_mask_np.sum()),
                "target_idx_gt": int(target_idx),
                "chosen_idx": int(chosen_idx),
                "gs_correct": bool(chosen_idx == target_idx),
                "model_choices": model_choices,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    correct = sum(1 for rec in records if rec["gs_correct"])
    total = len(records)
    lo, hi = wilson_ci(correct, total)
    return {
        "checkpoints": [str(path) for path in checkpoints],
        "output": str(output),
        "total": total,
        "correct": correct,
        "accuracy": 100.0 * correct / total if total else 0.0,
        "ci95": [lo * 100.0, hi * 100.0],
        "combine": combine,
        "weights": [float(item) for item in weights_tensor.detach().cpu().tolist()],
        "wrong_scenarios": [rec["scen_id"] for rec in records if not rec["gs_correct"]],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    parser.add_argument("--graph-list", type=Path)
    parser.add_argument("--protocol-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-scen-id", type=int, default=299)
    parser.add_argument("--protocol", action="store_true")
    parser.add_argument("--combine", choices=["mean", "max_margin", "max_prob"], default="mean")
    parser.add_argument("--weights", type=float, nargs="+")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.graph_list is None and args.protocol_dir is None:
        raise SystemExit("Either --graph-list or --protocol-dir is required")
    if args.graph_list is not None:
        graph_paths = load_paths_from_list(args.graph_list)
    else:
        graph_paths = load_paths_from_protocol_dir(args.protocol_dir, args.max_scen_id)

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    summary = evaluate_paths(
        checkpoints=args.checkpoint,
        graph_paths=graph_paths,
        output=args.output,
        device=torch.device(device_name),
        protocol=bool(args.protocol),
        combine=args.combine,
        weights=args.weights,
    )
    summary_path = args.summary or args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
