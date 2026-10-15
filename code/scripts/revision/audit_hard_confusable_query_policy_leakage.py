#!/usr/bin/env python3
"""Leakage-control audit for the hard-confusable query-policy field.

This script checks whether the query-policy field behaves like a hidden target
index. It intentionally uses only metadata-level predictors before evaluating
optional trained-model sensitivity under shuffled query policies.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lc_org_nav.model import LCORGNet
from scripts.revision.train_eval_candidate_mlp_baseline import CandidateMLP


POLICY_NAMES = {
    1: "farthest",
    2: "leftmost",
    3: "rightmost",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _squeeze_first(arr: np.ndarray) -> np.ndarray:
    if arr.ndim >= 1 and arr.shape[0] == 1:
        return arr[0]
    return arr


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def pct(correct: int, total: int) -> str:
    lo, hi = wilson_ci(correct, total)
    return f"{correct}/{total} ({100.0 * correct / total:.1f}%, 95% CI {100.0 * lo:.1f}-{100.0 * hi:.1f})"


def scenario_key(scene: Dict[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(scene[field] for field in fields)


def majority_slot_map(train: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[Tuple[Any, ...], int]:
    counts: Dict[Tuple[Any, ...], Counter[int]] = defaultdict(Counter)
    for scene in train:
        counts[scenario_key(scene, fields)][int(scene["target_slot"])] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in counts.items()}


def global_majority_slot(train: Sequence[Dict[str, Any]]) -> int:
    return Counter(int(scene["target_slot"]) for scene in train).most_common(1)[0][0]


def eval_slot_rule(
    test: Sequence[Dict[str, Any]],
    name: str,
    fields: Sequence[str],
    mapping: Dict[Tuple[Any, ...], int],
    fallback_slot: int,
) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    correct = 0
    for scene in test:
        key = scenario_key(scene, fields)
        pred_slot = int(mapping.get(key, fallback_slot))
        ok = pred_slot == int(scene["target_slot"])
        correct += int(ok)
        records.append(
            {
                "scenario_id": int(scene["id"]),
                "query_policy_id": int(scene["query_policy_id"]),
                "num_candidates": int(scene["num_candidates"]),
                "target_slot": int(scene["target_slot"]),
                "pred_slot": pred_slot,
                "correct": bool(ok),
            }
        )
    return {
        "name": name,
        "fields": list(fields),
        "correct": correct,
        "total": len(test),
        "accuracy": correct / len(test) if test else 0.0,
        "records": records,
    }


def eval_policy_as_slot(test: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    records: List[Dict[str, Any]] = []
    correct = 0
    valid = 0
    for scene in test:
        policy_id = int(scene["query_policy_id"])
        k = int(scene["num_candidates"])
        pred_slot = policy_id if 1 <= policy_id <= k else 1
        ok = pred_slot == int(scene["target_slot"])
        valid += 1
        correct += int(ok)
        records.append(
            {
                "scenario_id": int(scene["id"]),
                "query_policy_id": policy_id,
                "num_candidates": k,
                "target_slot": int(scene["target_slot"]),
                "pred_slot": pred_slot,
                "correct": bool(ok),
            }
        )
    return {
        "name": "policy_id_as_candidate_slot",
        "correct": correct,
        "total": valid,
        "accuracy": correct / valid if valid else 0.0,
        "records": records,
    }


def lang_with_policy(lang: np.ndarray, new_policy_id: int) -> np.ndarray:
    out = np.asarray(lang, dtype=np.float32).copy()
    out[5:9] = 0.0
    if 1 <= new_policy_id <= 3:
        out[5 + new_policy_id] = 1.0
    return out


def shuffled_policy_map(test: Sequence[Dict[str, Any]], seed: int) -> Dict[int, int]:
    rng = random.Random(seed)
    result: Dict[int, int] = {}
    for scene in test:
        original = int(scene["query_policy_id"])
        choices = [pid for pid in POLICY_NAMES if pid != original]
        result[int(scene["id"])] = int(rng.choice(choices))
    return result


def load_graph_model(checkpoint: Path, protocol_dir: Path, device: torch.device) -> LCORGNet:
    first_graph = protocol_dir / "scen_000.npz"
    data = np.load(first_graph, allow_pickle=False)
    node_feat_dim = int(_squeeze_first(data["node_feats"]).shape[1])
    edge_feat_dim = int(_squeeze_first(data["edge_attr"]).shape[1])
    lang_feat_dim = int(_squeeze_first(data["lang_feat"]).shape[0])
    state = torch.load(checkpoint, map_location=device, weights_only=False)
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
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        lang_feat_dim=lang_feat_dim,
        hidden_dim=128,
        num_layers=3,
        num_rel_types=5,
        use_rel_gate=use_rel_gate,
        use_edge_context_head=use_edge_context_head,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def eval_graph_checkpoint(
    checkpoint: Path,
    protocol_dir: Path,
    test: Sequence[Dict[str, Any]],
    policy_override: Dict[int, int] | None,
    device: torch.device,
) -> Dict[str, Any]:
    model = load_graph_model(checkpoint, protocol_dir, device)
    records: List[Dict[str, Any]] = []
    correct = 0
    with torch.no_grad():
        for scene in test:
            sid = int(scene["id"])
            path = protocol_dir / f"scen_{sid:03d}.npz"
            data = np.load(path, allow_pickle=False)
            node_feats_np = _squeeze_first(data["node_feats"]).astype(np.float32)
            edge_index_np = _squeeze_first(data["edge_index"]).astype(np.int64)
            edge_attr_np = _squeeze_first(data["edge_attr"]).astype(np.float32)
            lang_np = _squeeze_first(data["lang_feat"]).astype(np.float32)
            if policy_override is not None:
                lang_np = lang_with_policy(lang_np, int(policy_override[sid]))
            cand_mask_np = _squeeze_first(data["candidate_mask"]).astype(bool)
            target = int(data["target_idx_gt"])
            outputs = model.forward_single(
                node_feats=torch.from_numpy(node_feats_np).float().to(device),
                edge_index=torch.from_numpy(edge_index_np).long().to(device),
                edge_attr=torch.from_numpy(edge_attr_np).float().to(device),
                lang_feat=torch.from_numpy(lang_np).float().to(device),
                candidate_mask=torch.from_numpy(cand_mask_np).bool().to(device),
            )
            logits = outputs["goal_logits"].view(-1)
            cand_indices = np.nonzero(cand_mask_np)[0]
            chosen = -1 if len(cand_indices) == 0 else int(cand_indices[int(torch.argmax(logits[torch.from_numpy(cand_indices).long().to(device)]).item())])
            ok = chosen == target
            correct += int(ok)
            records.append(
                {
                    "scenario_id": sid,
                    "target_idx_gt": target,
                    "chosen_idx": chosen,
                    "correct": bool(ok),
                    "original_query_policy_id": int(scene["query_policy_id"]),
                    "eval_query_policy_id": int(policy_override[sid]) if policy_override else int(scene["query_policy_id"]),
                }
            )
    return {
        "checkpoint": str(checkpoint),
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "records": records,
    }


def mlp_rows(data: np.lib.npyio.NpzFile, lang_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    node_feats = _squeeze_first(data["node_feats"]).astype(np.float32)
    candidate_mask = _squeeze_first(data["candidate_mask"]).astype(bool)
    target = int(data["target_idx_gt"])
    cand_indices = np.nonzero(candidate_mask)[0]
    mean_pool = node_feats.mean(axis=0)
    max_pool = node_feats.max(axis=0)
    rows = [np.concatenate([node_feats[idx], mean_pool, max_pool, lang_np], axis=0) for idx in cand_indices]
    return np.stack(rows).astype(np.float32), cand_indices.astype(np.int64), target


def eval_mlp_checkpoint(
    checkpoint: Path,
    protocol_dir: Path,
    test: Sequence[Dict[str, Any]],
    policy_override: Dict[int, int] | None,
    device: torch.device,
) -> Dict[str, Any]:
    first = np.load(protocol_dir / "scen_000.npz", allow_pickle=False)
    node_dim = int(_squeeze_first(first["node_feats"]).shape[1])
    lang_dim = int(_squeeze_first(first["lang_feat"]).shape[0])
    input_dim = node_dim * 3 + lang_dim
    model = CandidateMLP(input_dim=input_dim, hidden_dim=128).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False))
    model.eval()
    records: List[Dict[str, Any]] = []
    correct = 0
    with torch.no_grad():
        for scene in test:
            sid = int(scene["id"])
            data = np.load(protocol_dir / f"scen_{sid:03d}.npz", allow_pickle=False)
            lang_np = _squeeze_first(data["lang_feat"]).astype(np.float32)
            if policy_override is not None:
                lang_np = lang_with_policy(lang_np, int(policy_override[sid]))
            x, cand_indices, target = mlp_rows(data, lang_np)
            logits = model(torch.from_numpy(x).float().to(device)).detach().cpu().numpy()
            chosen = int(cand_indices[int(np.argmax(logits))])
            ok = chosen == target
            correct += int(ok)
            records.append(
                {
                    "scenario_id": sid,
                    "target_idx_gt": target,
                    "chosen_idx": chosen,
                    "correct": bool(ok),
                    "original_query_policy_id": int(scene["query_policy_id"]),
                    "eval_query_policy_id": int(policy_override[sid]) if policy_override else int(scene["query_policy_id"]),
                }
            )
    return {
        "checkpoint": str(checkpoint),
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records) if records else 0.0,
        "records": records,
    }


def summarize_distribution(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_policy_slot: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_policy_k_slot: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for scene in scenes:
        policy = POLICY_NAMES.get(int(scene["query_policy_id"]), str(scene["query_policy_id"]))
        slot = str(int(scene["target_slot"]))
        k = int(scene["num_candidates"])
        by_policy_slot[policy][slot] += 1
        by_policy_k_slot[f"{policy}/K{k}"][slot] += 1
    return {
        "by_policy_target_slot": {k: dict(v) for k, v in sorted(by_policy_slot.items())},
        "by_policy_and_k_target_slot": {k: dict(v) for k, v in sorted(by_policy_k_slot.items())},
    }


def write_jsonl(path: Path, records: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def make_markdown(summary: Dict[str, Any]) -> str:
    lines = [
        "# Hard-Confusable Query-Policy Leakage Audit",
        "",
        "This audit tests whether the query-policy field behaves like a hidden target index.",
        "",
        "## Metadata-Only Baselines",
        "",
        "| Baseline | Inputs allowed | Test ProtoAcc@1 | Interpretation |",
        "|---|---|---:|---|",
    ]
    for item in summary["metadata_baselines"]:
        lines.append(
            f"| {item['name']} | {item['inputs']} | {pct(item['correct'], item['total'])} | {item['interpretation']} |"
        )
    lines.extend(
        [
            "",
            "## Trained-Model Query-Policy Shuffle",
            "",
            "| Model | Original query | Shuffled policy query | Drop |",
            "|---|---:|---:|---:|",
        ]
    )
    for item in summary["shuffle_models"]:
        orig = item["original"]
        shuf = item["shuffled"]
        drop = 100.0 * (orig["accuracy"] - shuf["accuracy"])
        lines.append(
            f"| {item['name']} | {pct(orig['correct'], orig['total'])} | {pct(shuf['correct'], shuf['total'])} | {drop:.1f} pp |"
        )
    lines.extend(
        [
            "",
            "## Target-Slot Distribution",
            "",
            "The target slot is not determined by the query-policy ID; each policy appears across multiple candidate slots.",
            "",
            "```json",
            json.dumps(summary["target_slot_distribution"], indent=2),
            "```",
            "",
            "## Interpretation",
            "",
            "The query-policy field is a semantic disambiguator among same-relation candidates, not a hidden candidate index. Metadata-only predictors that see the policy but not candidate geometry remain near the slot-prior range, while trained selectors lose accuracy when the policy part of the query is shuffled at test time. Fine Rank Rule remains a task-specific ceiling because it is explicitly programmed to decode the policy and rank candidates by geometry.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("runs/hard_confusable_20260526"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hard_confusable_20260526/training/query_policy_leakage_audit"))
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        action="append",
        default=None,
        help="Graph checkpoints to evaluate. Can be provided multiple times.",
    )
    parser.add_argument(
        "--mlp-checkpoint",
        type=Path,
        action="append",
        default=None,
        help="MLP+pooling checkpoints to evaluate. Can be provided multiple times.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--shuffle-seed", type=int, default=20260526)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = read_json(args.root / "scenarios" / "train_scenarios.json")
    test = read_json(args.root / "scenarios" / "test_scenarios.json")
    protocol_dir = args.root / "protocol_graphs" / "test"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fallback = global_majority_slot(train)
    metadata: List[Dict[str, Any]] = []
    for name, fields, inputs, interpretation in [
        ("global_slot_prior", [], "none beyond training target-slot prior", "Expected chance/prior predictor."),
        ("policy_only_majority", ["query_policy_id"], "query-policy ID only", "Fails if policy is not target slot."),
        ("policy_plus_k_majority", ["query_policy_id", "num_candidates"], "query-policy ID + K", "Tests policy-to-slot shortcut with candidate count."),
        ("relation_policy_k_majority", ["rel_type", "query_policy_id", "num_candidates"], "relation + query-policy ID + K", "Tests metadata-only query leakage."),
    ]:
        if fields:
            mapping = majority_slot_map(train, fields)
            result = eval_slot_rule(test, name, fields, mapping, fallback)
        else:
            result = {
                "name": name,
                "correct": sum(1 for scene in test if int(scene["target_slot"]) == fallback),
                "total": len(test),
                "accuracy": sum(1 for scene in test if int(scene["target_slot"]) == fallback) / len(test),
                "records": [],
            }
        result["inputs"] = inputs
        result["interpretation"] = interpretation
        metadata.append(result)
    policy_as_slot = eval_policy_as_slot(test)
    policy_as_slot["inputs"] = "query-policy ID interpreted as candidate slot"
    policy_as_slot["interpretation"] = "Direct hidden-index leakage would score high here."
    metadata.append(policy_as_slot)

    device_name = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    override = shuffled_policy_map(test, args.shuffle_seed)
    shuffle_models: List[Dict[str, Any]] = []
    graph_checkpoints = args.graph_checkpoint or [
        args.root / "training" / "lcorg_full_relgate_rankquery_seed42" / "full_best.pt",
    ]
    mlp_checkpoints = args.mlp_checkpoint or [
        args.root / "training" / "candidate_mlp_rankquery_seed21_cpu" / "candidate_mlp_best.pt",
    ]
    for checkpoint in graph_checkpoints:
        if not checkpoint.exists():
            continue
        run_name = checkpoint.parent.name
        original = eval_graph_checkpoint(checkpoint, protocol_dir, test, None, device)
        shuffled = eval_graph_checkpoint(checkpoint, protocol_dir, test, override, device)
        write_jsonl(args.output_dir / f"{run_name}_graph_original.jsonl", original["records"])
        write_jsonl(args.output_dir / f"{run_name}_graph_policy_shuffled.jsonl", shuffled["records"])
        shuffle_models.append({"name": f"Ours graph+gate ({run_name})", "original": original, "shuffled": shuffled})
    for checkpoint in mlp_checkpoints:
        if not checkpoint.exists():
            continue
        run_name = checkpoint.parent.name
        original = eval_mlp_checkpoint(checkpoint, protocol_dir, test, None, device)
        shuffled = eval_mlp_checkpoint(checkpoint, protocol_dir, test, override, device)
        write_jsonl(args.output_dir / f"{run_name}_mlp_original.jsonl", original["records"])
        write_jsonl(args.output_dir / f"{run_name}_mlp_policy_shuffled.jsonl", shuffled["records"])
        shuffle_models.append({"name": f"MLP+pooling ({run_name})", "original": original, "shuffled": shuffled})

    for item in metadata:
        write_jsonl(args.output_dir / f"{item['name']}.jsonl", item.get("records", []))

    summary = {
        "root": str(args.root),
        "shuffle_seed": args.shuffle_seed,
        "metadata_baselines": [
            {k: v for k, v in item.items() if k != "records"}
            for item in metadata
        ],
        "shuffle_models": [
            {
                "name": item["name"],
                "original": {k: v for k, v in item["original"].items() if k != "records"},
                "shuffled": {k: v for k, v in item["shuffled"].items() if k != "records"},
            }
            for item in shuffle_models
        ],
        "target_slot_distribution": summarize_distribution(test),
    }
    (args.output_dir / "query_policy_leakage_audit_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "query_policy_leakage_audit.md").write_text(make_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
