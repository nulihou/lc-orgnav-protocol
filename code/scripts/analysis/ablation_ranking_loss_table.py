#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 5: Ranking loss 在线稳定性消融（带95% CI）

目标：输出一张短文友好的小表：
  - Proto GS@1（选择正确率）
  - SemNavSucc@8m / @4m（严格绑定GT，使用统一evaluate_episode口径）+ 95% CI（Wilson）

用法：
  python scripts/analysis/ablation_ranking_loss_table.py \
    --full logs_v2/proto_main_eval/gs_proto_driven.jsonl \
    --no-rank logs_v2/proto_main_eval/ablation/no_rank_gs_30ep.jsonl \
    --out-md results/proto_main_eval/ablation/ranking_loss_ablation_table.md \
    --out-tex results/proto_main_eval/ablation/ranking_loss_ablation_table.tex
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 让脚本在“直接python运行”时也能稳定导入同目录工具模块
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.statistics_utils import calculate_statistics_with_ci  # type: ignore
from scripts.analysis.unified_episode_evaluator import evaluate_episode  # type: ignore


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e
    return items


def _extract_flags(eps: List[Dict[str, Any]]) -> Tuple[List[bool], List[bool], List[bool]]:
    """
    Returns:
      - gs_at_1: 选择是否正确
      - semnavsucc8: is_correct AND d_min_to_gt < 8m（统一口径）
      - semnavsucc4: is_correct AND d_min_to_gt < 4m（统一口径）
    """
    gs_at_1: List[bool] = []
    sem8: List[bool] = []
    sem4: List[bool] = []
    for ep in eps:
        # gs_correct字段在不同日志中可能叫 gs_correct / gs_at_1
        gs_at_1.append(bool(ep.get("gs_correct", False) or ep.get("gs_at_1", False)))
        ev = evaluate_episode(ep)
        sem8.append(bool(ev.get("semnavsucc8", False)))
        sem4.append(bool(ev.get("semnavsucc4", False)))
    return gs_at_1, sem8, sem4


def _fmt_ci(stats: Dict[str, Any]) -> str:
    # 统一使用 “xx.x (CI: aa.a–bb.b)” 的短文格式
    return stats["ci_string"]


def _make_md_row(name: str, gs: Dict[str, Any], s8: Dict[str, Any], s4: Dict[str, Any]) -> str:
    return f"| {name} | {_fmt_ci(gs)} | {_fmt_ci(s8)} | {_fmt_ci(s4)} |"


def _make_tex_row(name: str, gs: Dict[str, Any], s8: Dict[str, Any], s4: Dict[str, Any]) -> str:
    def _tex_ci(st: Dict[str, Any]) -> str:
        return f"{st['mean_percent']:.1f}\\% ({st['ci_lower_percent']:.1f}--{st['ci_upper_percent']:.1f})"
    return f"{name} & {_tex_ci(gs)} & {_tex_ci(s8)} & {_tex_ci(s4)} \\\\"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", type=str, required=True, help="Full(含ranking loss) 在线jsonl")
    ap.add_argument("--no-rank", type=str, required=True, help="w/o ranking loss 在线jsonl")
    ap.add_argument("--out-md", type=str, default="results/proto_main_eval/ablation/ranking_loss_ablation_table.md")
    ap.add_argument("--out-tex", type=str, default="results/proto_main_eval/ablation/ranking_loss_ablation_table.tex")
    ap.add_argument("--ci-method", type=str, default="wilson", choices=["wilson", "bootstrap"])
    args = ap.parse_args()

    full_path = Path(args.full)
    no_rank_path = Path(args.no_rank)
    out_md = Path(args.out_md)
    out_tex = Path(args.out_tex)

    full_eps = _load_jsonl(full_path)
    no_rank_eps = _load_jsonl(no_rank_path)

    f_gs, f_s8, f_s4 = _extract_flags(full_eps)
    n_gs, n_s8, n_s4 = _extract_flags(no_rank_eps)

    f_gs_s = calculate_statistics_with_ci(f_gs, method=args.ci_method)
    f_s8_s = calculate_statistics_with_ci(f_s8, method=args.ci_method)
    f_s4_s = calculate_statistics_with_ci(f_s4, method=args.ci_method)

    n_gs_s = calculate_statistics_with_ci(n_gs, method=args.ci_method)
    n_s8_s = calculate_statistics_with_ci(n_s8, method=args.ci_method)
    n_s4_s = calculate_statistics_with_ci(n_s4, method=args.ci_method)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    md_lines = [
        "## Ranking loss 在线消融（Step 5）",
        "",
        f"- Full: `{full_path.as_posix()}` (n={len(full_eps)})",
        f"- w/o ranking loss: `{no_rank_path.as_posix()}` (n={len(no_rank_eps)})",
        f"- CI method: `{args.ci_method}`",
        "",
        "| Variant | Proto GS@1 | SemNavSucc@8m | SemNavSucc@4m |",
        "|---|---:|---:|---:|",
        _make_md_row("Full (with ranking loss)", f_gs_s, f_s8_s, f_s4_s),
        _make_md_row("w/o ranking loss", n_gs_s, n_s8_s, n_s4_s),
        "",
        "> 注：SemNavSucc 严格绑定 GT（is_correct AND d_min_to_gt < thr），口径与 `unified_episode_evaluator.evaluate_episode()` 一致。",
        "",
    ]
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    tex_lines = [
        "% Auto-generated by scripts/analysis/ablation_ranking_loss_table.py",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        "Variant & Proto GS@1 & SemNavSucc@8m & SemNavSucc@4m \\\\",
        "\\midrule",
        _make_tex_row("Full (with ranking loss)", f_gs_s, f_s8_s, f_s4_s),
        _make_tex_row("w/o ranking loss", n_gs_s, n_s8_s, n_s4_s),
        "\\bottomrule",
        "\\end{tabular}",
        "",
    ]
    out_tex.write_text("\n".join(tex_lines), encoding="utf-8")

    print(f"[OK] wrote md : {out_md}")
    print(f"[OK] wrote tex: {out_tex}")


if __name__ == "__main__":
    main()


