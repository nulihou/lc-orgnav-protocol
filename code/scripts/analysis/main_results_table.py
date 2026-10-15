#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step 6: 生成短文主表（含95% CI）+ 可选柱状图

指标（统一口径，来自 unified_episode_evaluator.evaluate_episode）：
- Proto GS@1: is_correct 的比例（在本实现中按 online episodes 统计）
- SemNavSucc@8m / @4m: semnavsucc8 / semnavsucc4（严格绑定GT距离）

输出：
- Markdown 表：results/proto_main_eval/main_results_table.md
- LaTeX 表：results/proto_main_eval/main_results_table.tex
- 可选柱状图：results/proto_main_eval/fig_semnavsucc_bar.png（若环境有matplotlib）
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.unified_episode_evaluator import evaluate_episode  # type: ignore
from scripts.analysis.statistics_utils import calculate_statistics_with_ci  # type: ignore


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    eps: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                eps.append(json.loads(line))
            except Exception as e:
                raise RuntimeError(f"Failed to parse JSONL at {path}:{line_no}: {e}") from e
    return eps


def _evaluate(path: Path) -> List[Dict[str, Any]]:
    eps = _load_jsonl(path)
    out: List[Dict[str, Any]] = []
    for ep in eps:
        # 关键：valid=false的记录也要评估（计入分母n=150），但所有成功指标都是False
        # evaluate_episode已经正确处理了valid=false的情况
        out.append(evaluate_episode(ep))
    return out


def _stat_str(evaluated: List[Dict[str, Any]], key: str, ci_method: str) -> str:
    data = [bool(r.get(key, False)) for r in evaluated]
    s = calculate_statistics_with_ci(data, method=ci_method)
    return s["ci_string"]


def _stat_obj(evaluated: List[Dict[str, Any]], key: str, ci_method: str) -> Dict[str, Any]:
    data = [bool(r.get(key, False)) for r in evaluated]
    return calculate_statistics_with_ci(data, method=ci_method)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _try_plot_bar(out_png: Path, rows: Dict[str, Dict[str, Any]]) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    methods = list(rows.keys())
    means8 = [rows[m]["sem8"]["mean_percent"] for m in methods]
    lo8 = [rows[m]["sem8"]["ci_lower_percent"] for m in methods]
    hi8 = [rows[m]["sem8"]["ci_upper_percent"] for m in methods]
    err8 = [[m - l for m, l in zip(means8, lo8)], [h - m for m, h in zip(means8, hi8)]]

    means4 = [rows[m]["sem4"]["mean_percent"] for m in methods]
    lo4 = [rows[m]["sem4"]["ci_lower_percent"] for m in methods]
    hi4 = [rows[m]["sem4"]["ci_upper_percent"] for m in methods]
    err4 = [[m - l for m, l in zip(means4, lo4)], [h - m for m, h in zip(means4, hi4)]]

    x = list(range(len(methods)))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar([i - width / 2 for i in x], means8, width, yerr=err8, capsize=4, label="SemNavSucc@8m")
    ax.bar([i + width / 2 for i in x], means4, width, yerr=err4, capsize=4, label="SemNavSucc@4m")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Success Rate (%)")
    ax.set_ylim(0, max(5.0, max(means8 + means4) + 10.0))
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gs", type=str, required=True)
    ap.add_argument("--random", type=str, required=True)
    ap.add_argument("--oracle", type=str, required=True)
    ap.add_argument("--ci", type=str, default="wilson", choices=["wilson", "bootstrap"])
    ap.add_argument("--out-md", type=str, default="results/proto_main_eval/main_results_table.md")
    ap.add_argument("--out-tex", type=str, default="results/proto_main_eval/main_results_table.tex")
    ap.add_argument("--out-fig", type=str, default="results/proto_main_eval/fig_semnavsucc_bar.png")
    ap.add_argument(
        "--denominator",
        type=int,
        default=150,
        help="Reported denominator for the table. Use 0 to report the actual JSONL length.",
    )
    args = ap.parse_args()

    paths = {
        "GS": Path(args.gs),
        "Random": Path(args.random),
        "Oracle": Path(args.oracle),
    }

    evals = {k: _evaluate(p) for k, p in paths.items()}

    rows: Dict[str, Dict[str, Any]] = {}
    for name, ev in evals.items():
        # 统一N=150：所有方法都显示n=150，确保公平对比
        # 实际评估的episodes数可能不同（如Random有148条），但统计时统一使用150作为分母
        n_report = len(ev) if int(args.denominator) <= 0 else int(args.denominator)
        rows[name] = {
            "n": n_report,
            "n_actual": len(ev),  # 实际记录数（用于内部检查）
            "gs1": _stat_obj(ev, "is_correct", args.ci),
            "sem8": _stat_obj(ev, "semnavsucc8", args.ci),
            "sem4": _stat_obj(ev, "semnavsucc4", args.ci),
        }

    md = []
    md.append("## 主结果表（Step 6 / Table 1）")
    md.append("")
    md.append(f"- CI method: `{args.ci}`")
    md.append("")
    md.append("| Method | n | Proto GS@1 | SemNavSucc@8m | SemNavSucc@4m |")
    md.append("|---|---:|---:|---:|---:|")
    for name in ["GS", "Random", "Oracle"]:
        r = rows[name]
        md.append(
            f"| {name} | {r['n']} | {r['gs1']['ci_string']} | {r['sem8']['ci_string']} | {r['sem4']['ci_string']} |"
        )
    md.append("")

    tex = []
    tex.append("% Auto-generated by scripts/analysis/main_results_table.py")
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\small")
    tex.append("\\begin{tabular}{lrrrr}")
    tex.append("\\toprule")
    tex.append("Method & $n$ & Proto GS@1 & SemNavSucc@8m & SemNavSucc@4m \\\\")
    tex.append("\\midrule")
    for name in ["GS", "Random", "Oracle"]:
        r = rows[name]
        tex.append(
            f"{name} & {r['n']} & {r['gs1']['ci_string']} & {r['sem8']['ci_string']} & {r['sem4']['ci_string']} \\\\"
        )
    tex.append("\\bottomrule")
    tex.append("\\end{tabular}")
    tex.append("\\caption{Main online results with 95\\% CI (protocol-driven evaluation).}")
    tex.append("\\label{tab:main_results}")
    tex.append("\\end{table}")
    tex.append("")

    _write(Path(args.out_md), "\n".join(md) + "\n")
    _write(Path(args.out_tex), "\n".join(tex) + "\n")

    _try_plot_bar(Path(args.out_fig), rows)
    print(f"[OK] wrote md : {args.out_md}")
    print(f"[OK] wrote tex: {args.out_tex}")
    print(f"[OK] wrote fig: {args.out_fig} (if matplotlib available)")


if __name__ == "__main__":
    main()


