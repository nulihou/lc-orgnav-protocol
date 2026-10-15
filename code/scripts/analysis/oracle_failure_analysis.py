import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 让脚本在“直接python运行”时也能稳定导入同目录工具模块
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analysis.unified_episode_evaluator import evaluate_episode  # type: ignore


@dataclass
class FailureTag:
    tag: str
    detail: str = ""


def _safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


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


def _extract_trajectory_dist_phase(traj: Any) -> Tuple[List[float], List[str], List[float]]:
    """
    Returns:
      - all_dists: 轨迹中所有dist
      - all_phases: 轨迹中所有phase（缺失用空串）
      - all_speeds: 轨迹中所有speed（缺失用nan）
    """
    if not isinstance(traj, list):
        return [], [], []
    dists: List[float] = []
    phases: List[str] = []
    speeds: List[float] = []
    for p in traj:
        if not isinstance(p, dict):
            continue
        if "dist" in p:
            dists.append(_safe_float(p.get("dist")))
        else:
            # 保持长度一致性（极少数case），这里不强制
            pass
        phases.append(str(p.get("phase", "")))
        speeds.append(_safe_float(p.get("speed")))
    return dists, phases, speeds


def _extract_g3_alignment_errors(traj: Any) -> Tuple[List[float], List[float]]:
    """
    从trajectory中提取G3(DOCK)阶段的对齐误差序列（如果日志包含 e_y / e_psi_deg 字段）。

    Returns:
      - abs_e_y: List[float]，|e_y|（m）
      - abs_e_psi_deg: List[float]，|e_psi_deg|（deg）
    """
    if not isinstance(traj, list):
        return [], []
    abs_e_y: List[float] = []
    abs_e_psi: List[float] = []
    for p in traj:
        if not isinstance(p, dict):
            continue
        # Dock控制可能发生在两种形式：
        # 1) phase=="G3"（显式G3阶段）
        # 2) phase=="G2" 但 dock_mode==True（G2内软切到DOCK控制）
        in_g3 = (str(p.get("phase", "")) == "G3")
        in_dock = bool(p.get("dock_mode", False))
        if not (in_g3 or in_dock):
            continue
        if p.get("e_y", None) is not None:
            abs_e_y.append(abs(_safe_float(p.get("e_y"))))
        if p.get("e_psi_deg", None) is not None:
            abs_e_psi.append(abs(_safe_float(p.get("e_psi_deg"))))
    return abs_e_y, abs_e_psi


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs2 = sorted([x for x in xs if math.isfinite(x)])
    if not xs2:
        return float("nan")
    q = max(0.0, min(100.0, float(q)))
    k = (len(xs2) - 1) * (q / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(xs2[int(k)])
    d0 = xs2[int(f)] * (c - k)
    d1 = xs2[int(c)] * (k - f)
    return float(d0 + d1)


def _last_phase_window(traj: List[Dict[str, Any]], phase: str, max_points: int = 80) -> List[float]:
    if not isinstance(traj, list) or not traj:
        return []
    buf: List[float] = []
    for p in reversed(traj):
        if not isinstance(p, dict):
            continue
        if str(p.get("phase", "")) != phase:
            continue
        if "dist" not in p:
            continue
        buf.append(_safe_float(p.get("dist")))
        if len(buf) >= max_points:
            break
    buf.reverse()
    return buf


def _count_derivative_sign_changes(xs: List[float]) -> int:
    if len(xs) < 3:
        return 0
    diffs = [xs[i + 1] - xs[i] for i in range(len(xs) - 1)]
    signs = []
    for d in diffs:
        if abs(d) < 1e-3:
            continue
        signs.append(1 if d > 0 else -1)
    if len(signs) < 3:
        return 0
    changes = 0
    for i in range(1, len(signs)):
        if signs[i] != signs[i - 1]:
            changes += 1
    return changes


def tag_failure_reason(ep: Dict[str, Any], max_steps_total: int = 1000) -> Optional[FailureTag]:
    """
    只对“8m未成功”的episode打失败原因标签；成功则返回None。

    标签集合（尽量贴合你给审稿人的那套话术）：
    - collision
    - timeout
    - stuck
    - not_converged_distance
    - oscillation_close_range
    - overshoot_then_brake
    - lane_misalignment (兜底：近距离但无法收敛到阈值)
    - unknown
    """
    success_8m = bool(ep.get("success_8m", False))
    if success_8m:
        return None

    # 1) 显式字段（如果未来日志里加了collision/timeout）
    if bool(ep.get("collision", False)):
        return FailureTag("collision", "collision==True")
    if bool(ep.get("timeout", False)):
        return FailureTag("timeout", "timeout==True")
    if bool(ep.get("stuck", False)):
        return FailureTag("stuck", "stuck==True")

    steps = int(ep.get("steps", -1)) if ep.get("steps") is not None else -1
    if steps >= max_steps_total:
        return FailureTag("timeout", f"steps={steps} >= max_steps_total={max_steps_total}")

    traj = ep.get("trajectory", [])
    dists, phases, _speeds = _extract_trajectory_dist_phase(traj)

    # 2) 距离特征（优先使用轨迹的min/final，更可信）
    if dists:
        min_dist = min(dists)
        final_dist = dists[-1]
    else:
        min_dist = _safe_float(ep.get("min_dist", float("nan")))
        final_dist = _safe_float(ep.get("final_dist", float("nan")))

    entered_g3 = "G3" in phases if phases else False

    # 3) overshoot：曾经进到4m内，但最终没能停住在4m内
    # 注意：controller的success是按final_dist判定，因此“min<4但final>=4”就是典型overshoot
    if math.isfinite(min_dist) and math.isfinite(final_dist):
        if min_dist < 4.0 and final_dist >= 4.0:
            return FailureTag("overshoot_then_brake", f"min_dist={min_dist:.2f} < 4m but final_dist={final_dist:.2f} >= 4m")

    # 4) oscillation：在近距离（<8m）持续来回抖动
    # 用G3尾段（或最后一段G3）距离序列的振幅+导数符号翻转数做一个稳健判别
    g3_tail = _last_phase_window(traj, "G3", max_points=80)
    if g3_tail:
        amp = max(g3_tail) - min(g3_tail)
        sign_changes = _count_derivative_sign_changes(g3_tail)
        if min(g3_tail) < 8.0 and amp > 1.5 and sign_changes >= 6:
            return FailureTag("oscillation_close_range", f"g3_tail_amp={amp:.2f}, sign_changes={sign_changes}, g3_tail_min={min(g3_tail):.2f}")

    # 5) not_converged：整体没能靠近到8m
    if math.isfinite(min_dist) and min_dist >= 8.0:
        detail = "min_dist>=8m"
        if not entered_g3:
            detail = "never_entered_G3; " + detail
        else:
            detail = "entered_G3; " + detail
        return FailureTag("not_converged_distance", detail)

    # 6) lane_misalignment：进入近距离(<8m)但始终无法完成8m/4m停靠（兜底）
    if math.isfinite(min_dist) and min_dist < 8.0:
        # 如果最终距离又明显回退，通常是贴不住/对不齐导致的“近距离失败”
        if math.isfinite(final_dist) and final_dist - min_dist > 2.0:
            return FailureTag("lane_misalignment", f"min_dist={min_dist:.2f}<8m but final_dist={final_dist:.2f} (regressed)")
        return FailureTag("lane_misalignment", f"min_dist={min_dist:.2f}<8m but success_8m==False")

    return FailureTag("unknown", "insufficient signals")


def _write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def _write_md(path: Path, summary_lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines).rstrip() + "\n")


def _try_plot_bar(path: Path, counts: Dict[str, int], title: str) -> bool:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return False

    labels = list(counts.keys())
    values = [counts[k] for k in labels]
    plt.figure(figsize=(10, 4.8))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylabel("Count")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-jsonl", type=str, required=True, help="Oracle JSONL log path")
    ap.add_argument("--max-steps-total", type=int, default=1000, help="Max steps used in run_episode(max_steps_total)")
    ap.add_argument("--output-csv", type=str, default="results/proto_main_eval/oracle_failure_analysis/oracle_failure_tags.csv")
    ap.add_argument("--output-md", type=str, default="results/proto_main_eval/oracle_failure_analysis/oracle_failure_summary.md")
    ap.add_argument("--output-plot", type=str, default="results/proto_main_eval/oracle_failure_analysis/oracle_failure_reasons.png")
    args = ap.parse_args()

    in_path = Path(args.input_jsonl)
    episodes = _load_jsonl(in_path)
    total = len(episodes)

    # 统计：8m失败原因 + 4m失败(在8m成功条件下)原因
    rows: List[Dict[str, Any]] = []
    counts_8m: Counter[str] = Counter()
    counts_4m_given_8m: Counter[str] = Counter()
    # 仅用于“8m成功但4m失败”的lane_misalignment量化证据（如果日志里有e_y/e_psi）
    align_abs_ey_p95: List[float] = []
    align_abs_epsideg_p95: List[float] = []

    for ep in episodes:
        tag = tag_failure_reason(ep, max_steps_total=int(args.max_steps_total))
        # 统一口径：使用 evaluate_episode() 的 min-dist 判定（避免success_8m/success_4m使用final_dist导致口径不一致）
        ev = evaluate_episode(ep)
        success_8m = bool(ev.get("navsucc8", False))
        success_4m = bool(ev.get("navsucc4", False))
        ey_p95 = float("nan")
        epsi_p95 = float("nan")

        if tag is not None:
            counts_8m[tag.tag] += 1

        # 针对“8m过了但4m没过”的精细失败分析（对控制改进更敏感）
        if success_8m and (not success_4m):
            # 这里复用tag逻辑，但用“4m未成功”的判断：把success_8m当作True绕过
            # 简化实现：用 min_dist / final_dist 识别 overshoot / oscillation / lane_misalignment
            traj = ep.get("trajectory", [])
            dists, phases, _ = _extract_trajectory_dist_phase(traj)
            if dists:
                min_dist = min(dists)
                final_dist = dists[-1]
            else:
                min_dist = _safe_float(ep.get("min_dist", float("nan")))
                final_dist = _safe_float(ep.get("final_dist", float("nan")))
            if math.isfinite(min_dist) and math.isfinite(final_dist) and min_dist < 4.0 and final_dist >= 4.0:
                counts_4m_given_8m["overshoot_then_brake"] += 1
            else:
                g3_tail = _last_phase_window(traj, "G3", max_points=80)
                if g3_tail:
                    amp = max(g3_tail) - min(g3_tail)
                    sign_changes = _count_derivative_sign_changes(g3_tail)
                    if min(g3_tail) < 6.0 and amp > 1.0 and sign_changes >= 6:
                        counts_4m_given_8m["oscillation_close_range"] += 1
                    else:
                        counts_4m_given_8m["lane_misalignment"] += 1
                else:
                    counts_4m_given_8m["lane_misalignment"] += 1

            # 如果有对齐误差字段，把“对不齐”变成可量化对象（用于阈值建议/论文堵嘴）
            abs_ey, abs_epsideg = _extract_g3_alignment_errors(traj)
            ey_p95 = _percentile(abs_ey, 95.0)
            epsi_p95 = _percentile(abs_epsideg, 95.0)
            if math.isfinite(ey_p95):
                align_abs_ey_p95.append(float(ey_p95))
            if math.isfinite(epsi_p95):
                align_abs_epsideg_p95.append(float(epsi_p95))

        rows.append({
            "episode": ep.get("episode"),
            "scene_id": ep.get("scene_id", ep.get("scen_id")),
            "run_id": ep.get("run_id"),
            "seed_index": ep.get("seed_index"),
            "success_8m": success_8m,
            "success_4m": success_4m,
            "stuck": bool(ep.get("stuck", False)),
            "steps": ep.get("steps"),
            "min_dist": ep.get("min_dist"),
            "final_dist": ep.get("final_dist"),
            "failure_tag_8m": tag.tag if tag is not None else "",
            "failure_detail_8m": tag.detail if tag is not None else "",
            # 新增：对齐误差统计（如果日志里有 e_y/e_psi_deg；否则为空）
            "g3_abs_e_y_p95": f"{ey_p95:.3f}" if (success_8m and (not success_4m) and math.isfinite(ey_p95)) else "",
            "g3_abs_e_psi_deg_p95": f"{epsi_p95:.2f}" if (success_8m and (not success_4m) and math.isfinite(epsi_p95)) else "",
        })

    out_csv = Path(args.output_csv)
    out_md = Path(args.output_md)
    out_plot = Path(args.output_plot)

    fieldnames = [
        "episode", "scene_id", "run_id", "seed_index",
        "success_8m", "success_4m", "stuck", "steps", "min_dist", "final_dist",
        "failure_tag_8m", "failure_detail_8m",
        "g3_abs_e_y_p95", "g3_abs_e_psi_deg_p95",
    ]
    _write_csv(out_csv, rows, fieldnames)

    fail8 = sum(counts_8m.values())
    succ8 = total - fail8
    fail4 = sum(1 for r in rows if (r.get("success_4m") is False))
    succ4 = total - fail4
    fail4_given_8m = sum(1 for r in rows if (r.get("success_8m") is True and r.get("success_4m") is False))

    # markdown summary
    lines: List[str] = []
    lines.append("## Oracle 失败原因分析（Step 3.1）")
    lines.append("")
    lines.append(f"- 输入日志：`{in_path.as_posix()}`")
    lines.append(f"- 总episodes：**{total}**")
    lines.append(f"- NavSucc@8m：**{succ8}/{total}**，失败 **{fail8}**")
    lines.append(f"- NavSucc@4m：**{succ4}/{total}**，失败 **{fail4}**")
    lines.append(f"- 条件失败（8m成功但4m失败）：**{fail4_given_8m}**")
    lines.append("")

    lines.append("### 8m失败原因统计（success_8m==False）")
    lines.append("")
    lines.append("| Failure Tag | Count | Percent |")
    lines.append("|---|---:|---:|")
    for k, c in counts_8m.most_common():
        pct = (100.0 * c / max(1, fail8))
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    if fail8 == 0:
        lines.append("| (none) | 0 | 0.0% |")
    lines.append("")

    lines.append("### 4m失败原因统计（success_8m==True && success_4m==False）")
    lines.append("")
    lines.append("| Failure Tag | Count | Percent |")
    lines.append("|---|---:|---:|")
    denom = max(1, sum(counts_4m_given_8m.values()))
    for k, c in counts_4m_given_8m.most_common():
        pct = (100.0 * c / denom)
        lines.append(f"| {k} | {c} | {pct:.1f}% |")
    if sum(counts_4m_given_8m.values()) == 0:
        lines.append("| (none) | 0 | 0.0% |")
    lines.append("")

    # 量化“lane_misalignment”的证据：|e_y|与|e_psi|分布（只有在新日志包含这些字段时才会出现）
    if align_abs_ey_p95 or align_abs_epsideg_p95:
        lines.append("### lane_misalignment 量化（若日志包含 e_y/e_psi_deg）")
        lines.append("")
        if align_abs_ey_p95:
            lines.append(f"- |e_y| P95（每episode，在G3阶段）：n={len(align_abs_ey_p95)}, "
                         f"median={_percentile(align_abs_ey_p95, 50):.3f}m, "
                         f"p75={_percentile(align_abs_ey_p95, 75):.3f}m, "
                         f"p90={_percentile(align_abs_ey_p95, 90):.3f}m, "
                         f"p95={_percentile(align_abs_ey_p95, 95):.3f}m")
            # 简单分箱，帮助你快速定 y_thresh
            bins = [0.3, 0.5, 0.7]
            cnt = [0, 0, 0, 0]
            for v in align_abs_ey_p95:
                if v <= bins[0]:
                    cnt[0] += 1
                elif v <= bins[1]:
                    cnt[1] += 1
                elif v <= bins[2]:
                    cnt[2] += 1
                else:
                    cnt[3] += 1
            lines.append(f"- |e_y| P95 分箱（<=0.3 / <=0.5 / <=0.7 / >0.7 m）：{cnt[0]} / {cnt[1]} / {cnt[2]} / {cnt[3]}")
        if align_abs_epsideg_p95:
            lines.append(f"- |e_psi| P95（每episode，在G3阶段）：n={len(align_abs_epsideg_p95)}, "
                         f"median={_percentile(align_abs_epsideg_p95, 50):.2f}deg, "
                         f"p75={_percentile(align_abs_epsideg_p95, 75):.2f}deg, "
                         f"p90={_percentile(align_abs_epsideg_p95, 90):.2f}deg, "
                         f"p95={_percentile(align_abs_epsideg_p95, 95):.2f}deg")
        lines.append("")

    lines.append("### 输出文件")
    lines.append("")
    lines.append(f"- 明细CSV：`{out_csv.as_posix()}`")
    lines.append(f"- 汇总Markdown：`{out_md.as_posix()}`")
    lines.append(f"- 柱状图（若环境有matplotlib）：`{out_plot.as_posix()}`")

    _write_md(out_md, lines)

    plotted = _try_plot_bar(out_plot, dict(counts_8m), title="Oracle failure reasons (NavSucc@8m failed)")
    if not plotted:
        # 不强制依赖matplotlib，保持脚本可落地
        pass

    print(f"[OK] wrote csv: {out_csv}")
    print(f"[OK] wrote md : {out_md}")
    if plotted:
        print(f"[OK] wrote png: {out_plot}")
    else:
        print("[Info] matplotlib not available; skip png plot")


if __name__ == "__main__":
    main()


