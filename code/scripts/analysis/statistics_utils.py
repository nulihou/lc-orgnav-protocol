#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统计工具函数（Step 2.3）

实现均值+95% CI（Wilson区间或bootstrap）
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any


def _norm_ppf(p: float) -> float:
    """
    标准正态分布的分位数函数（inverse CDF）。
    使用 Peter John Acklam 的有理逼近，避免依赖scipy。
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0,1), got {p}")

    # Coefficients in rational approximations
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]

    # Define break-points.
    plow = 0.02425
    phigh = 1 - plow

    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)

    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def wilson_confidence_interval(
    successes: int,
    total: int,
    confidence: float = 0.95
) -> Tuple[float, float]:
    """
    计算Wilson置信区间
    
    Args:
        successes: 成功次数
        total: 总次数
        confidence: 置信水平（默认0.95）
    
    Returns:
        (lower_bound, upper_bound)
    """
    if total == 0:
        return (0.0, 0.0)
    
    z = _norm_ppf((1 + confidence) / 2)  # 对于95% CI，z ≈ 1.96
    p_hat = successes / total
    
    denominator = 1 + (z**2 / total)
    center = (p_hat + (z**2 / (2 * total))) / denominator
    margin = (z / denominator) * np.sqrt((p_hat * (1 - p_hat) / total) + (z**2 / (4 * total**2)))
    
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    
    return (lower, upper)


def bootstrap_confidence_interval(
    data: List[bool],
    confidence: float = 0.95,
    n_bootstrap: int = 10000
) -> Tuple[float, float]:
    """
    使用Bootstrap方法计算置信区间
    
    Args:
        data: 布尔值列表（True表示成功，False表示失败）
        confidence: 置信水平（默认0.95）
        n_bootstrap: Bootstrap采样次数
    
    Returns:
        (lower_bound, upper_bound)
    """
    if not data:
        return (0.0, 0.0)
    
    data_array = np.array(data, dtype=float)
    n = len(data_array)
    
    # Bootstrap采样
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data_array, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    # 计算分位数
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return (lower, upper)


def calculate_statistics_with_ci(
    data: List[bool],
    method: str = "wilson"
) -> Dict[str, Any]:
    """
    计算统计量（均值+95% CI）
    
    Args:
        data: 布尔值列表
        method: 方法（"wilson" 或 "bootstrap"）
    
    Returns:
        统计字典：
            - mean: 均值
            - success_count: 成功次数
            - total: 总次数
            - ci_lower: 置信区间下界
            - ci_upper: 置信区间上界
            - ci_string: 格式化字符串（如 "50.0 (CI: 42.1–57.9)"）
    """
    if not data:
        return {
            "mean": 0.0,
            "success_count": 0,
            "total": 0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "ci_string": "0.0 (95% CI: 0.0–0.0)"
        }
    
    success_count = sum(1 for x in data if x)
    total = len(data)
    mean = success_count / total if total > 0 else 0.0
    
    # 计算置信区间
    if method == "wilson":
        ci_lower, ci_upper = wilson_confidence_interval(success_count, total)
    elif method == "bootstrap":
        ci_lower, ci_upper = bootstrap_confidence_interval(data)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # 格式化字符串（T-ITS期刊标准格式：明确标注95% CI）
    ci_string = f"{mean*100:.1f} (95% CI: {ci_lower*100:.1f}–{ci_upper*100:.1f})"
    
    return {
        "mean": mean,
        "mean_percent": mean * 100,
        "success_count": success_count,
        "total": total,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_lower_percent": ci_lower * 100,
        "ci_upper_percent": ci_upper * 100,
        "ci_string": ci_string,
        "method": method
    }


def calculate_metrics_with_ci(
    results: List[Dict[str, Any]],
    metric_key: str,
    method: str = "wilson"
) -> Dict[str, Any]:
    """
    从结果列表中计算某个指标的统计量（带CI）
    
    Args:
        results: 结果字典列表
        metric_key: 指标键名（如 "semnavsucc8", "navsucc8"）
        method: 统计方法
    
    Returns:
        统计字典
    """
    data = [r.get(metric_key, False) for r in results]
    return calculate_statistics_with_ci(data, method)


def format_statistics_table(
    results_dict: Dict[str, List[Dict[str, Any]]],
    metrics: List[str],
    method: str = "wilson"
) -> str:
    """
    格式化统计表（Markdown格式）
    
    Args:
        results_dict: {mode: results_list}
        metrics: 指标列表（如 ["semnavsucc8", "navsucc8"]）
        method: 统计方法
    
    Returns:
        Markdown表格字符串
    """
    lines = []
    
    # 表头
    header = ["Method"] + [m.replace("_", " ").title() for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    
    # 每行数据
    for mode, results in results_dict.items():
        row = [mode.upper()]
        for metric in metrics:
            stats_dict = calculate_metrics_with_ci(results, metric, method)
            row.append(stats_dict["ci_string"])
        lines.append("| " + " | ".join(row) + " |")
    
    return "\n".join(lines)


if __name__ == "__main__":
    # 测试
    test_data = [True] * 75 + [False] * 25  # 75%成功率
    stats = calculate_statistics_with_ci(test_data, method="wilson")
    print(f"Mean: {stats['mean']:.3f}")
    print(f"CI: {stats['ci_string']}")

