# 文件名: scripts/common/visualize_results.py
# 功能: 读取对比结果，生成延迟对比图、吞吐量对比图和性能差异热图
# 用法: python visualize_results.py
# 依赖: matplotlib、seaborn、numpy

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Optional

# 设置中文显示与绘图风格
plt.rcParams["font.sans-serif"] = [
    "Noto Sans CJK SC", "SimHei", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False
sns.set_style("whitegrid")

FIGURES_DIR = Path.home() / "llm-benchmark" / "results" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

COMPARISON_DIR = (
    Path.home() / "llm-benchmark" / "results" / "comparison"
)

# 场景显示名称映射
SCENARIO_LABELS = {
    "short_short": "短入短出",
    "short_long":  "短入长出",
    "long_short":  "长入短出",
    "long_long":   "长入长出",
    "single_turn": "单轮对话",
    "multi_turn":  "多轮对话",
}


def load_summary() -> Optional[Dict]:
    """加载 compare_results.py 生成的 summary.json。"""
    f = COMPARISON_DIR / "summary.json"
    if not f.exists():
        print("未找到 summary.json，请先运行 compare_results.py")
        return None
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def _extract_series(
    summary: Dict, metric_key: str
) -> tuple:
    """
    从 summary 字典中提取指定指标的双框架数值序列。
    返回 (scenarios, labels, vllm_values, sglang_values)。
    """
    scenarios, labels, v_vals, s_vals = [], [], [], []
    for scenario, data in summary.items():
        m = data.get("metrics", {}).get(metric_key)
        if m:
            scenarios.append(scenario)
            labels.append(SCENARIO_LABELS.get(scenario, scenario))
            v_vals.append(m["vllm"])
            s_vals.append(m["sglang"])
    return scenarios, labels, v_vals, s_vals


def _add_improvement_labels(
    ax, x, v_vals, s_vals, lower_is_better: bool
) -> None:
    """在柱状图上方标注 SGLang 相对改进百分比。"""
    for i, (v, s) in enumerate(zip(v_vals, s_vals)):
        if v > 0:
            if lower_is_better:
                imp = (v - s) / v * 100.0
            else:
                imp = (s - v) / v * 100.0
            color = "#27ae60" if imp > 0 else "#e74c3c"
            ax.annotate(
                f"{imp:+.1f}%",
                xy=(x[i], max(v, s) * 1.08),
                ha="center", va="bottom",
                fontsize=8, fontweight="bold", color=color,
            )


def plot_latency_comparison(summary: Dict) -> None:
    """
    生成 TTFT P50 和总延迟 P50 的双子图对比柱状图。

    TTFT 改进远大于总延迟改进 → 改进主要来自预填充阶段
    两者改进幅度接近 → 改进贯穿整个推理流程
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        (axes[0], "ttft_p50_ms",    "TTFT P50 对比",   "延迟 (ms)"),
        (axes[1], "latency_p50_ms", "总延迟 P50 对比", "延迟 (ms)"),
    ]

    for ax, key, title, ylabel in configs:
        _, labels, v_vals, s_vals = _extract_series(summary, key)
        x = np.arange(len(labels))
        width = 0.35

        ax.bar(x - width / 2, v_vals, width,
               label="vLLM", alpha=0.85, color="#3498db")
        ax.bar(x + width / 2, s_vals, width,
               label="SGLang", alpha=0.85, color="#e74c3c")

        _add_improvement_labels(ax, x, v_vals, s_vals, True)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "latency_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 延迟对比图: {out}")


def plot_throughput_comparison(summary: Dict) -> None:
    """生成平均吞吐量对比柱状图。"""
    _, labels, v_vals, s_vals = _extract_series(
        summary, "throughput_mean_tps"
    )
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, v_vals, width,
           label="vLLM", alpha=0.85, color="#3498db")
    ax.bar(x + width / 2, s_vals, width,
           label="SGLang", alpha=0.85, color="#e74c3c")

    _add_improvement_labels(ax, x, v_vals, s_vals, False)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_title("平均吞吐量对比（tokens/s）",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("吞吐量 (tokens/s)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "throughput_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 吞吐量对比图: {out}")


def plot_performance_heatmap(summary: Dict) -> None:
    """
    生成 SGLang 相对 vLLM 改进幅度的热图。

    颜色编码：绿色 = SGLang 更优，红色 = vLLM 更优
    行 = 指标，列 = 场景，单元格数值 = 改进百分比
    """
    metric_keys = [
        ("ttft_p50_ms",         "TTFT P50",   True),
        ("ttft_p95_ms",         "TTFT P95",   True),
        ("latency_p50_ms",      "延迟 P50",   True),
        ("latency_p95_ms",      "延迟 P95",   True),
        ("throughput_mean_tps", "吞吐量",     False),
    ]

    scenarios = list(summary.keys())
    col_labels = [
        SCENARIO_LABELS.get(s, s) for s in scenarios
    ]

    matrix = []
    for key, _, lower_is_better in metric_keys:
        row = []
        for scenario in scenarios:
            m = summary.get(scenario, {}).get("metrics", {}).get(key)
            if m and m["vllm"] > 0:
                if lower_is_better:
                    diff = (m["vllm"] - m["sglang"]) / m["vllm"] * 100
                else:
                    diff = (m["sglang"] - m["vllm"]) / m["vllm"] * 100
            else:
                diff = 0.0
            row.append(diff)
        matrix.append(row)

    mat = np.array(matrix)
    row_labels = [t for _, t, _ in metric_keys]

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        mat, cmap="RdYlGn", vmin=-30, vmax=30, aspect="auto"
    )

    ax.set_xticks(range(len(col_labels)))
    ax.set_yticks(range(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticklabels(row_labels)

    # 在每个单元格中心标注数值
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = mat[i, j]
            text_color = "black" if abs(val) < 15 else "white"
            ax.text(
                j, i, f"{val:+.1f}%",
                ha="center", va="center",
                fontsize=9, color=text_color,
            )

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label(
        "SGLang 相对 vLLM 改进幅度（%）",
        rotation=270, labelpad=18,
    )
    ax.set_title(
        "性能差异热图（正值 = SGLang 更优）",
        fontsize=13, fontweight="bold", pad=15,
    )

    plt.tight_layout()
    out = FIGURES_DIR / "performance_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✓ 性能热图: {out}")


def main():
    summary = load_summary()
    if summary is None:
        return

    print("生成可视化图表...")
    plot_latency_comparison(summary)
    plot_throughput_comparison(summary)
    plot_performance_heatmap(summary)
    print(f"\n✓ 全部图表已写入 {FIGURES_DIR}")


if __name__ == "__main__":
    main()
