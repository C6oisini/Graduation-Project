"""
读取手工构造的示例 CSV 数据并绘制 2x2 对比图。
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "demo_mechanism_results.csv"
OUTPUT_PATH = ROOT / "asset" / "demo_mechanism_results.png"

COLORS = {
    "vMF": "#2563EB",
    "Gaussian": "#DC2626",
    "Laplace": "#16A34A",
    "NormPreserving": "#F59E0B",
}

LABELS = {
    "vMF": "vMF",
    "Gaussian": "Gaussian",
    "Laplace": "Laplace",
    "NormPreserving": "Norm-Preserving Gaussian",
}


def load_rows(path: Path) -> list[dict[str, float | str]]:
    """加载 CSV 结果。"""
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | str] = {"mechanism": row["mechanism"]}
            for key, value in row.items():
                if key == "mechanism":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def group_by_mechanism(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    """按机制聚合并按 epsilon 排序。"""
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["mechanism"])].append(row)
    for mechanism in grouped:
        grouped[mechanism].sort(key=lambda item: float(item["epsilon"]))
    return grouped


def plot_metric(ax, grouped, mean_key: str, std_key: str, title: str, ylabel: str) -> None:
    """绘制带误差带的折线图。"""
    for mechanism, rows in grouped.items():
        eps = np.array([float(row["epsilon"]) for row in rows])
        mean = np.array([float(row[mean_key]) for row in rows])
        std = np.array([float(row[std_key]) for row in rows])
        color = COLORS[mechanism]

        ax.plot(eps, mean, marker="o", linewidth=2, color=color, label=LABELS[mechanism])
        ax.fill_between(eps, mean - std, mean + std, color=color, alpha=0.12)

    ax.set_xscale("log")
    ax.set_xlabel("隐私预算 ε")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    """主函数。"""
    rows = load_rows(DATA_PATH)
    grouped = group_by_mechanism(rows)

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Embedding 扰动机制示例对比（展示用数据）", fontsize=16, fontweight="bold")

    plot_metric(axes[0, 0], grouped, "angle_mean", "angle_std", "角度偏差", "角度 (°)")
    plot_metric(axes[0, 1], grouped, "cos_sim_mean", "cos_sim_std", "余弦相似度", "Cosine Similarity")
    plot_metric(axes[1, 0], grouped, "norm_ratio_mean", "norm_ratio_std", "范数保持性", "Norm Ratio")
    plot_metric(axes[1, 1], grouped, "l2_dist_mean", "l2_dist_std", "L2 距离", "L2 Distance")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.96))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"已生成图像: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
