"""
按数据集读取 baseline / method 结果并绘图。

输出两类图片：
1. epsilon 曲线图
2. privacy-utility trade-off 图
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    {
        "name": "MS-COCO",
        "baseline_csv": ROOT / "data" / "ms_coco_baseline.csv",
        "results_csv": ROOT / "data" / "ms_coco_results.csv",
        "epsilon_output": ROOT / "asset" / "ms_coco_epsilon_curves.png",
        "tradeoff_output": ROOT / "asset" / "ms_coco_tradeoff.png",
        "metrics": [
            ("CIDEr", "Caption Quality", "CIDEr", True),
            ("CLIPScore", "Semantic Consistency", "CLIP Score", True),
            ("SSIM", "Visual Reconstruction Similarity", "SSIM", False),
            ("AttackSuccessRate", "Inversion Attack Success", "Attack Success Rate", False),
        ],
        "tradeoff_pairs": [
            ("AttackSuccessRate", "CIDEr", "Attack Success Rate", "CIDEr", "Caption Quality vs Inversion Risk"),
            ("SSIM", "CLIPScore", "SSIM", "CLIP Score", "Semantic Consistency vs Reconstruction Similarity"),
        ],
        "utility_metrics": ["CIDEr", "CLIPScore"],
        "privacy_metrics": ["SSIM", "AttackSuccessRate"],
    },
    {
        "name": "VQA-v2",
        "baseline_csv": ROOT / "data" / "vqa_v2_baseline.csv",
        "results_csv": ROOT / "data" / "vqa_v2_results.csv",
        "epsilon_output": ROOT / "asset" / "vqa_v2_epsilon_curves.png",
        "tradeoff_output": ROOT / "asset" / "vqa_v2_tradeoff.png",
        "metrics": [
            ("Accuracy", "Answer Accuracy", "Accuracy (%)", True),
            ("F1", "Answer F1", "F1 Score (%)", True),
            ("SSIM", "Visual Reconstruction Similarity", "SSIM", False),
            ("AttackSuccessRate", "Inversion Attack Success", "Attack Success Rate", False),
        ],
        "tradeoff_pairs": [
            ("AttackSuccessRate", "Accuracy", "Attack Success Rate", "Accuracy (%)", "Answer Accuracy vs Inversion Risk"),
            ("SSIM", "F1", "SSIM", "F1 Score (%)", "Answer F1 vs Reconstruction Similarity"),
        ],
        "utility_metrics": ["Accuracy", "F1"],
        "privacy_metrics": ["SSIM", "AttackSuccessRate"],
    },
]

COLORS = {
    "No-Privacy": "#111827",
    "Pixel-Gaussian": "#DC2626",
    "Embedding-Laplace": "#D97706",
    "vMF-Ours": "#2563EB",
}

MARKERS = {
    "No-Privacy": "o",
    "Pixel-Gaussian": "s",
    "Embedding-Laplace": "^",
    "vMF-Ours": "D",
}

LABELS = {
    "No-Privacy": "No Privacy Baseline",
    "Pixel-Gaussian": "Pixel Gaussian",
    "Embedding-Laplace": "Embedding Laplace",
    "vMF-Ours": "vMF (Ours)",
}


def load_rows(path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: dict[str, float | str] = {"method": row["method"]}
            for key, value in row.items():
                if key == "method":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    return rows


def group_rows(rows: list[dict[str, float | str]]) -> dict[str, list[dict[str, float | str]]]:
    grouped: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    for method in grouped:
        grouped[method].sort(key=lambda item: float(item["epsilon"]))
    return grouped


def validate_dataset(config: dict[str, object], baseline_rows: list[dict[str, float | str]], grouped: dict[str, list[dict[str, float | str]]]) -> None:
    if len(baseline_rows) != 1:
        raise ValueError(f"{config['name']} baseline 必须且只能有 1 行")

    for metric in config["utility_metrics"]:
        for method, rows in grouped.items():
            vals = [float(row[metric]) for row in rows]
            if any(curr < prev for prev, curr in zip(vals, vals[1:])):
                raise ValueError(f"{config['name']} {method} 的 {metric} 不是单调上升")

    for metric in config["privacy_metrics"]:
        for method, rows in grouped.items():
            vals = [float(row[metric]) for row in rows]
            if any(curr < prev for prev, curr in zip(vals, vals[1:])):
                raise ValueError(f"{config['name']} {method} 的 {metric} 不是单调上升")


def plot_metric(ax, baseline_row: dict[str, float | str], grouped, metric: str, title: str, ylabel: str, higher_is_better: bool) -> None:
    std_key = f"{metric}_std"
    non_baseline_eps = []
    baseline_value = float(baseline_row[metric])
    baseline_std = float(baseline_row[std_key])

    for rows in grouped.values():
        non_baseline_eps.extend(float(row["epsilon"]) for row in rows)
    x_min = min(non_baseline_eps) if non_baseline_eps else 0.1
    x_max = max(non_baseline_eps) if non_baseline_eps else 2.0

    ax.axhline(
        baseline_value,
        color=COLORS["No-Privacy"],
        linestyle="--",
        linewidth=2.0,
        label=LABELS["No-Privacy"],
    )
    ax.fill_between(
        [x_min, x_max],
        [baseline_value - baseline_std, baseline_value - baseline_std],
        [baseline_value + baseline_std, baseline_value + baseline_std],
        color=COLORS["No-Privacy"],
        alpha=0.08,
    )

    for method, rows in grouped.items():
        eps = np.array([float(row["epsilon"]) for row in rows])
        vals = np.array([float(row[metric]) for row in rows])
        stds = np.array([float(row[std_key]) for row in rows])
        ax.errorbar(
            eps,
            vals,
            yerr=stds,
            marker=MARKERS[method],
            markersize=5.5,
            linewidth=2.0,
            color=COLORS[method],
            label=LABELS[method],
            capsize=3,
            elinewidth=1.2,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)

    if higher_is_better:
        ax.annotate("Higher is better", xy=(0.04, 0.92), xycoords="axes fraction", fontsize=8, color="#374151")
    else:
        ax.annotate("Lower is better", xy=(0.04, 0.92), xycoords="axes fraction", fontsize=8, color="#374151")


def plot_tradeoff_pair(ax, baseline_row: dict[str, float | str], grouped, x_metric: str, y_metric: str, x_label: str, y_label: str, title: str) -> None:
    x_std_key = f"{x_metric}_std"
    y_std_key = f"{y_metric}_std"

    base_x = float(baseline_row[x_metric])
    base_y = float(baseline_row[y_metric])
    base_x_std = float(baseline_row[x_std_key])
    base_y_std = float(baseline_row[y_std_key])

    ax.errorbar(
        [base_x],
        [base_y],
        xerr=[base_x_std],
        yerr=[base_y_std],
        fmt="*",
        markersize=12,
        color=COLORS["No-Privacy"],
        capsize=3,
        label=LABELS["No-Privacy"],
    )

    for method, rows in grouped.items():
        x_vals = np.array([float(row[x_metric]) for row in rows])
        y_vals = np.array([float(row[y_metric]) for row in rows])
        x_err = np.array([float(row[x_std_key]) for row in rows])
        y_err = np.array([float(row[y_std_key]) for row in rows])
        eps = [str(row["epsilon"]) for row in rows]

        ax.errorbar(
            x_vals,
            y_vals,
            xerr=x_err,
            yerr=y_err,
            fmt=f"{MARKERS[method]}-",
            linewidth=1.9,
            markersize=5.5,
            color=COLORS[method],
            capsize=2,
            label=LABELS[method],
        )

        for x, y, ep in zip(x_vals, y_vals, eps):
            ax.annotate(ep, (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7, color=COLORS[method])

    ax.set_xlabel(f"{x_label} (Lower is better)")
    ax.set_ylabel(f"{y_label} (Higher is better)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)


def render_epsilon_curves(config: dict[str, object], baseline_row: dict[str, float | str], grouped) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"{config['name']} Results Across Privacy Budgets", fontsize=16, fontweight="bold")

    metrics = config["metrics"]
    for ax, (metric, title, ylabel, higher_is_better) in zip(axes.flat, metrics):
        plot_metric(ax, baseline_row, grouped, metric, title, ylabel, higher_is_better)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.97))
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    output_path = config["epsilon_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"已生成图像: {output_path}")


def render_tradeoff(config: dict[str, object], baseline_row: dict[str, float | str], grouped) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    fig.suptitle(f"{config['name']} Privacy-Utility Trade-off", fontsize=16, fontweight="bold")

    for ax, (x_metric, y_metric, x_label, y_label, title) in zip(axes.flat, config["tradeoff_pairs"]):
        plot_tradeoff_pair(ax, baseline_row, grouped, x_metric, y_metric, x_label, y_label, title)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.98))
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path = config["tradeoff_output"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"已生成图像: {output_path}")


def main() -> None:
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10

    for config in DATASETS:
        baseline_rows = load_rows(config["baseline_csv"])
        result_rows = load_rows(config["results_csv"])
        grouped = group_rows(result_rows)
        baseline_row = baseline_rows[0]

        validate_dataset(config, baseline_rows, grouped)
        render_epsilon_curves(config, baseline_row, grouped)
        render_tradeoff(config, baseline_row, grouped)


if __name__ == "__main__":
    main()
