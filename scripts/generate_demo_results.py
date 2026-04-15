"""
写出一组手工构造的示例实验数据。

注意：该脚本输出的是展示用 mock 数据，不是实际运行实验得到的结果。
"""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_PATH = ROOT / "data" / "demo_mechanism_results.csv"

ROWS = [
    {"epsilon": 0.1, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 84.2, "angle_std": 1.6, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.96, "l2_dist_std": 0.03, "cos_sim_mean": 0.101, "cos_sim_std": 0.028},
    {"epsilon": 0.1, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 88.8, "angle_std": 0.4, "norm_ratio_mean": 72.400, "norm_ratio_std": 1.900, "l2_dist_mean": 74.10, "l2_dist_std": 2.10, "cos_sim_mean": 0.021, "cos_sim_std": 0.013},
    {"epsilon": 0.1, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 89.3, "angle_std": 0.3, "norm_ratio_mean": 127.800, "norm_ratio_std": 5.400, "l2_dist_mean": 130.50, "l2_dist_std": 5.80, "cos_sim_mean": 0.012, "cos_sim_std": 0.009},
    {"epsilon": 0.1, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 88.1, "angle_std": 0.6, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.99, "l2_dist_std": 0.02, "cos_sim_mean": 0.033, "cos_sim_std": 0.011},
    {"epsilon": 0.2, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 78.7, "angle_std": 1.5, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.88, "l2_dist_std": 0.04, "cos_sim_mean": 0.196, "cos_sim_std": 0.025},
    {"epsilon": 0.2, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 87.6, "angle_std": 0.5, "norm_ratio_mean": 36.900, "norm_ratio_std": 1.200, "l2_dist_mean": 37.80, "l2_dist_std": 1.30, "cos_sim_mean": 0.041, "cos_sim_std": 0.017},
    {"epsilon": 0.2, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 88.9, "angle_std": 0.4, "norm_ratio_mean": 63.700, "norm_ratio_std": 2.700, "l2_dist_mean": 65.20, "l2_dist_std": 2.90, "cos_sim_mean": 0.019, "cos_sim_std": 0.010},
    {"epsilon": 0.2, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 86.2, "angle_std": 0.7, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.96, "l2_dist_std": 0.03, "cos_sim_mean": 0.067, "cos_sim_std": 0.014},
    {"epsilon": 0.5, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 63.4, "angle_std": 0.8, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.53, "l2_dist_std": 0.03, "cos_sim_mean": 0.447, "cos_sim_std": 0.016},
    {"epsilon": 0.5, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 85.8, "angle_std": 0.7, "norm_ratio_mean": 14.900, "norm_ratio_std": 0.600, "l2_dist_mean": 15.70, "l2_dist_std": 0.70, "cos_sim_mean": 0.073, "cos_sim_std": 0.021},
    {"epsilon": 0.5, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 88.1, "angle_std": 0.5, "norm_ratio_mean": 25.600, "norm_ratio_std": 1.100, "l2_dist_mean": 26.90, "l2_dist_std": 1.20, "cos_sim_mean": 0.032, "cos_sim_std": 0.013},
    {"epsilon": 0.5, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 82.7, "angle_std": 0.9, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.87, "l2_dist_std": 0.03, "cos_sim_mean": 0.126, "cos_sim_std": 0.019},
    {"epsilon": 1.0, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 45.0, "angle_std": 0.6, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.09, "l2_dist_std": 0.02, "cos_sim_mean": 0.707, "cos_sim_std": 0.011},
    {"epsilon": 1.0, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 83.2, "angle_std": 0.9, "norm_ratio_mean": 7.600, "norm_ratio_std": 0.400, "l2_dist_mean": 8.20, "l2_dist_std": 0.40, "cos_sim_mean": 0.118, "cos_sim_std": 0.024},
    {"epsilon": 1.0, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 86.9, "angle_std": 0.7, "norm_ratio_mean": 12.800, "norm_ratio_std": 0.700, "l2_dist_mean": 13.70, "l2_dist_std": 0.80, "cos_sim_mean": 0.055, "cos_sim_std": 0.015},
    {"epsilon": 1.0, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 77.1, "angle_std": 1.1, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.74, "l2_dist_std": 0.04, "cos_sim_mean": 0.223, "cos_sim_std": 0.018},
    {"epsilon": 2.0, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 26.6, "angle_std": 0.5, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 0.65, "l2_dist_std": 0.02, "cos_sim_mean": 0.894, "cos_sim_std": 0.008},
    {"epsilon": 2.0, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 78.4, "angle_std": 1.1, "norm_ratio_mean": 4.100, "norm_ratio_std": 0.300, "l2_dist_mean": 4.60, "l2_dist_std": 0.30, "cos_sim_mean": 0.201, "cos_sim_std": 0.023},
    {"epsilon": 2.0, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 84.3, "angle_std": 0.9, "norm_ratio_mean": 6.900, "norm_ratio_std": 0.500, "l2_dist_mean": 7.60, "l2_dist_std": 0.50, "cos_sim_mean": 0.097, "cos_sim_std": 0.019},
    {"epsilon": 2.0, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 68.8, "angle_std": 1.2, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.57, "l2_dist_std": 0.04, "cos_sim_mean": 0.362, "cos_sim_std": 0.021},
    {"epsilon": 5.0, "mechanism": "vMF", "dim": 4096, "n_samples": 100, "angle_mean": 11.3, "angle_std": 0.3, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 0.28, "l2_dist_std": 0.01, "cos_sim_mean": 0.981, "cos_sim_std": 0.004},
    {"epsilon": 5.0, "mechanism": "Gaussian", "dim": 4096, "n_samples": 100, "angle_mean": 66.2, "angle_std": 1.7, "norm_ratio_mean": 2.100, "norm_ratio_std": 0.200, "l2_dist_mean": 2.70, "l2_dist_std": 0.20, "cos_sim_mean": 0.403, "cos_sim_std": 0.027},
    {"epsilon": 5.0, "mechanism": "Laplace", "dim": 4096, "n_samples": 100, "angle_mean": 76.8, "angle_std": 1.4, "norm_ratio_mean": 3.300, "norm_ratio_std": 0.300, "l2_dist_mean": 3.90, "l2_dist_std": 0.30, "cos_sim_mean": 0.228, "cos_sim_std": 0.024},
    {"epsilon": 5.0, "mechanism": "NormPreserving", "dim": 4096, "n_samples": 100, "angle_mean": 54.9, "angle_std": 1.3, "norm_ratio_mean": 1.000, "norm_ratio_std": 0.000, "l2_dist_mean": 1.23, "l2_dist_std": 0.03, "cos_sim_mean": 0.576, "cos_sim_std": 0.020},
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epsilon",
                "mechanism",
                "dim",
                "n_samples",
                "angle_mean",
                "angle_std",
                "norm_ratio_mean",
                "norm_ratio_std",
                "l2_dist_mean",
                "l2_dist_std",
                "cos_sim_mean",
                "cos_sim_std",
            ],
        )
        writer.writeheader()
        writer.writerows(ROWS)
    print(f"已写入示例数据: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
