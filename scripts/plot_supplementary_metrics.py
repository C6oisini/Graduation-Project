import matplotlib.pyplot as plt
import numpy as np
import csv
from pathlib import Path

from plot_style import apply_plot_style, style_axis

ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = ROOT / "asset"
DATA_DIR = ROOT / "data"

def load_csv(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed = {}
            for k, v in row.items():
                try:
                    parsed[k] = float(v)
                except ValueError:
                    parsed[k] = v
            data.append(parsed)
    return data

def plot_ablation_study():
    data = load_csv(DATA_DIR / "ablation_study.csv")
    settings = sorted(list(set(d['setting'] for d in data)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.8))
    
    colors = ['#10b981', '#f59e0b', '#2563eb']
    markers = ['o', 's', 'D']
    
    for i, setting in enumerate(settings):
        subset = [d for d in data if d['setting'] == setting]
        subset.sort(key=lambda x: x['epsilon'])
        eps = [d['epsilon'] for d in subset]
        cider = [d['CIDEr'] for d in subset]
        asr = [d['AttackSuccessRate'] for d in subset]
        
        ax1.plot(eps, cider, label=setting, marker=markers[i], color=colors[i], linewidth=2.2, markersize=6.5, markeredgecolor="white", markeredgewidth=0.8)
        ax2.plot(eps, asr, label=setting, marker=markers[i], color=colors[i], linewidth=2.2, markersize=6.5, markeredgecolor="white", markeredgewidth=0.8)
    
    ax1.set_title("Utility: CIDEr Score (Ablation)", fontweight='bold')
    ax1.set_xscale("log")
    ax1.set_xlabel("Privacy Budget ε")
    ax1.set_ylabel("CIDEr")
    ax1.legend(frameon=False, loc="upper left")
    style_axis(ax1)
    
    ax2.set_title("Privacy: Attack Success Rate (Ablation)", fontweight='bold')
    ax2.set_xscale("log")
    ax2.set_xlabel("Privacy Budget ε")
    ax2.set_ylabel("ASR")
    ax2.legend(frameon=False, loc="upper left")
    style_axis(ax2)
    
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.14, top=0.88, wspace=0.25)
    fig.savefig(ASSET_DIR / "supplementary_ablation.png", dpi=300, bbox_inches="tight")
    print(f"已生成: {ASSET_DIR / 'supplementary_ablation.png'}")

def plot_norm_stability():
    data = load_csv(DATA_DIR / "norm_stability.csv")
    data = [d for d in data if d['mechanism'] != 'None']
    mechs = ['Gaussian', 'Laplace', 'vMF(Ours)'] # 固定顺序
    epsilons = sorted(list(set(d['epsilon'] for d in data)))
    
    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    
    x = np.arange(len(epsilons))
    width = 0.22
    
    colors = {'vMF(Ours)': '#2563eb', 'Laplace': '#d97706', 'Gaussian': '#dc2626'}
    
    for i, mech in enumerate(mechs):
        norms = []
        for e in epsilons:
            # 查找匹配 mech 和 epsilon 的行
            match = [d for d in data if d['mechanism'] == mech and d['epsilon'] == e]
            if match:
                norms.append(match[0]['norm_mean'])
            else:
                norms.append(0) # 或者 np.nan，但 bar 绘图 0 比较安全
        
        offset = (i - 1) * width * 1.15
        ax.bar(x + offset, norms, width, label=mech, color=colors.get(mech, '#999'), edgecolor="white", linewidth=0.8)
    
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1.5, label="Ideal Norm (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels([f"ε={e}" for e in epsilons])
    ax.set_xlabel("Privacy Budget ε")
    ax.set_ylabel("Mean Norm")
    ax.set_ylim(0.0, 1.4)
    ax.set_title("Embedding Norm Stability Comparison", fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    style_axis(ax, grid_axis="y")
    
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.14, top=0.88)
    fig.savefig(ASSET_DIR / "supplementary_norm_stability.png", dpi=300, bbox_inches="tight")
    print(f"已生成: {ASSET_DIR / 'supplementary_norm_stability.png'}")

def plot_cross_model():
    data = load_csv(DATA_DIR / "cross_model_llava.csv")
    methods = sorted(list(set(d['method'] for d in data)))
    
    fig, ax1 = plt.subplots(figsize=(9.6, 5.8))
    ax2 = ax1.twinx()
    
    colors = {'vMF-Ours': '#2563eb', 'Laplace': '#d97706'}
    
    for method in methods:
        subset = [d for d in data if d['method'] == method]
        subset.sort(key=lambda x: x['epsilon'])
        eps = [d['epsilon'] for d in subset]
        acc = [d['Accuracy'] for d in subset]
        asr = [d['AttackSuccessRate'] for d in subset]
        
        ax1.plot(eps, acc, label=f"{method} Accuracy", marker='o', color=colors.get(method), linewidth=2.2, markersize=6.5, markeredgecolor="white", markeredgewidth=0.8)
        ax2.plot(eps, asr, label=f"{method} ASR", marker='x', color=colors.get(method), linestyle='--', linewidth=2.0, alpha=0.85)
        
    ax1.set_xlabel("Privacy Budget ε")
    ax1.set_ylabel("Accuracy (%)", color='#2563eb')
    ax2.set_ylabel("Attack Success Rate", color='#dc2626')
    ax1.set_title("Generalization on LLaVA-1.5 Model", fontweight='bold')
    
    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', frameon=True, framealpha=0.9)
    
    style_axis(ax1)
    ax2.tick_params(axis="y", pad=5)
    fig.subplots_adjust(left=0.10, right=0.88, bottom=0.14, top=0.88)
    fig.savefig(ASSET_DIR / "supplementary_cross_model.png", dpi=300, bbox_inches="tight")
    print(f"已生成: {ASSET_DIR / 'supplementary_cross_model.png'}")

if __name__ == "__main__":
    apply_plot_style()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    plot_ablation_study()
    plot_norm_stability()
    plot_cross_model()
