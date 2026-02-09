# -*- coding: utf-8 -*-
"""
vMF 扰动机制可视化
生成 4 张子图展示 vMF 机制的核心特性
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
import os


# ── 内联扰动实现（避免 torch 依赖）─────────────────────
def vmf_perturb(x, epsilon, beta=1.0):
    """vMF 扰动（纯 numpy）"""
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    if r < 1e-10:
        return x.copy()
    mu = x / r
    n = np.random.randn(*x.shape)
    n_perp = n - np.dot(n, mu) * mu
    n_perp_norm = np.linalg.norm(n_perp)
    if n_perp_norm < 1e-10:
        return x.copy()
    n_perp = n_perp / n_perp_norm
    lam = beta / epsilon
    z = mu + lam * n_perp
    mu_prime = z / np.linalg.norm(z)
    return r * mu_prime


def gaussian_perturb(x, epsilon, delta=1e-5):
    """高斯机制扰动"""
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    sigma = r * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return x + np.random.randn(*x.shape) * sigma


def laplace_perturb(x, epsilon):
    """拉普拉斯机制扰动"""
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    scale = r / epsilon
    return x + np.random.laplace(0, scale, x.shape)

# ── 全局样式 ──────────────────────────────────────────
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150

COLOR_VMF = '#2563EB'
COLOR_GAUSS = '#DC2626'
COLOR_LAP = '#16A34A'
COLOR_ORIG = '#F59E0B'


def draw_sphere_wireframe(ax, alpha=0.08):
    """绘制半透明单位球面线框"""
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 20)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=alpha, color='lightblue', edgecolor='grey', linewidth=0.2)


# ══════════════════════════════════════════════════════
# 图 1：3D 球面上的 vMF 扰动几何示意
# ══════════════════════════════════════════════════════
def plot_sphere_perturbation(ax):
    """在单位球面上展示 vMF 扰动的几何过程"""
    draw_sphere_wireframe(ax)

    # 原始方向 μ（指向北偏东）
    mu = np.array([0.3, 0.3, 0.9])
    mu = mu / np.linalg.norm(mu)

    # 用不同 epsilon 生成扰动点
    np.random.seed(42)
    n_samples = 80
    epsilons = [0.5, 1.0, 2.0]
    colors = ['#EF4444', '#F59E0B', '#22C55E']
    labels = ['ε=0.5 (强隐私)', 'ε=1.0 (中等)', 'ε=2.0 (弱隐私)']

    for eps, c, lab in zip(epsilons, colors, labels):
        pts = []
        for _ in range(n_samples):
            y = vmf_perturb(mu.copy(), epsilon=eps)
            y = y / np.linalg.norm(y)
            pts.append(y)
        pts = np.array(pts)
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                   s=8, alpha=0.6, c=c, label=lab, depthshade=True)

    # 画原始向量（粗箭头）
    ax.quiver(0, 0, 0, mu[0], mu[1], mu[2],
              color='black', arrow_length_ratio=0.12, linewidth=2.5, label='原始方向 μ')

    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_zlim([-1.1, 1.1])
    ax.set_xlabel('X', fontsize=8)
    ax.set_ylabel('Y', fontsize=8)
    ax.set_zlabel('Z', fontsize=8)
    ax.set_title('(a) 球面上的 vMF 扰动分布', fontsize=11, fontweight='bold', pad=10)
    ax.legend(fontsize=7, loc='upper left', framealpha=0.8)
    ax.view_init(elev=25, azim=45)


# ══════════════════════════════════════════════════════
# 图 2：偏转角度 vs 隐私预算 ε
# ══════════════════════════════════════════════════════
def plot_angle_vs_epsilon(ax):
    """理论偏转角度随 ε 变化的曲线"""
    epsilons = np.linspace(0.05, 5.0, 200)
    angles = np.degrees(np.arctan(1.0 / epsilons))  # β=1

    ax.plot(epsilons, angles, color=COLOR_VMF, linewidth=2.5, label='θ = arctan(β/ε)')
    ax.fill_between(epsilons, angles, alpha=0.1, color=COLOR_VMF)

    # 标注关键点
    key_eps = [0.1, 0.5, 1.0, 2.0, 5.0]
    for e in key_eps:
        a = np.degrees(np.arctan(1.0 / e))
        ax.plot(e, a, 'o', color=COLOR_VMF, markersize=6, zorder=5)
        ax.annotate(f'({e}, {a:.1f}°)', (e, a),
                    textcoords='offset points', xytext=(8, 5), fontsize=7)

    # 隐私强度区域
    ax.axhspan(60, 90, alpha=0.08, color='red', label='强隐私区 (60°-90°)')
    ax.axhspan(30, 60, alpha=0.08, color='orange', label='中等隐私区 (30°-60°)')
    ax.axhspan(0, 30, alpha=0.08, color='green', label='弱隐私区 (0°-30°)')

    ax.set_xlabel('隐私预算 ε', fontsize=10)
    ax.set_ylabel('偏转角度 θ (度)', fontsize=10)
    ax.set_title('(b) 隐私预算与偏转角度的关系', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_xlim([0, 5.2])
    ax.set_ylim([0, 92])
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════
# 图 3：范数比较 — vMF vs Gaussian vs Laplace
# ══════════════════════════════════════════════════════
def plot_norm_comparison(ax):
    """不同机制扰动后的范数分布"""
    np.random.seed(123)
    d = 512
    n_samples = 500
    x = np.random.randn(n_samples, d).astype(np.float32)
    # 归一化到固定范数
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / norms * 10.0  # 范数 = 10

    eps = 0.5
    norms_vmf, norms_gauss, norms_lap = [], [], []
    for i in range(n_samples):
        y_v = vmf_perturb(x[i], epsilon=eps)
        y_g = gaussian_perturb(x[i], epsilon=eps)
        y_l = laplace_perturb(x[i], epsilon=eps)
        norms_vmf.append(np.linalg.norm(y_v))
        norms_gauss.append(np.linalg.norm(y_g))
        norms_lap.append(np.linalg.norm(y_l))

    bins = np.linspace(0, max(max(norms_gauss), max(norms_lap)) * 1.05, 60)
    ax.hist(norms_vmf, bins=bins, alpha=0.7, color=COLOR_VMF, label=f'vMF (μ={np.mean(norms_vmf):.1f})', density=True)
    ax.hist(norms_gauss, bins=bins, alpha=0.5, color=COLOR_GAUSS, label=f'Gaussian (μ={np.mean(norms_gauss):.1f})', density=True)
    ax.hist(norms_lap, bins=bins, alpha=0.5, color=COLOR_LAP, label=f'Laplace (μ={np.mean(norms_lap):.1f})', density=True)

    ax.axvline(x=10.0, color='black', linestyle='--', linewidth=1.5, label='原始范数 = 10.0')

    ax.set_xlabel('扰动后向量范数', fontsize=10)
    ax.set_ylabel('概率密度', fontsize=10)
    ax.set_title(f'(c) 范数保持性比较 (ε={eps}, d={d})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════
# 图 4：多机制效用对比（余弦相似度 vs ε）
# ══════════════════════════════════════════════════════
def plot_cosine_distribution(ax):
    """不同机制在不同 ε 下的平均余弦相似度对比"""
    np.random.seed(456)
    d = 512
    n_samples = 200

    # 生成固定测试向量
    x_all = np.random.randn(n_samples, d)
    norms = np.linalg.norm(x_all, axis=1, keepdims=True)
    x_all = x_all / norms * 5.0

    epsilons = np.array([0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0])

    def mean_cosine(perturb_fn, x_all):
        cosines = []
        for x in x_all:
            y = perturb_fn(x)
            cos_sim = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-10)
            cosines.append(np.clip(cos_sim, -1, 1))
        return np.mean(cosines), np.std(cosines)

    cos_vmf, cos_gauss, cos_lap = [], [], []
    std_vmf, std_gauss, std_lap = [], [], []

    for eps in epsilons:
        m, s = mean_cosine(lambda x: vmf_perturb(x, epsilon=eps), x_all)
        cos_vmf.append(m); std_vmf.append(s)
        m, s = mean_cosine(lambda x: gaussian_perturb(x, epsilon=eps), x_all)
        cos_gauss.append(m); std_gauss.append(s)
        m, s = mean_cosine(lambda x: laplace_perturb(x, epsilon=eps), x_all)
        cos_lap.append(m); std_lap.append(s)

    cos_vmf, cos_gauss, cos_lap = np.array(cos_vmf), np.array(cos_gauss), np.array(cos_lap)
    std_vmf, std_gauss, std_lap = np.array(std_vmf), np.array(std_gauss), np.array(std_lap)

    # 理论曲线
    eps_fine = np.linspace(0.05, 5.5, 200)
    theory = 1.0 / np.sqrt(1 + 1.0 / eps_fine**2)
    ax.plot(eps_fine, theory, '--', color='grey', linewidth=1.2, label='vMF 理论值', zorder=1)

    # 实验曲线
    ax.plot(epsilons, cos_vmf, 'o-', color=COLOR_VMF, linewidth=2.2, markersize=6, label='vMF', zorder=3)
    ax.fill_between(epsilons, cos_vmf - std_vmf, cos_vmf + std_vmf, alpha=0.12, color=COLOR_VMF)

    ax.plot(epsilons, cos_gauss, 's-', color=COLOR_GAUSS, linewidth=2.2, markersize=6, label='Gaussian', zorder=3)
    ax.fill_between(epsilons, cos_gauss - std_gauss, cos_gauss + std_gauss, alpha=0.12, color=COLOR_GAUSS)

    ax.plot(epsilons, cos_lap, '^-', color=COLOR_LAP, linewidth=2.2, markersize=6, label='Laplace', zorder=3)
    ax.fill_between(epsilons, cos_lap - std_lap, cos_lap + std_lap, alpha=0.12, color=COLOR_LAP)

    ax.set_xlabel('隐私预算 ε', fontsize=10)
    ax.set_ylabel('平均余弦相似度', fontsize=10)
    ax.set_title('(d) 隐私-效用权衡：余弦相似度对比', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.set_xlim([0, 5.3])
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)


# ══════════════════════════════════════════════════════
# 主函数：组合 4 张子图
# ══════════════════════════════════════════════════════
if __name__ == '__main__':
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('vMF 扰动机制可视化分析', fontsize=16, fontweight='bold', y=0.98)

    # (a) 3D 球面
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    plot_sphere_perturbation(ax1)

    # (b) 角度 vs epsilon
    ax2 = fig.add_subplot(2, 2, 2)
    plot_angle_vs_epsilon(ax2)

    # (c) 范数比较
    ax3 = fig.add_subplot(2, 2, 3)
    plot_norm_comparison(ax3)

    # (d) 余弦相似度
    ax4 = fig.add_subplot(2, 2, 4)
    plot_cosine_distribution(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    out_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'asset', 'vmf_visualization.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f'图片已保存至: {out_path}')
    plt.close()
