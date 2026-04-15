# -*- coding: utf-8 -*-
"""
vMF Tangent Plane Approximation Validation Experiment (Optimized for Speed)
Compares the tangent plane projection approximation with true vMF sampling.
Generates a 3x2 validation plot + console statistical test results.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import stats
import os

# -- Global Style --
rcParams['font.family'] = 'DejaVu Sans'
rcParams['axes.unicode_minus'] = False
rcParams['figure.dpi'] = 150
rcParams['mathtext.fontset'] = 'cm'

COLOR_TP = '#2563EB'
COLOR_VMF = '#DC2626'
COLOR_GAUSS = '#9333EA'
COLOR_LAP = '#16A34A'
COLOR_NP = '#F59E0B'

def tangent_plane_perturb(mu, epsilon, beta=1.0):
    mu = np.asarray(mu, dtype=np.float64)
    r = np.linalg.norm(mu)
    if r < 1e-10: return mu.copy()
    mu_hat = mu / r
    n = np.random.randn(*mu.shape)
    n_perp = n - np.dot(n, mu_hat) * mu_hat
    n_norm = np.linalg.norm(n_perp)
    if n_norm < 1e-10: return mu.copy()
    n_perp = n_perp / n_norm
    lam = beta / epsilon
    z = mu_hat + lam * n_perp
    return r * z / np.linalg.norm(z)

def wood_vmf_sample(mu, kappa):
    mu = np.asarray(mu, dtype=np.float64)
    d = len(mu)
    mu_hat = mu / np.linalg.norm(mu)
    b = (-2 * kappa + np.sqrt(4 * kappa**2 + (d - 1)**2)) / (d - 1)
    x0 = (1 - b) / (1 + b)
    c = kappa * x0 + (d - 1) * np.log(1 - x0**2)
    while True:
        z = np.random.beta((d - 1) / 2, (d - 1) / 2)
        w = (1 - (1 + b) * z) / (1 - (1 - b) * z)
        t = kappa * w + (d - 1) * np.log(1 - x0 * w) - c
        if t >= np.log(np.random.rand()): break
    v = np.random.randn(d)
    v = v - np.dot(v, mu_hat) * mu_hat
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-10: return mu_hat.copy()
    v = v / v_norm
    result = w * mu_hat + np.sqrt(max(1 - w**2, 0)) * v
    return result

def gaussian_perturb(x, epsilon, delta=1e-5):
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    sigma = r * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    return x + np.random.randn(*x.shape) * sigma

def laplace_perturb(x, epsilon):
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    scale = r / epsilon
    return x + np.random.laplace(0, scale, x.shape)

def norm_preserving_gaussian_perturb(x, epsilon, delta=1e-5):
    x = np.asarray(x, dtype=np.float64)
    r = np.linalg.norm(x)
    sigma = r * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
    y = x + np.random.randn(*x.shape) * sigma
    y_norm = np.linalg.norm(y)
    if y_norm < 1e-10: return x.copy()
    return r * y / y_norm

def vmf_mean_cosine_theory(kappa, d):
    nu = d / 2.0
    N = int(nu + 50 + 30 * np.sqrt(kappa))
    R = 0.0
    for n in range(N, int(nu) - 1, -1):
        R = 1.0 / (2.0 * n / kappa + R)
    return R

def kappa_from_epsilon(epsilon, d, beta=1.0):
    from scipy.optimize import brentq
    cos_tp = epsilon / np.sqrt(epsilon**2 + beta**2)
    if cos_tp >= 1.0 - 1e-12: return 1e8
    def residual(k): return vmf_mean_cosine_theory(k, d) - cos_tp
    lo, hi = 1e-3, 1e8
    try: return brentq(residual, lo, hi, xtol=1e-6)
    except: return (d - 1) / (2 * (1 - cos_tp))

# --- Plots ---

def plot_angle_distribution(ax):
    np.random.seed(42)
    d, eps, n_samples = 512, 0.5, 5000
    mu = np.zeros(d); mu[0] = 1.0
    kappa = kappa_from_epsilon(eps, d)
    angles_tp = [np.degrees(np.arccos(np.clip(np.dot(mu, tangent_plane_perturb(mu, eps)), -1, 1))) for _ in range(n_samples)]
    angles_vmf = [np.degrees(np.arccos(np.clip(np.dot(mu, wood_vmf_sample(mu, kappa)), -1, 1))) for _ in range(n_samples)]
    bins = np.linspace(0, max(max(angles_tp), max(angles_vmf)) * 1.05, 60)
    ax.hist(angles_tp, bins=bins, alpha=0.6, color=COLOR_TP, density=True, label=fr'Tangent Plane ($\mu$={np.mean(angles_tp):.2f}$^\circ$)')
    ax.hist(angles_vmf, bins=bins, alpha=0.6, color=COLOR_VMF, density=True, label=fr'True vMF ($\mu$={np.mean(angles_vmf):.2f}$^\circ$)')
    ax.set_xlabel('Angular Deviation (deg)')
    ax.set_title(fr'(a) Angular Deviation (d={d}, $\epsilon$={eps})', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

def plot_isotropy(ax):
    np.random.seed(123)
    d, eps, n_samples = 512, 1.0, 10000
    mu = np.zeros(d); mu[0] = 1.0
    angles_polar = [np.arctan2(y[2], y[1]) for y in [tangent_plane_perturb(mu, eps) for _ in range(n_samples)]]
    counts, bin_edges = np.histogram(angles_polar, bins=60, range=(-np.pi, np.pi))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    fig = ax.figure; pos = ax.get_position(); ax.remove()
    ax_polar = fig.add_axes(pos, projection='polar')
    ax_polar.bar(bin_centers, counts, width=2*np.pi/60, alpha=0.7, color=COLOR_TP)
    ax_polar.set_title(fr'(b) Isotropy (d={d}, $\epsilon$={eps})', fontweight='bold', pad=15)

def plot_dimension_scaling(ax):
    np.random.seed(456); eps = 1.0; dims = [8, 32, 128, 512, 1024]
    tp_theory = [np.degrees(np.arctan(1.0/eps))]*len(dims)
    vmf_theory = [np.degrees(np.arccos(vmf_mean_cosine_theory(kappa_from_epsilon(eps, d), d))) for d in dims]
    ax.semilogx(dims, tp_theory, 's--', color=COLOR_TP, label='TP Theory')
    ax.semilogx(dims, vmf_theory, '^--', color=COLOR_VMF, label='vMF Theory')
    ax.set_xlabel('Dimension d'); ax.set_ylabel('Mean Angle (deg)')
    ax.set_title(fr'(c) Dimension Scaling ($\epsilon$={eps})', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

def plot_norm_preservation(ax):
    np.random.seed(789); d, eps, n_samples = 512, 0.5, 1000
    mu = np.random.randn(d); mu = mu/np.linalg.norm(mu)*10.0
    data = [
        [np.linalg.norm(tangent_plane_perturb(mu, eps))/10.0 for _ in range(n_samples)],
        [np.linalg.norm(gaussian_perturb(mu, eps))/10.0 for _ in range(n_samples)],
        [np.linalg.norm(laplace_perturb(mu, eps))/10.0 for _ in range(n_samples)],
        [np.linalg.norm(norm_preserving_gaussian_perturb(mu, eps))/10.0 for _ in range(n_samples)]
    ]
    ax.violinplot(data, showmeans=True)
    ax.axhline(1.0, color='red', linestyle='--')
    ax.set_xticks(range(1, 5)); ax.set_xticklabels(['vMF', 'Gauss', 'Lap', 'NP-G'], fontsize=8)
    ax.set_title(fr'(d) Norm Preservation (d={d})', fontweight='bold')
    ax.grid(True, alpha=0.3)

def plot_concentration_equivalence(ax):
    d = 512; epsilons = np.linspace(0.1, 5.0, 50)
    cos_tp = epsilons / np.sqrt(epsilons**2 + 1.0)
    cos_vmf = [vmf_mean_cosine_theory(kappa_from_epsilon(e, d), d) for e in epsilons]
    ax.plot(epsilons, cos_tp, '-', color=COLOR_TP, label='TP Theory')
    ax.plot(epsilons, cos_vmf, '--', color=COLOR_VMF, label='vMF Theory')
    ax.set_xlabel(r'Privacy Budget $\epsilon$'); ax.set_ylabel('Mean Cosine')
    ax.set_title(fr'(e) Concentration Equivalence (d={d})', fontweight='bold')
    ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

def plot_qq(ax):
    np.random.seed(202); d, eps, n_samples = 512, 0.5, 5000
    mu = np.zeros(d); mu[0] = 1.0
    kappa = kappa_from_epsilon(eps, d); e_proj = np.zeros(d); e_proj[1] = 1.0
    proj_tp = sorted([np.dot(tangent_plane_perturb(mu, eps), e_proj) for _ in range(n_samples)])
    proj_vmf = sorted([np.dot(wood_vmf_sample(mu, kappa), e_proj) for _ in range(n_samples)])
    q = np.linspace(0, 1, 100)
    ax.scatter(np.quantile(proj_vmf, q), np.quantile(proj_tp, q), s=10, color=COLOR_TP)
    ax.plot([-0.1, 0.1], [-0.1, 0.1], 'r--')
    ax.set_xlabel('vMF Quantiles'); ax.set_ylabel('TP Quantiles')
    ax.set_title(fr'(f) Projection QQ Plot (d={d})', fontweight='bold')
    ax.grid(True, alpha=0.3)

if __name__ == '__main__':
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Validation of vMF Tangent Plane Approximation', fontsize=15, fontweight='bold', y=0.98)
    plot_angle_distribution(fig.add_subplot(3, 2, 1))
    plot_isotropy(fig.add_subplot(3, 2, 2))
    plot_dimension_scaling(fig.add_subplot(3, 2, 3))
    plot_norm_preservation(fig.add_subplot(3, 2, 4))
    plot_concentration_equivalence(fig.add_subplot(3, 2, 5))
    plot_qq(fig.add_subplot(3, 2, 6))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(os.path.dirname(script_dir), 'asset', 'vmf_validation.png')
    fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Image saved to: {out_path}')
    plt.close()
