"""
公平对比多种差分隐私机制在 Embedding 扰动上的效果
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from privacy import vMFMechanism, GaussianMechanism, LaplaceMechanism, NormPreservingGaussian


def compute_stats(original, perturbed):
    """计算扰动统计量"""
    if original.ndim == 1:
        orig_2d = original.reshape(1, -1)
        pert_2d = perturbed.reshape(1, -1)
    elif original.ndim > 2:
        orig_2d = original.reshape(-1, original.shape[-1])
        pert_2d = perturbed.reshape(-1, perturbed.shape[-1])
    else:
        orig_2d = original
        pert_2d = perturbed

    orig_norms = np.linalg.norm(orig_2d, axis=1)
    pert_norms = np.linalg.norm(pert_2d, axis=1)
    norm_ratio = pert_norms / (orig_norms + 1e-10)

    cos_sim = np.sum(orig_2d * pert_2d, axis=1) / (orig_norms * pert_norms + 1e-10)
    cos_sim = np.clip(cos_sim, -1, 1)
    angles = np.arccos(cos_sim) * 180 / np.pi

    l2_dist = np.linalg.norm(orig_2d - pert_2d, axis=1)

    return {
        'angle_mean': np.mean(angles),
        'angle_std': np.std(angles),
        'norm_ratio_mean': np.mean(norm_ratio),
        'norm_ratio_std': np.std(norm_ratio),
        'l2_dist_mean': np.mean(l2_dist),
        'l2_dist_std': np.std(l2_dist),
        'cos_sim_mean': np.mean(cos_sim),
    }


def run_comparison(dim=4096, n_samples=100, epsilons=[0.1, 0.5, 1.0, 2.0]):
    """运行对比实验"""
    print("=" * 90)
    print("公平对比：Embedding 扰动机制")
    print("=" * 90)
    print(f"维度: {dim}, 样本数: {n_samples}")

    # 生成模拟 embedding 数据
    np.random.seed(42)
    data = np.random.randn(n_samples, dim).astype(np.float32)
    target_norms = np.random.uniform(0.5, 2.0, (n_samples, 1))
    data = data / np.linalg.norm(data, axis=1, keepdims=True) * target_norms

    print(f"\n数据统计:")
    print(f"  范数范围: [{np.min(np.linalg.norm(data, axis=1)):.2f}, {np.max(np.linalg.norm(data, axis=1)):.2f}]")
    print(f"  平均范数: {np.mean(np.linalg.norm(data, axis=1)):.2f}")

    mechanisms = {
        'vMF': vMFMechanism,
        'Gaussian': GaussianMechanism,
        'Laplace': LaplaceMechanism,
        'NormPreserving': NormPreservingGaussian,
    }

    all_results = {}

    for eps in epsilons:
        print(f"\n{'='*90}")
        print(f"Epsilon = {eps}")
        print("=" * 90)

        all_results[eps] = {}

        for name, MechClass in mechanisms.items():
            if name == 'vMF':
                mech = MechClass(epsilon=eps, beta=1.0)
            elif name in ['Gaussian', 'NormPreserving']:
                mech = MechClass(epsilon=eps, delta=1e-5)
            else:
                mech = MechClass(epsilon=eps)

            perturbed = mech.perturb(data.copy())
            stats = compute_stats(data, perturbed)
            all_results[eps][name] = stats

            print(f"\n[{name}]")
            print(f"  角度偏差: {stats['angle_mean']:.2f}° ± {stats['angle_std']:.2f}°")
            print(f"  范数比例: {stats['norm_ratio_mean']:.4f} ± {stats['norm_ratio_std']:.4f}")
            print(f"  L2 距离:  {stats['l2_dist_mean']:.4f} ± {stats['l2_dist_std']:.4f}")

    # 打印对比表格
    print("\n" + "=" * 90)
    print("对比总结表")
    print("=" * 90)

    mech_names = list(mechanisms.keys())

    print("\n【角度偏差 (度)】")
    print(f"{'ε':<8}", end="")
    for name in mech_names:
        print(f"{name:<18}", end="")
    print()
    print("-" * 80)
    for eps in epsilons:
        print(f"{eps:<8}", end="")
        for name in mech_names:
            angle = all_results[eps][name]['angle_mean']
            print(f"{angle:.2f}°".ljust(18), end="")
        print()

    print("\n【范数比例】")
    print(f"{'ε':<8}", end="")
    for name in mech_names:
        print(f"{name:<18}", end="")
    print()
    print("-" * 80)
    for eps in epsilons:
        print(f"{eps:<8}", end="")
        for name in mech_names:
            ratio = all_results[eps][name]['norm_ratio_mean']
            print(f"{ratio:.4f}".ljust(18), end="")
        print()

    return all_results


if __name__ == "__main__":
    results = run_comparison(
        dim=4096,
        n_samples=100,
        epsilons=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    )
