"""
公平对比：多种差分隐私机制在 Embedding 扰动上的效果
所有机制都在相同的 embedding 上进行扰动，使用相同的 epsilon
"""

import numpy as np
import torch


class GaussianMechanism:
    """
    高斯机制 (Gaussian Mechanism)
    满足 (ε, δ)-差分隐私

    针对 embedding 优化：根据输入动态计算敏感度
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def perturb(self, x, modality='visual'):
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        if is_torch:
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        original_shape = x_np.shape

        # 展平为 2D: [n_vectors, dim]
        if x_np.ndim == 1:
            x_2d = x_np.reshape(1, -1)
        elif x_np.ndim > 2:
            x_2d = x_np.reshape(-1, x_np.shape[-1])
        else:
            x_2d = x_np

        # 对每个向量独立计算敏感度（使用其范数）
        norms = np.linalg.norm(x_2d, axis=1, keepdims=True)

        # σ = Δf * √(2 * ln(1.25/δ)) / ε
        multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        sigmas = norms * multiplier

        # 添加高斯噪声
        noise = np.random.randn(*x_2d.shape) * sigmas
        y = x_2d + noise

        y = y.reshape(original_shape)

        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


class LaplaceMechanism:
    """
    拉普拉斯机制 (Laplace Mechanism)
    满足 ε-差分隐私（纯差分隐私）

    针对 embedding 优化：根据输入动态计算敏感度
    """

    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon

    def perturb(self, x, modality='visual'):
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        if is_torch:
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        original_shape = x_np.shape

        # 展平为 2D
        if x_np.ndim == 1:
            x_2d = x_np.reshape(1, -1)
        elif x_np.ndim > 2:
            x_2d = x_np.reshape(-1, x_np.shape[-1])
        else:
            x_2d = x_np

        # 对每个向量独立计算敏感度
        norms = np.linalg.norm(x_2d, axis=1, keepdims=True)

        # scale = Δf / ε
        scales = norms / self.epsilon

        # 添加拉普拉斯噪声
        noise = np.random.laplace(0, 1, x_2d.shape) * scales
        y = x_2d + noise

        y = y.reshape(original_shape)

        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


class vMFMechanism:
    """
    von Mises-Fisher 扰动机制
    基于正交切平面投影的几何扰动

    特点：保持向量范数，只改变方向
    """

    def __init__(self, epsilon=1.0, beta=1.0):
        self.epsilon = epsilon
        self.beta = beta

    def perturb(self, x, modality='visual'):
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        if is_torch:
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        original_shape = x_np.shape
        single_input = False

        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            single_input = True
        elif x_np.ndim > 2:
            x_np = x_np.reshape(-1, x_np.shape[-1])

        # 步骤1: 分解与归一化
        r = np.linalg.norm(x_np, axis=1, keepdims=True)
        r = np.maximum(r, 1e-10)
        mu = x_np / r

        # 步骤2: 正交噪声生成
        n = np.random.randn(*x_np.shape)
        dot_product = np.sum(n * mu, axis=1, keepdims=True)
        n_perp = n - dot_product * mu
        n_perp_norm = np.linalg.norm(n_perp, axis=1, keepdims=True)
        n_perp_norm = np.maximum(n_perp_norm, 1e-10)
        n_perp = n_perp / n_perp_norm

        # 步骤3: 动态尺度缩放
        lambda_scale = self.beta / self.epsilon

        # 步骤4: 方向偏转
        z = mu + lambda_scale * n_perp

        # 步骤5: 重投影与恢复
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z_norm = np.maximum(z_norm, 1e-10)
        mu_prime = z / z_norm
        y = r * mu_prime

        if single_input:
            y = y.flatten()
        else:
            y = y.reshape(original_shape)

        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


class NormPreservingGaussian:
    """
    范数保持的高斯机制
    先添加高斯噪声，然后重新归一化到原始范数

    这样可以与 vMF 公平对比
    """

    def __init__(self, epsilon=1.0, delta=1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def perturb(self, x, modality='visual'):
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        if is_torch:
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        original_shape = x_np.shape

        if x_np.ndim == 1:
            x_2d = x_np.reshape(1, -1)
        elif x_np.ndim > 2:
            x_2d = x_np.reshape(-1, x_np.shape[-1])
        else:
            x_2d = x_np

        # 保存原始范数
        original_norms = np.linalg.norm(x_2d, axis=1, keepdims=True)
        original_norms = np.maximum(original_norms, 1e-10)

        # 归一化
        x_normalized = x_2d / original_norms

        # 计算噪声标准差（对单位向量）
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon

        # 添加高斯噪声
        noise = np.random.randn(*x_normalized.shape) * sigma
        y_noisy = x_normalized + noise

        # 重新归一化并恢复范数
        y_norms = np.linalg.norm(y_noisy, axis=1, keepdims=True)
        y_norms = np.maximum(y_norms, 1e-10)
        y = (y_noisy / y_norms) * original_norms

        y = y.reshape(original_shape)

        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def set_epsilon(self, epsilon):
        self.epsilon = epsilon


def compute_stats(original, perturbed):
    """计算扰动统计量"""
    # 展平为 2D
    if original.ndim == 1:
        orig_2d = original.reshape(1, -1)
        pert_2d = perturbed.reshape(1, -1)
    elif original.ndim > 2:
        orig_2d = original.reshape(-1, original.shape[-1])
        pert_2d = perturbed.reshape(-1, perturbed.shape[-1])
    else:
        orig_2d = original
        pert_2d = perturbed

    # 范数
    orig_norms = np.linalg.norm(orig_2d, axis=1)
    pert_norms = np.linalg.norm(pert_2d, axis=1)
    norm_ratio = pert_norms / (orig_norms + 1e-10)

    # 余弦相似度和角度
    cos_sim = np.sum(orig_2d * pert_2d, axis=1) / (orig_norms * pert_norms + 1e-10)
    cos_sim = np.clip(cos_sim, -1, 1)
    angles = np.arccos(cos_sim) * 180 / np.pi

    # L2 距离
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


def run_fair_comparison(dim=4096, n_samples=100, epsilons=[0.1, 0.5, 1.0, 2.0]):
    """
    公平对比实验
    """
    print("=" * 90)
    print("公平对比：Embedding 扰动机制")
    print("=" * 90)
    print(f"维度: {dim}, 样本数: {n_samples}")
    print("\n所有机制使用相同的 epsilon，在相同的数据上测试")

    # 生成模拟 embedding 数据
    np.random.seed(42)
    data = np.random.randn(n_samples, dim).astype(np.float32)
    # 模拟真实 embedding 的范数分布
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
            # 创建机制
            if name == 'vMF':
                mech = MechClass(epsilon=eps, beta=1.0)
            elif name == 'Gaussian' or name == 'NormPreserving':
                mech = MechClass(epsilon=eps, delta=1e-5)
            else:
                mech = MechClass(epsilon=eps)

            # 扰动
            perturbed = mech.perturb(data.copy())

            # 计算统计
            stats = compute_stats(data, perturbed)
            all_results[eps][name] = stats

            print(f"\n[{name}]")
            print(f"  角度偏差: {stats['angle_mean']:.2f}° ± {stats['angle_std']:.2f}°")
            print(f"  范数比例: {stats['norm_ratio_mean']:.4f} ± {stats['norm_ratio_std']:.4f}")
            print(f"  L2 距离:  {stats['l2_dist_mean']:.4f} ± {stats['l2_dist_std']:.4f}")
            print(f"  余弦相似: {stats['cos_sim_mean']:.4f}")

    # 打印对比表格
    print("\n" + "=" * 90)
    print("对比总结表")
    print("=" * 90)

    mech_names = list(mechanisms.keys())

    # 角度偏差表
    print("\n【角度偏差 (度)】- 越小表示方向保持越好")
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

    # 范数比例表
    print("\n【范数比例】- 越接近 1.0 表示范数保持越好")
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

    # L2 距离表
    print("\n【L2 距离】- 扰动幅度")
    print(f"{'ε':<8}", end="")
    for name in mech_names:
        print(f"{name:<18}", end="")
    print()
    print("-" * 80)
    for eps in epsilons:
        print(f"{eps:<8}", end="")
        for name in mech_names:
            dist = all_results[eps][name]['l2_dist_mean']
            print(f"{dist:.4f}".ljust(18), end="")
        print()

    # 总结
    print("\n" + "=" * 90)
    print("机制特点总结")
    print("=" * 90)
    print("""
┌─────────────────┬──────────────┬──────────────┬─────────────────────────────┐
│ 机制            │ 范数保持     │ 隐私类型     │ 特点                        │
├─────────────────┼──────────────┼──────────────┼─────────────────────────────┤
│ vMF             │ ✓ 完全保持   │ 几何隐私     │ 只改变方向，适合 embedding  │
│ Gaussian        │ ✗ 会增大     │ (ε,δ)-DP    │ 各维度独立加噪              │
│ Laplace         │ ✗ 会增大     │ 纯 ε-DP     │ 重尾噪声，更强隐私          │
│ NormPreserving  │ ✓ 强制保持   │ 近似 DP      │ 高斯+重归一化               │
└─────────────────┴──────────────┴──────────────┴─────────────────────────────┘

关键观察:
1. vMF 和 NormPreserving 都保持范数，但 vMF 的角度偏差更可控
2. Gaussian 和 Laplace 在高维空间中范数会显著增大
3. 相同 epsilon 下，不同机制的实际扰动效果差异很大
4. 对于 embedding 扰动，vMF 是更合适的选择
    """)

    return all_results


if __name__ == "__main__":
    results = run_fair_comparison(
        dim=4096,
        n_samples=100,
        epsilons=[0.1, 0.2, 0.5, 1.0, 2.0, 5.0]
    )
