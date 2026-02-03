"""
von Mises-Fisher 语义感知扰动算法实现
基于正交切平面投影的几何扰动方法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from mpl_toolkits.mplot3d import Axes3D


class vMFPerturbation:
    """基于几何投影的语义感知扰动算法"""

    def __init__(self, epsilon=1.0, beta=1.0, alpha=2.0):
        """
        参数:
            epsilon: 隐私预算，越小隐私保护越强
            beta: 调节系数（超参数）
            alpha: 非对称分配因子，用于文本通道
        """
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha

    def perturb(self, x, modality='visual'):
        """
        对输入向量进行扰动

        参数:
            x: 输入向量，shape为 (n_samples, d) 或 (d,)
            modality: 模态类型，'visual' 或 'text'

        返回:
            扰动后的向量
        """
        # 确保输入是2D
        single_input = False
        if x.ndim == 1:
            x = x.reshape(1, -1)
            single_input = True

        # 根据模态选择隐私预算
        if modality == 'text':
            eps = self.alpha * self.epsilon
        else:
            eps = self.epsilon

        # 步骤1: 分解与归一化
        r = np.linalg.norm(x, axis=1, keepdims=True)  # 模长
        # 避免除零
        r = np.maximum(r, 1e-10)
        mu = x / r  # 单位方向向量

        # 步骤2: 正交噪声生成
        n = np.random.randn(*x.shape)  # 标准正态噪声
        # 投影到切平面: n_perp = n - (n·mu)*mu
        dot_product = np.sum(n * mu, axis=1, keepdims=True)
        n_perp = n - dot_product * mu
        # 归一化正交噪声
        n_perp_norm = np.linalg.norm(n_perp, axis=1, keepdims=True)
        n_perp_norm = np.maximum(n_perp_norm, 1e-10)
        n_perp = n_perp / n_perp_norm

        # 步骤3: 动态尺度缩放
        lambda_scale = self.beta / eps

        # 步骤4: 方向偏转
        z = mu + lambda_scale * n_perp

        # 步骤5: 重投影与恢复
        z_norm = np.linalg.norm(z, axis=1, keepdims=True)
        z_norm = np.maximum(z_norm, 1e-10)
        mu_prime = z / z_norm  # 扰动后的方向
        y = r * mu_prime  # 恢复模长

        if single_input:
            return y.flatten()
        return y

    def perturb_with_mask(self, x, mask):
        """
        带掩码的扰动（用于文本敏感区保护）

        参数:
            x: 输入向量
            mask: 掩码矩阵，1表示扰动，0表示保留原样

        返回:
            扰动后的向量
        """
        y_perturbed = self.perturb(x, modality='text')
        mask = mask.reshape(-1, 1)
        y = mask * y_perturbed + (1 - mask) * x
        return y


def generate_spherical_clusters(n_clusters=3, n_samples_per_cluster=100, d=3, spread=0.1, kappa=None):
    """
    在单位超球面上生成聚类数据（使用vMF采样）

    参数:
        n_clusters: 聚类数量
        n_samples_per_cluster: 每个聚类的样本数
        d: 维度
        spread: 聚类的分散程度（低维使用）
        kappa: vMF集中度参数（高维使用），越大越集中

    返回:
        data: 数据点
        labels: 真实标签
        centers: 聚类中心
    """
    data = []
    labels = []
    centers = []

    # 高维空间需要使用vMF采样来生成紧凑的簇
    use_vmf = d > 10

    if use_vmf and kappa is None:
        # 高维空间中，kappa需要足够大才能形成紧凑的簇
        # 经验公式：kappa与维度成正比
        kappa = d * 2  # 集中度参数

    # 生成随机的聚类中心（在单位球面上）
    for i in range(n_clusters):
        center = np.random.randn(d)
        center = center / np.linalg.norm(center)
        centers.append(center)

        # 在中心周围生成样本
        for _ in range(n_samples_per_cluster):
            if use_vmf:
                # 使用vMF采样：Wood's algorithm的简化版本
                # 在切平面上生成噪声，然后投影
                noise = np.random.randn(d)
                noise = noise - np.dot(noise, center) * center  # 正交化
                noise = noise / np.linalg.norm(noise)  # 归一化

                # 采样角度偏移（近似vMF）
                # 对于大kappa，角度集中在0附近
                angle = np.random.vonmises(0, kappa)  # 1D von Mises
                angle = np.abs(angle)  # 取正值

                # 构造点：沿着center方向cos(angle)，沿着noise方向sin(angle)
                point = np.cos(angle) * center + np.sin(angle) * noise
                point = point / np.linalg.norm(point)  # 确保在球面上
            else:
                # 低维使用简单的高斯噪声
                noise = np.random.randn(d) * spread
                point = center + noise
                point = point / np.linalg.norm(point)

            data.append(point)
            labels.append(i)

    return np.array(data), np.array(labels), np.array(centers)


def evaluate_clustering(data, true_labels, n_clusters):
    """评估聚类效果"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    pred_labels = kmeans.fit_predict(data)

    ari = adjusted_rand_score(true_labels, pred_labels)
    nmi = normalized_mutual_info_score(true_labels, pred_labels)

    return ari, nmi, pred_labels


def visualize_3d(original_data, perturbed_data, true_labels, title_suffix=""):
    """3D可视化"""
    fig = plt.figure(figsize=(14, 6))

    # 原始数据
    ax1 = fig.add_subplot(121, projection='3d')
    scatter1 = ax1.scatter(original_data[:, 0], original_data[:, 1], original_data[:, 2],
                           c=true_labels, cmap='viridis', alpha=0.6)
    ax1.set_title(f'Original Data {title_suffix}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # 扰动后数据
    ax2 = fig.add_subplot(122, projection='3d')
    scatter2 = ax2.scatter(perturbed_data[:, 0], perturbed_data[:, 1], perturbed_data[:, 2],
                           c=true_labels, cmap='viridis', alpha=0.6)
    ax2.set_title(f'Perturbed Data {title_suffix}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    plt.tight_layout()
    return fig


def run_experiment():
    """运行完整实验"""
    print("=" * 60)
    print("vMF 语义感知扰动算法 - 聚类测试")
    print("=" * 60)

    # 参数设置
    n_clusters = 4
    n_samples_per_cluster = 100
    d = 3  # 使用3维便于可视化
    spread = 0.15

    # 生成数据
    print(f"\n生成数据: {n_clusters}个聚类, 每个{n_samples_per_cluster}个样本, {d}维")
    data, true_labels, centers = generate_spherical_clusters(
        n_clusters=n_clusters,
        n_samples_per_cluster=n_samples_per_cluster,
        d=d,
        spread=spread
    )

    # 原始数据聚类评估
    ari_orig, nmi_orig, _ = evaluate_clustering(data, true_labels, n_clusters)
    print(f"\n原始数据聚类效果:")
    print(f"  ARI (Adjusted Rand Index): {ari_orig:.4f}")
    print(f"  NMI (Normalized Mutual Info): {nmi_orig:.4f}")

    # 测试不同隐私预算
    epsilons = [0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    print("\n" + "-" * 60)
    print("不同隐私预算下的聚类效果:")
    print("-" * 60)
    print(f"{'Epsilon':<10} {'ARI':<12} {'NMI':<12} {'角度偏差(度)':<15}")
    print("-" * 60)

    for eps in epsilons:
        perturbator = vMFPerturbation(epsilon=eps, beta=1.0)
        perturbed_data = perturbator.perturb(data.copy(), modality='visual')

        # 聚类评估
        ari, nmi, _ = evaluate_clustering(perturbed_data, true_labels, n_clusters)

        # 计算平均角度偏差
        cos_sim = np.sum(data * perturbed_data, axis=1)
        cos_sim = np.clip(cos_sim, -1, 1)
        angles = np.arccos(cos_sim) * 180 / np.pi
        mean_angle = np.mean(angles)

        results.append({
            'epsilon': eps,
            'ari': ari,
            'nmi': nmi,
            'mean_angle': mean_angle
        })

        print(f"{eps:<10.1f} {ari:<12.4f} {nmi:<12.4f} {mean_angle:<15.2f}")

    # 可视化
    print("\n生成可视化图表...")

    # 选择两个代表性的epsilon进行可视化
    fig1 = plt.figure(figsize=(16, 12))

    eps_to_show = [0.5, 2.0, 10.0]
    for idx, eps in enumerate(eps_to_show):
        perturbator = vMFPerturbation(epsilon=eps, beta=1.0)
        perturbed_data = perturbator.perturb(data.copy(), modality='visual')

        # 原始数据
        ax1 = fig1.add_subplot(2, 3, idx + 1, projection='3d')
        ax1.scatter(data[:, 0], data[:, 1], data[:, 2],
                   c=true_labels, cmap='viridis', alpha=0.6, s=20)
        ax1.set_title(f'Original (ε={eps})')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        # 扰动后数据
        ax2 = fig1.add_subplot(2, 3, idx + 4, projection='3d')
        ax2.scatter(perturbed_data[:, 0], perturbed_data[:, 1], perturbed_data[:, 2],
                   c=true_labels, cmap='viridis', alpha=0.6, s=20)
        ax2.set_title(f'Perturbed (ε={eps})')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')

    plt.suptitle('vMF Perturbation: Effect of Privacy Budget ε', fontsize=14)
    plt.tight_layout()
    plt.savefig('F:/HNUST/vmf_3d_visualization.png', dpi=150, bbox_inches='tight')
    print("  保存: vmf_3d_visualization.png")

    # 绘制性能曲线
    fig2, axes = plt.subplots(1, 3, figsize=(15, 4))

    eps_vals = [r['epsilon'] for r in results]
    ari_vals = [r['ari'] for r in results]
    nmi_vals = [r['nmi'] for r in results]
    angle_vals = [r['mean_angle'] for r in results]

    # ARI曲线
    axes[0].plot(eps_vals, ari_vals, 'bo-', linewidth=2, markersize=8)
    axes[0].axhline(y=ari_orig, color='r', linestyle='--', label='Original')
    axes[0].set_xlabel('Privacy Budget (ε)')
    axes[0].set_ylabel('ARI')
    axes[0].set_title('Adjusted Rand Index vs ε')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # NMI曲线
    axes[1].plot(eps_vals, nmi_vals, 'go-', linewidth=2, markersize=8)
    axes[1].axhline(y=nmi_orig, color='r', linestyle='--', label='Original')
    axes[1].set_xlabel('Privacy Budget (ε)')
    axes[1].set_ylabel('NMI')
    axes[1].set_title('Normalized Mutual Info vs ε')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # 角度偏差曲线
    axes[2].plot(eps_vals, angle_vals, 'mo-', linewidth=2, markersize=8)
    axes[2].set_xlabel('Privacy Budget (ε)')
    axes[2].set_ylabel('Mean Angular Deviation (degrees)')
    axes[2].set_title('Angular Deviation vs ε')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('F:/HNUST/vmf_performance_curves.png', dpi=150, bbox_inches='tight')
    print("  保存: vmf_performance_curves.png")

    # 高维测试
    print("\n" + "=" * 60)
    print("高维空间测试 (d=4096, 模拟Qwen-VL Embedding维度)")
    print("=" * 60)

    d_high = 4096
    data_high, labels_high, _ = generate_spherical_clusters(
        n_clusters=5,
        n_samples_per_cluster=200,
        d=d_high,
        spread=0.2
    )

    ari_orig_high, nmi_orig_high, _ = evaluate_clustering(data_high, labels_high, 5)
    print(f"\n原始数据: ARI={ari_orig_high:.4f}, NMI={nmi_orig_high:.4f}")

    print(f"\n{'Epsilon':<10} {'ARI':<12} {'NMI':<12} {'角度偏差(度)':<15}")
    print("-" * 50)

    for eps in [0.5, 1.0, 2.0, 5.0]:
        perturbator = vMFPerturbation(epsilon=eps, beta=1.0)
        perturbed_high = perturbator.perturb(data_high.copy())

        ari, nmi, _ = evaluate_clustering(perturbed_high, labels_high, 5)

        cos_sim = np.sum(data_high * perturbed_high, axis=1)
        cos_sim = np.clip(cos_sim, -1, 1)
        angles = np.arccos(cos_sim) * 180 / np.pi
        mean_angle = np.mean(angles)

        print(f"{eps:<10.1f} {ari:<12.4f} {nmi:<12.4f} {mean_angle:<15.2f}")

    print("\n" + "=" * 60)
    print("实验完成!")
    print("=" * 60)

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    run_experiment()
