"""
高斯机制 (Gaussian Mechanism)
满足 (ε, δ)-差分隐私
"""

import numpy as np
import torch


class GaussianMechanism:
    """
    高斯机制

    噪声标准差: σ = Δf * √(2 * ln(1.25/δ)) / ε
    其中 Δf 是函数的 L2 敏感度

    针对 embedding 优化：根据输入动态计算敏感度

    参数:
        epsilon: 隐私预算
        delta: 隐私松弛参数
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def perturb(self, x, modality: str = 'visual'):
        """
        对输入添加高斯噪声

        参数:
            x: 输入向量，支持 numpy.ndarray 或 torch.Tensor
            modality: 模态类型（保留参数）

        返回:
            扰动后的向量
        """
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

    def set_epsilon(self, epsilon: float):
        """更新 epsilon"""
        self.epsilon = epsilon

    def get_sigma(self, sensitivity: float = 1.0) -> float:
        """获取给定敏感度下的噪声标准差"""
        return sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
