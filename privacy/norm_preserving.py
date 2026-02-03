"""
范数保持的高斯机制 (Norm-Preserving Gaussian)
先添加高斯噪声，然后重新归一化到原始范数
"""

import numpy as np
import torch


class NormPreservingGaussian:
    """
    范数保持的高斯机制

    算法步骤:
    1. 保存原始范数
    2. 归一化到单位向量
    3. 添加高斯噪声
    4. 重新归一化
    5. 恢复原始范数

    这样可以与 vMF 公平对比，都保持范数不变

    参数:
        epsilon: 隐私预算
        delta: 隐私松弛参数
    """

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def perturb(self, x, modality: str = 'visual'):
        """
        对输入进行范数保持的高斯扰动

        参数:
            x: 输入向量
            modality: 模态类型（保留参数）

        返回:
            扰动后的向量（范数与原始相同）
        """
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

    def set_epsilon(self, epsilon: float):
        """更新 epsilon"""
        self.epsilon = epsilon
