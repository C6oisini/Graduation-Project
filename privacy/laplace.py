"""
拉普拉斯机制 (Laplace Mechanism)
满足 ε-差分隐私（纯差分隐私）
"""

import numpy as np
import torch


class LaplaceMechanism:
    """
    拉普拉斯机制

    噪声尺度: b = Δf / ε
    其中 Δf 是函数的 L1 敏感度

    针对 embedding 优化：根据输入动态计算敏感度

    参数:
        epsilon: 隐私预算
    """

    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def perturb(self, x, modality: str = 'visual'):
        """
        对输入添加拉普拉斯噪声

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

    def set_epsilon(self, epsilon: float):
        """更新 epsilon"""
        self.epsilon = epsilon

    def get_scale(self, sensitivity: float = 1.0) -> float:
        """获取给定敏感度下的噪声尺度"""
        return sensitivity / self.epsilon
