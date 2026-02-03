"""
von Mises-Fisher (vMF) 扰动机制
基于正交切平面投影的几何扰动方法

特点：保持向量范数，只改变方向，适合 embedding 扰动
"""

import numpy as np
import torch


class vMFMechanism:
    """
    von Mises-Fisher 扰动机制

    算法步骤:
    1. 分解向量为模长 r 和方向 μ
    2. 在切平面上生成正交噪声
    3. 根据 epsilon 缩放噪声
    4. 偏转方向并重投影到单位球面
    5. 恢复原始模长

    参数:
        epsilon: 隐私预算，越小隐私保护越强
        beta: 调节系数，控制扰动幅度
    """

    def __init__(self, epsilon: float = 1.0, beta: float = 1.0):
        self.epsilon = epsilon
        self.beta = beta

    def perturb(self, x, modality: str = 'visual'):
        """
        对输入向量进行 vMF 扰动

        参数:
            x: 输入向量，支持 numpy.ndarray 或 torch.Tensor
               shape 为 (n_samples, d) 或 (d,) 或 (batch, seq, d)
            modality: 模态类型（保留参数，用于兼容性）

        返回:
            扰动后的向量，类型和形状与输入相同
        """
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        if is_torch:
            x_np = x.detach().cpu().float().numpy()
        else:
            x_np = np.asarray(x, dtype=np.float32)

        original_shape = x_np.shape
        single_input = False

        # 展平为 2D: [n_vectors, dim]
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            single_input = True
        elif x_np.ndim > 2:
            x_np = x_np.reshape(-1, x_np.shape[-1])

        # 步骤1: 分解与归一化
        r = np.linalg.norm(x_np, axis=1, keepdims=True)  # 模长
        r = np.maximum(r, 1e-10)  # 避免除零
        mu = x_np / r  # 单位方向向量

        # 步骤2: 正交噪声生成
        n = np.random.randn(*x_np.shape)  # 标准正态噪声
        # 投影到切平面: n_perp = n - (n·μ)*μ
        dot_product = np.sum(n * mu, axis=1, keepdims=True)
        n_perp = n - dot_product * mu
        # 归一化正交噪声
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
        mu_prime = z / z_norm  # 扰动后的方向
        y = r * mu_prime  # 恢复模长

        # 恢复原始形状
        if single_input:
            y = y.flatten()
        else:
            y = y.reshape(original_shape)

        # 转换回原始类型
        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

        return y

    def set_epsilon(self, epsilon: float):
        """动态调整隐私预算"""
        self.epsilon = epsilon

    def get_theoretical_angle(self) -> float:
        """获取理论角度偏差（度）"""
        lambda_scale = self.beta / self.epsilon
        return np.arctan(lambda_scale) * 180 / np.pi
