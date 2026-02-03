"""
vMF (von Mises-Fisher) 扰动核心算法
基于正交切平面投影的几何扰动方法
"""

import numpy as np
import torch


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
            x: 输入向量，支持 numpy.ndarray 或 torch.Tensor
               shape为 (n_samples, d) 或 (d,)
            modality: 模态类型，'visual' 或 'text'

        返回:
            扰动后的向量，类型与输入相同
        """
        # 检测输入类型
        is_torch = isinstance(x, torch.Tensor)
        device = x.device if is_torch else None
        dtype = x.dtype if is_torch else None

        # 转换为 numpy
        if is_torch:
            x_np = x.detach().cpu().numpy()
        else:
            x_np = x

        # 确保输入是2D
        single_input = False
        if x_np.ndim == 1:
            x_np = x_np.reshape(1, -1)
            single_input = True

        # 根据模态选择隐私预算
        if modality == 'text':
            eps = self.alpha * self.epsilon
        else:
            eps = self.epsilon

        # 步骤1: 分解与归一化
        r = np.linalg.norm(x_np, axis=1, keepdims=True)  # 模长
        # 避免除零
        r = np.maximum(r, 1e-10)
        mu = x_np / r  # 单位方向向量

        # 步骤2: 正交噪声生成
        n = np.random.randn(*x_np.shape)  # 标准正态噪声
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
            y = y.flatten()

        # 转换回原始类型
        if is_torch:
            y = torch.from_numpy(y).to(device=device, dtype=dtype)

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
        is_torch = isinstance(x, torch.Tensor)

        if is_torch:
            mask_np = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
            x_np = x.detach().cpu().numpy()
        else:
            mask_np = mask
            x_np = x

        y_perturbed = self.perturb(x_np, modality='text')

        if is_torch:
            y_perturbed = torch.from_numpy(y_perturbed).to(device=x.device, dtype=x.dtype)

        mask_reshaped = mask_np.reshape(-1, 1)
        y = mask_reshaped * y_perturbed + (1 - mask_reshaped) * x_np

        if is_torch:
            y = torch.from_numpy(y).to(device=x.device, dtype=x.dtype)

        return y
