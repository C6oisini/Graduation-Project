"""
QwenVL 隐私保护包装器
支持多种差分隐私机制，在 embedding 层后进行扰动
"""

import warnings
import torch
import numpy as np
from typing import Optional, Literal

from privacy import vMFMechanism, GaussianMechanism, LaplaceMechanism, NormPreservingGaussian


MechanismType = Literal['vmf', 'gaussian', 'laplace', 'norm_preserving', 'none']


class QwenVLPrivacyWrapper:
    """
    QwenVL 模型的隐私保护包装器

    支持的机制:
    - vmf: vMF 扰动（推荐，保持范数）
    - gaussian: 高斯机制
    - laplace: 拉普拉斯机制
    - norm_preserving: 范数保持的高斯机制
    - none: 不扰动

    使用示例:
        model = Qwen3VLForConditionalGeneration.from_pretrained(...)
        wrapper = QwenVLPrivacyWrapper(model, mechanism='vmf', epsilon=0.5)
        output = wrapper.generate(**inputs)
    """

    def __init__(
        self,
        model,
        mechanism: MechanismType = 'vmf',
        epsilon: float = 1.0,
        delta: float = 1e-5,
        beta: float = 1.0,
        perturb_visual: bool = True,
        perturb_text: bool = True,
        verbose: bool = False
    ):
        """
        参数:
            model: QwenVL 模型实例
            mechanism: 扰动机制类型
            epsilon: 隐私预算
            delta: 隐私松弛参数（仅高斯机制使用）
            beta: vMF 调节系数
            perturb_visual: 是否扰动视觉 embedding
            perturb_text: 是否扰动文本 embedding
            verbose: 是否输出详细信息
        """
        self.model = model
        self.mechanism_name = mechanism
        self.epsilon = epsilon
        self.perturb_visual = perturb_visual
        self.perturb_text = perturb_text
        self.verbose = verbose
        self.enabled = True

        # 统计信息
        self.stats = {'visual': [], 'text': []}

        # 创建扰动机制
        self.perturbation = self._create_mechanism(mechanism, epsilon, delta, beta)

        # 注册 hooks
        self._hooks = []
        self._setup_hooks()

        if verbose and mechanism != 'none':
            print(f"[Privacy] 机制: {mechanism}, epsilon={epsilon}")

    def _create_mechanism(self, mechanism: str, epsilon: float, delta: float, beta: float):
        """创建扰动机制"""
        if mechanism == 'vmf':
            return vMFMechanism(epsilon=epsilon, beta=beta)
        elif mechanism == 'gaussian':
            return GaussianMechanism(epsilon=epsilon, delta=delta)
        elif mechanism == 'laplace':
            return LaplaceMechanism(epsilon=epsilon)
        elif mechanism == 'norm_preserving':
            return NormPreservingGaussian(epsilon=epsilon, delta=delta)
        else:
            return None

    def _setup_hooks(self):
        """设置 forward hooks"""
        self._remove_hooks()

        if self.perturb_visual:
            visual_module = self._find_module('model.visual.merger.linear_fc2')
            if visual_module:
                hook = visual_module.register_forward_hook(self._visual_hook)
                self._hooks.append(hook)
            else:
                warnings.warn(
                    "[Privacy] 视觉 hook 注册失败: 未找到 model.visual.merger.linear_fc2，"
                    "视觉 embedding 将不会被扰动",
                    RuntimeWarning, stacklevel=2
                )

        if self.perturb_text:
            text_module = self._find_module('model.language_model.embed_tokens')
            if text_module:
                hook = text_module.register_forward_hook(self._text_hook)
                self._hooks.append(hook)
            else:
                warnings.warn(
                    "[Privacy] 文本 hook 注册失败: 未找到 model.language_model.embed_tokens，"
                    "文本 embedding 将不会被扰动",
                    RuntimeWarning, stacklevel=2
                )

    def _find_module(self, path: str):
        """通过路径查找模块"""
        parts = path.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _visual_hook(self, module, input, output):
        """视觉 embedding hook"""
        if not self.enabled or self.perturbation is None:
            return output
        return self._apply_perturbation(output, 'visual')

    def _text_hook(self, module, input, output):
        """文本 embedding hook"""
        if not self.enabled or self.perturbation is None:
            return output
        return self._apply_perturbation(output, 'text')

    def _apply_perturbation(self, output, modality: str):
        """应用扰动"""
        if not isinstance(output, torch.Tensor):
            return output

        original_shape = output.shape
        device = output.device
        dtype = output.dtype

        # 扰动
        perturbed = self.perturbation.perturb(output, modality=modality)

        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.to(device=device, dtype=dtype)

        # 逐 embedding 统计（展平为 [n_vectors, dim]）
        flat_orig = output.detach().float().reshape(-1, output.shape[-1])
        flat_pert = perturbed.detach().float().reshape(-1, perturbed.shape[-1])

        orig_norms = torch.norm(flat_orig, dim=1)
        pert_norms = torch.norm(flat_pert, dim=1)
        norm_ratio = (pert_norms / (orig_norms + 1e-10)).mean().item()

        cos_sim = torch.nn.functional.cosine_similarity(flat_orig, flat_pert, dim=1)
        cos_sim = cos_sim.clamp(-1, 1)
        angle_deg = torch.acos(cos_sim).mean().item() * 180 / np.pi

        self.stats[modality].append({
            'original_norm': orig_norms.mean().item(),
            'perturbed_norm': pert_norms.mean().item(),
            'norm_ratio': norm_ratio,
            'angle': angle_deg,
        })

        if self.verbose:
            print(f"[{self.mechanism_name.upper()}] {modality} | "
                  f"shape={original_shape} | "
                  f"norm: {orig_norms.mean().item():.2f}→{pert_norms.mean().item():.2f} | "
                  f"角度: {angle_deg:.1f}°")

        return perturbed

    def get_stats_summary(self) -> dict:
        """获取统计摘要"""
        summary = {}
        for modality in ['visual', 'text']:
            if self.stats[modality]:
                angles = [s['angle'] for s in self.stats[modality]]
                ratios = [s['norm_ratio'] for s in self.stats[modality]]
                summary[modality] = {
                    'angle_mean': np.mean(angles),
                    'angle_std': np.std(angles),
                    'norm_ratio_mean': np.mean(ratios),
                    'norm_ratio_std': np.std(ratios),
                }
        return summary

    def reset_stats(self):
        """重置统计"""
        self.stats = {'visual': [], 'text': []}

    def set_epsilon(self, epsilon: float):
        """动态调整隐私预算"""
        self.epsilon = epsilon
        if self.perturbation:
            self.perturbation.set_epsilon(epsilon)

    def enable(self):
        """启用扰动"""
        self.enabled = True

    def disable(self):
        """禁用扰动"""
        self.enabled = False

    def _remove_hooks(self):
        """移除所有 hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __getattr__(self, name):
        """代理到原始模型"""
        # 防止递归：如果访问的是 wrapper 自身的属性，直接抛出 AttributeError
        if name in ('model', '_hooks', 'perturbation', 'enabled', 'stats'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        """调用原始模型"""
        return self.model(*args, **kwargs)

    def __del__(self):
        """析构时移除 hooks"""
        self._remove_hooks()


def wrap_qwenvl(
    model,
    mechanism: MechanismType = 'vmf',
    epsilon: float = 1.0,
    **kwargs
) -> QwenVLPrivacyWrapper:
    """
    便捷函数：为 QwenVL 模型添加隐私保护

    使用示例:
        model = Qwen3VLForConditionalGeneration.from_pretrained(...)
        model = wrap_qwenvl(model, mechanism='vmf', epsilon=0.5)

    参数:
        model: QwenVL 模型
        mechanism: 扰动机制 ('vmf', 'gaussian', 'laplace', 'norm_preserving', 'none')
        epsilon: 隐私预算
        **kwargs: 其他参数传递给 QwenVLPrivacyWrapper

    返回:
        包装后的模型
    """
    return QwenVLPrivacyWrapper(
        model=model,
        mechanism=mechanism,
        epsilon=epsilon,
        **kwargs
    )
