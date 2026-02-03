"""
QwenVL + vMF 扰动集成模块
实现即插即用的隐私保护功能
"""

import torch
import torch.nn as nn
from typing import Optional, Union, List, Dict, Any
from vmf_core import vMFPerturbation


class vMFEmbeddingWrapper(nn.Module):
    """
    vMF 扰动 Embedding Wrapper
    可以包装任意 embedding 层，在输出后自动添加 vMF 扰动
    """

    def __init__(
        self,
        embedding_module: nn.Module,
        epsilon: float = 1.0,
        beta: float = 1.0,
        alpha: float = 2.0,
        modality: str = 'visual',
        enabled: bool = True
    ):
        """
        参数:
            embedding_module: 原始的 embedding 模块
            epsilon: 隐私预算，越小隐私保护越强
            beta: 调节系数
            alpha: 非对称分配因子
            modality: 模态类型 ('visual' 或 'text')
            enabled: 是否启用扰动
        """
        super().__init__()
        self.embedding = embedding_module
        self.perturbation = vMFPerturbation(epsilon=epsilon, beta=beta, alpha=alpha)
        self.modality = modality
        self.enabled = enabled

    def forward(self, *args, **kwargs):
        """前向传播，在 embedding 输出后添加扰动"""
        # 获取原始 embedding 输出
        output = self.embedding(*args, **kwargs)

        if not self.enabled or not self.training:
            return output

        # 对输出进行扰动
        if isinstance(output, torch.Tensor):
            return self._perturb_tensor(output)
        elif isinstance(output, tuple):
            # 处理返回多个值的情况
            return tuple(
                self._perturb_tensor(o) if isinstance(o, torch.Tensor) else o
                for o in output
            )
        elif isinstance(output, dict):
            # 处理返回字典的情况
            return {
                k: self._perturb_tensor(v) if isinstance(v, torch.Tensor) else v
                for k, v in output.items()
            }
        return output

    def _perturb_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """对单个 tensor 进行扰动"""
        original_shape = tensor.shape

        # 将 tensor 展平为 2D (batch, features)
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor

        # 应用 vMF 扰动
        perturbed = self.perturbation.perturb(tensor_2d, modality=self.modality)

        # 恢复原始形状
        if tensor.dim() > 2:
            perturbed = perturbed.view(original_shape)

        return perturbed

    def set_epsilon(self, epsilon: float):
        """动态调整隐私预算"""
        self.perturbation.epsilon = epsilon

    def enable(self):
        """启用扰动"""
        self.enabled = True

    def disable(self):
        """禁用扰动"""
        self.enabled = False


class QwenVLvMFWrapper:
    """
    QwenVL 模型的 vMF 扰动包装器
    即插即用，自动在 embedding 层后添加扰动
    """

    def __init__(
        self,
        model,
        epsilon: float = 1.0,
        beta: float = 1.0,
        alpha: float = 2.0,
        perturb_visual: bool = True,
        perturb_text: bool = True,
        enabled: bool = True
    ):
        """
        参数:
            model: QwenVL 模型实例
            epsilon: 隐私预算
            beta: 调节系数
            alpha: 非对称分配因子
            perturb_visual: 是否扰动视觉 embedding
            perturb_text: 是否扰动文本 embedding
            enabled: 是否启用扰动
        """
        self.model = model
        self.epsilon = epsilon
        self.beta = beta
        self.alpha = alpha
        self.perturb_visual = perturb_visual
        self.perturb_text = perturb_text
        self.enabled = enabled

        # 创建扰动器
        self.visual_perturbation = vMFPerturbation(epsilon=epsilon, beta=beta, alpha=alpha)
        self.text_perturbation = vMFPerturbation(epsilon=epsilon, beta=beta, alpha=alpha)

        # 注册 hooks
        self._hooks = []
        self._setup_hooks()

    def _setup_hooks(self):
        """设置 forward hooks 来拦截 embedding 输出"""
        # 清除旧的 hooks
        self._remove_hooks()

        # 查找并注册 visual encoder 的 hook
        if self.perturb_visual:
            visual_encoder = self._find_visual_encoder()
            if visual_encoder is not None:
                hook = visual_encoder.register_forward_hook(self._visual_hook)
                self._hooks.append(hook)

        # 查找并注册 text embedding 的 hook
        if self.perturb_text:
            text_embedding = self._find_text_embedding()
            if text_embedding is not None:
                hook = text_embedding.register_forward_hook(self._text_hook)
                self._hooks.append(hook)

    def _find_visual_encoder(self):
        """查找视觉编码器模块"""
        # Qwen2-VL / Qwen3-VL 的视觉编码器路径
        possible_paths = [
            'visual',
            'vision_model',
            'vision_tower',
            'visual_encoder',
            'model.visual',
            'model.vision_model',
        ]

        for path in possible_paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module

        # 遍历查找
        for name, module in self.model.named_modules():
            if 'visual' in name.lower() or 'vision' in name.lower():
                if hasattr(module, 'forward'):
                    return module

        return None

    def _find_text_embedding(self):
        """查找文本 embedding 模块"""
        possible_paths = [
            'model.embed_tokens',
            'embed_tokens',
            'model.model.embed_tokens',
            'transformer.wte',
            'model.transformer.wte',
        ]

        for path in possible_paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module

        return None

    def _get_module_by_path(self, path: str):
        """通过路径获取模块"""
        parts = path.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _visual_hook(self, module, input, output):
        """视觉 embedding 的 hook"""
        if not self.enabled:
            return output

        return self._apply_perturbation(output, 'visual')

    def _text_hook(self, module, input, output):
        """文本 embedding 的 hook"""
        if not self.enabled:
            return output

        return self._apply_perturbation(output, 'text')

    def _apply_perturbation(self, output, modality: str):
        """应用扰动到输出"""
        perturbation = self.visual_perturbation if modality == 'visual' else self.text_perturbation

        if isinstance(output, torch.Tensor):
            return self._perturb_tensor(output, perturbation, modality)
        elif isinstance(output, tuple):
            return tuple(
                self._perturb_tensor(o, perturbation, modality) if isinstance(o, torch.Tensor) else o
                for o in output
            )
        elif hasattr(output, 'last_hidden_state'):
            # 处理 BaseModelOutput 类型
            output.last_hidden_state = self._perturb_tensor(
                output.last_hidden_state, perturbation, modality
            )
            return output

        return output

    def _perturb_tensor(self, tensor: torch.Tensor, perturbation: vMFPerturbation, modality: str) -> torch.Tensor:
        """对 tensor 进行扰动"""
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        # 展平为 2D
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor

        # 应用扰动
        perturbed = perturbation.perturb(tensor_2d, modality=modality)

        # 确保类型和设备一致
        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.to(device=device, dtype=dtype)

        # 恢复形状
        if tensor.dim() > 2:
            perturbed = perturbed.view(original_shape)

        return perturbed

    def _remove_hooks(self):
        """移除所有 hooks"""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def set_epsilon(self, epsilon: float):
        """设置隐私预算"""
        self.epsilon = epsilon
        self.visual_perturbation.epsilon = epsilon
        self.text_perturbation.epsilon = epsilon

    def enable(self):
        """启用扰动"""
        self.enabled = True

    def disable(self):
        """禁用扰动"""
        self.enabled = False

    def __del__(self):
        """析构时移除 hooks"""
        self._remove_hooks()

    def __getattr__(self, name):
        """代理到原始模型"""
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        """调用原始模型"""
        return self.model(*args, **kwargs)


def wrap_qwenvl_with_vmf(
    model,
    epsilon: float = 1.0,
    beta: float = 1.0,
    alpha: float = 2.0,
    perturb_visual: bool = True,
    perturb_text: bool = True
) -> QwenVLvMFWrapper:
    """
    便捷函数：为 QwenVL 模型添加 vMF 扰动

    使用示例:
        model = Qwen3VLForConditionalGeneration.from_pretrained(...)
        model = wrap_qwenvl_with_vmf(model, epsilon=1.0)

    参数:
        model: QwenVL 模型
        epsilon: 隐私预算
        beta: 调节系数
        alpha: 非对称分配因子
        perturb_visual: 是否扰动视觉 embedding
        perturb_text: 是否扰动文本 embedding

    返回:
        包装后的模型
    """
    return QwenVLvMFWrapper(
        model=model,
        epsilon=epsilon,
        beta=beta,
        alpha=alpha,
        perturb_visual=perturb_visual,
        perturb_text=perturb_text
    )
