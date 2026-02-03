"""
模型包装器模块
提供 VLM 模型的隐私保护包装
"""

from .qwenvl_wrapper import QwenVLPrivacyWrapper, wrap_qwenvl

__all__ = [
    'QwenVLPrivacyWrapper',
    'wrap_qwenvl',
]
