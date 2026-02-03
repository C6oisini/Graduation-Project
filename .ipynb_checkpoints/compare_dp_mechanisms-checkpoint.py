"""
QwenVL + 多种差分隐私机制对比测试
"""

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np

# 导入各种机制
from vmf_core import vMFPerturbation
from dp_mechanisms import GaussianMechanism, LaplaceMechanism, ClippedGaussianMechanism


class MultiMechanismWrapper:
    """
    支持多种差分隐私机制的 QwenVL 包装器
    """

    def __init__(
        self,
        model,
        mechanism='vmf',  # 'vmf', 'gaussian', 'laplace', 'clipped_gaussian', 'none'
        epsilon=1.0,
        delta=1e-5,
        sensitivity=1.0,
        max_norm=1.0,
        beta=1.0,
        alpha=2.0,
        perturb_visual=True,
        perturb_text=True,
        verbose=True
    ):
        self.model = model
        self.mechanism_name = mechanism
        self.epsilon = epsilon
        self.perturb_visual = perturb_visual
        self.perturb_text = perturb_text
        self.verbose = verbose
        self.enabled = True

        # 创建扰动机制
        if mechanism == 'vmf':
            self.visual_perturbation = vMFPerturbation(epsilon=epsilon, beta=beta, alpha=alpha)
            self.text_perturbation = vMFPerturbation(epsilon=epsilon, beta=beta, alpha=alpha)
        elif mechanism == 'gaussian':
            self.visual_perturbation = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
            self.text_perturbation = GaussianMechanism(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        elif mechanism == 'laplace':
            self.visual_perturbation = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
            self.text_perturbation = LaplaceMechanism(epsilon=epsilon, sensitivity=sensitivity)
        elif mechanism == 'clipped_gaussian':
            self.visual_perturbation = ClippedGaussianMechanism(epsilon=epsilon, delta=delta, max_norm=max_norm)
            self.text_perturbation = ClippedGaussianMechanism(epsilon=epsilon, delta=delta, max_norm=max_norm)
        else:
            self.visual_perturbation = None
            self.text_perturbation = None

        # 注册 hooks
        self._hooks = []
        self._setup_hooks()

        if verbose:
            print(f"[DP] 机制: {mechanism}, epsilon={epsilon}")

    def _setup_hooks(self):
        """设置 hooks"""
        self._remove_hooks()

        if self.perturb_visual:
            visual_encoder = self._find_visual_encoder()
            if visual_encoder is not None:
                hook = visual_encoder.register_forward_hook(self._visual_hook)
                self._hooks.append(hook)

        if self.perturb_text:
            text_embedding = self._find_text_embedding()
            if text_embedding is not None:
                hook = text_embedding.register_forward_hook(self._text_hook)
                self._hooks.append(hook)

    def _find_visual_encoder(self):
        """查找视觉编码器"""
        paths = ['model.visual.merger.linear_fc2', 'visual.merger.linear_fc2']
        for path in paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module
        return None

    def _find_text_embedding(self):
        """查找文本嵌入"""
        paths = ['model.language_model.embed_tokens', 'model.embed_tokens']
        for path in paths:
            module = self._get_module_by_path(path)
            if module is not None:
                return module
        return None

    def _get_module_by_path(self, path):
        parts = path.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _visual_hook(self, module, input, output):
        if not self.enabled or self.visual_perturbation is None:
            return output
        return self._apply_perturbation(output, 'visual')

    def _text_hook(self, module, input, output):
        if not self.enabled or self.text_perturbation is None:
            return output
        return self._apply_perturbation(output, 'text')

    def _apply_perturbation(self, output, modality):
        perturbation = self.visual_perturbation if modality == 'visual' else self.text_perturbation

        if isinstance(output, torch.Tensor):
            return self._perturb_tensor(output, perturbation, modality)
        return output

    def _perturb_tensor(self, tensor, perturbation, modality):
        original_shape = tensor.shape
        device = tensor.device
        dtype = tensor.dtype

        # 计算原始统计
        original_norm = torch.norm(tensor).item()

        # 展平
        if tensor.dim() > 2:
            tensor_2d = tensor.view(-1, tensor.shape[-1])
        else:
            tensor_2d = tensor

        # 扰动
        perturbed = perturbation.perturb(tensor_2d, modality=modality)

        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.to(device=device, dtype=dtype)

        # 恢复形状
        if tensor.dim() > 2:
            perturbed = perturbed.view(original_shape)

        # 计算扰动后统计
        perturbed_norm = torch.norm(perturbed).item()

        # 计算角度偏差
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.view(-1).float(), perturbed.view(-1).float(), dim=0
        ).item()
        cos_sim = np.clip(cos_sim, -1, 1)
        angle_deg = np.arccos(cos_sim) * 180 / np.pi

        if self.verbose:
            print(f"[{self.mechanism_name.upper()}] {modality} | "
                  f"shape={original_shape} | "
                  f"norm: {original_norm:.2f}→{perturbed_norm:.2f} | "
                  f"角度偏差: {angle_deg:.1f}°")

        return perturbed

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def enable(self):
        self.enabled = True

    def disable(self):
        self.enabled = False

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __del__(self):
        self._remove_hooks()


def run_comparison():
    """运行对比实验"""
    print("=" * 80)
    print("加载模型...")
    print("=" * 80)

    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "描述一下这个图片."},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )
    inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

    # 测试配置
    epsilon = 10
    mechanisms = [
        ("无扰动", "none", {}),
        ("vMF", "vmf", {"beta": 1.0, "alpha": 2.0}),
        ("高斯机制", "gaussian", {"delta": 1e-5, "sensitivity": 1.0}),
        ("拉普拉斯机制", "laplace", {"sensitivity": 1.0}),
        ("带裁剪高斯", "clipped_gaussian", {"delta": 1e-5, "max_norm": 1.0}),
    ]

    results = []

    for name, mech, kwargs in mechanisms:
        print("\n" + "=" * 80)
        print(f"测试: {name} (epsilon={epsilon})")
        print("=" * 80)

        model = MultiMechanismWrapper(
            base_model,
            mechanism=mech,
            epsilon=epsilon,
            perturb_visual=True,
            perturb_text=True,
            verbose=True,
            **kwargs
        )

        print("\n生成中...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        results.append({
            "name": name,
            "mechanism": mech,
            "output": output_text[0]
        })

        print("\n" + "-" * 60)
        print("输出:")
        print("-" * 60)
        print(output_text[0][:200] + "..." if len(output_text[0]) > 200 else output_text[0])

        # 移除 hooks
        model._remove_hooks()

    # 打印对比
    print("\n" + "=" * 80)
    print("对比总结")
    print("=" * 80)

    baseline = results[0]['output']
    for result in results:
        print(f"\n【{result['name']}】")
        print(f"输出长度: {len(result['output'])} 字符")

        if result['name'] != "无扰动":
            # 简单相似度
            common = sum(1 for a, b in zip(baseline, result['output']) if a == b)
            sim = common / max(len(baseline), len(result['output'])) * 100
            print(f"与原始输出相似度: {sim:.1f}%")

    print("\n" + "=" * 80)
    print("机制特点对比:")
    print("=" * 80)
    print("""
    | 机制           | 范数保持 | 隐私类型      | 适用场景           |
    |----------------|----------|---------------|-------------------|
    | vMF            | ✓ 保持   | 几何隐私      | Embedding 扰动     |
    | 高斯机制       | ✗ 增大   | (ε,δ)-DP     | 通用数值数据       |
    | 拉普拉斯机制   | ✗ 增大   | 纯 ε-DP      | 通用数值数据       |
    | 带裁剪高斯     | ✗ 变化   | (ε,δ)-DP     | DP-SGD 训练        |
    """)


if __name__ == "__main__":
    run_comparison()
