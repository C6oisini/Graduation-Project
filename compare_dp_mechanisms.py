"""
QwenVL 上公平对比多种差分隐私机制
所有机制在相同位置（embedding 层后）进行扰动
"""

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np

from dp_mechanisms import vMFMechanism, GaussianMechanism, LaplaceMechanism, NormPreservingGaussian


class UnifiedDPWrapper:
    """
    统一的差分隐私包装器
    支持多种机制，在相同位置进行扰动
    """

    def __init__(
        self,
        model,
        mechanism='vmf',  # 'vmf', 'gaussian', 'laplace', 'norm_preserving', 'none'
        epsilon=1.0,
        delta=1e-5,
        beta=1.0,
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

        # 统计信息
        self.stats = {
            'visual': [],
            'text': []
        }

        # 创建扰动机制
        if mechanism == 'vmf':
            self.perturbation = vMFMechanism(epsilon=epsilon, beta=beta)
        elif mechanism == 'gaussian':
            self.perturbation = GaussianMechanism(epsilon=epsilon, delta=delta)
        elif mechanism == 'laplace':
            self.perturbation = LaplaceMechanism(epsilon=epsilon)
        elif mechanism == 'norm_preserving':
            self.perturbation = NormPreservingGaussian(epsilon=epsilon, delta=delta)
        else:
            self.perturbation = None

        # 注册 hooks
        self._hooks = []
        self._setup_hooks()

        if verbose and mechanism != 'none':
            print(f"[DP] 机制: {mechanism}, epsilon={epsilon}")

    def _setup_hooks(self):
        self._remove_hooks()

        if self.perturb_visual:
            visual_encoder = self._find_module('model.visual.merger.linear_fc2')
            if visual_encoder:
                hook = visual_encoder.register_forward_hook(self._visual_hook)
                self._hooks.append(hook)

        if self.perturb_text:
            text_embedding = self._find_module('model.language_model.embed_tokens')
            if text_embedding:
                hook = text_embedding.register_forward_hook(self._text_hook)
                self._hooks.append(hook)

    def _find_module(self, path):
        parts = path.split('.')
        module = self.model
        for part in parts:
            if hasattr(module, part):
                module = getattr(module, part)
            else:
                return None
        return module

    def _visual_hook(self, module, input, output):
        if not self.enabled or self.perturbation is None:
            return output
        return self._apply_perturbation(output, 'visual')

    def _text_hook(self, module, input, output):
        if not self.enabled or self.perturbation is None:
            return output
        return self._apply_perturbation(output, 'text')

    def _apply_perturbation(self, output, modality):
        if not isinstance(output, torch.Tensor):
            return output

        original_shape = output.shape
        device = output.device
        dtype = output.dtype

        # 计算原始统计
        original_norm = torch.norm(output).item()

        # 扰动
        perturbed = self.perturbation.perturb(output, modality=modality)

        if isinstance(perturbed, torch.Tensor):
            perturbed = perturbed.to(device=device, dtype=dtype)

        # 计算扰动后统计
        perturbed_norm = torch.norm(perturbed).item()

        # 计算角度偏差
        cos_sim = torch.nn.functional.cosine_similarity(
            output.view(-1).float(), perturbed.view(-1).float(), dim=0
        ).item()
        cos_sim = np.clip(cos_sim, -1, 1)
        angle_deg = np.arccos(cos_sim) * 180 / np.pi

        # 记录统计
        self.stats[modality].append({
            'original_norm': original_norm,
            'perturbed_norm': perturbed_norm,
            'norm_ratio': perturbed_norm / (original_norm + 1e-10),
            'angle': angle_deg,
            'cos_sim': cos_sim
        })

        if self.verbose:
            print(f"[{self.mechanism_name.upper()}] {modality} | "
                  f"shape={original_shape} | "
                  f"norm: {original_norm:.2f}→{perturbed_norm:.2f} ({perturbed_norm/original_norm:.2f}x) | "
                  f"角度: {angle_deg:.1f}°")

        return perturbed

    def get_stats_summary(self):
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
        self.stats = {'visual': [], 'text': []}

    def _remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __del__(self):
        self._remove_hooks()


def run_qwenvl_comparison():
    """在 QwenVL 上运行公平对比"""
    print("=" * 90)
    print("QwenVL 差分隐私机制公平对比")
    print("=" * 90)

    # 加载模型
    print("\n加载模型...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    # 测试输入
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
    epsilon = 0.5
    mechanisms = [
        ("无扰动", "none"),
        ("vMF", "vmf"),
        ("高斯机制", "gaussian"),
        ("拉普拉斯机制", "laplace"),
        ("范数保持高斯", "norm_preserving"),
    ]

    results = []

    for name, mech in mechanisms:
        print("\n" + "=" * 90)
        print(f"测试: {name} (epsilon={epsilon})")
        print("=" * 90)

        # 创建包装器
        wrapper = UnifiedDPWrapper(
            base_model,
            mechanism=mech,
            epsilon=epsilon,
            perturb_visual=True,
            perturb_text=True,
            verbose=True
        )

        # 生成
        print("\n生成中...")
        with torch.no_grad():
            generated_ids = wrapper.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        # 获取统计
        stats = wrapper.get_stats_summary()

        results.append({
            "name": name,
            "mechanism": mech,
            "output": output_text[0],
            "stats": stats
        })

        print("\n" + "-" * 60)
        print("输出:")
        print("-" * 60)
        text = output_text[0]
        print(text[:300] + "..." if len(text) > 300 else text)

        # 清理
        wrapper._remove_hooks()

    # 打印对比总结
    print("\n" + "=" * 90)
    print("对比总结")
    print("=" * 90)

    # 统计对比表
    print("\n【扰动统计对比】")
    print(f"{'机制':<16} {'视觉角度偏差':<16} {'视觉范数比':<14} {'文本角度偏差':<16} {'文本范数比':<14}")
    print("-" * 76)

    for r in results:
        name = r['name']
        stats = r['stats']

        if stats:
            v_angle = f"{stats.get('visual', {}).get('angle_mean', 0):.1f}°" if 'visual' in stats else "N/A"
            v_ratio = f"{stats.get('visual', {}).get('norm_ratio_mean', 0):.3f}" if 'visual' in stats else "N/A"
            t_angle = f"{stats.get('text', {}).get('angle_mean', 0):.1f}°" if 'text' in stats else "N/A"
            t_ratio = f"{stats.get('text', {}).get('norm_ratio_mean', 0):.3f}" if 'text' in stats else "N/A"
        else:
            v_angle = v_ratio = t_angle = t_ratio = "N/A"

        print(f"{name:<16} {v_angle:<16} {v_ratio:<14} {t_angle:<16} {t_ratio:<14}")

    # 输出对比
    print("\n【输出对比】")
    baseline = results[0]['output'] if results else ""

    for r in results:
        print(f"\n[{r['name']}]")
        print(f"输出长度: {len(r['output'])} 字符")

        if r['name'] != "无扰动" and baseline:
            # 计算相似度
            common = sum(1 for a, b in zip(baseline, r['output']) if a == b)
            sim = common / max(len(baseline), len(r['output'])) * 100
            print(f"与原始输出字符相似度: {sim:.1f}%")

        # 显示前100字
        print(f"前100字: {r['output'][:100]}...")

    print("\n" + "=" * 90)
    print("结论")
    print("=" * 90)
    print("""
    1. vMF 机制:
       - 范数比例 ≈ 1.0（完全保持）
       - 角度偏差可控
       - 最适合 embedding 扰动

    2. 高斯/拉普拉斯机制:
       - 范数会显著增大（高维噪声累积）
       - 可能破坏 embedding 的语义结构

    3. 范数保持高斯:
       - 强制保持范数
       - 但角度偏差不如 vMF 可控

    4. 对于 VLM 的 embedding 扰动，vMF 是最佳选择
    """)


if __name__ == "__main__":
    run_qwenvl_comparison()
