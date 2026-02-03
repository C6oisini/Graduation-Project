"""
QwenVL 上对比不同隐私机制的效果
"""

import torch
import numpy as np
import sys
sys.path.insert(0, '..')

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from models import wrap_qwenvl


def run_qwenvl_comparison(epsilon=0.5):
    """在 QwenVL 上运行对比实验"""
    print("=" * 90)
    print("QwenVL 差分隐私机制对比")
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

        wrapper = wrap_qwenvl(
            base_model,
            mechanism=mech,
            epsilon=epsilon,
            verbose=True
        )

        print("\n生成中...")
        with torch.no_grad():
            generated_ids = wrapper.generate(**inputs, max_new_tokens=100)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        stats = wrapper.get_stats_summary()
        results.append({
            "name": name,
            "mechanism": mech,
            "output": output_text[0],
            "stats": stats
        })

        print("\n输出:")
        print("-" * 60)
        print(output_text[0][:300] + "..." if len(output_text[0]) > 300 else output_text[0])

        wrapper._remove_hooks()

    # 打印对比
    print("\n" + "=" * 90)
    print("对比总结")
    print("=" * 90)

    print("\n【扰动统计】")
    print(f"{'机制':<16} {'视觉角度':<14} {'视觉范数比':<14} {'文本角度':<14} {'文本范数比':<14}")
    print("-" * 72)

    for r in results:
        name = r['name']
        stats = r['stats']
        v_angle = f"{stats.get('visual', {}).get('angle_mean', 0):.1f}°" if 'visual' in stats else "N/A"
        v_ratio = f"{stats.get('visual', {}).get('norm_ratio_mean', 0):.3f}" if 'visual' in stats else "N/A"
        t_angle = f"{stats.get('text', {}).get('angle_mean', 0):.1f}°" if 'text' in stats else "N/A"
        t_ratio = f"{stats.get('text', {}).get('norm_ratio_mean', 0):.3f}" if 'text' in stats else "N/A"
        print(f"{name:<16} {v_angle:<14} {v_ratio:<14} {t_angle:<14} {t_ratio:<14}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', type=float, default=0.5)
    args = parser.parse_args()

    run_qwenvl_comparison(epsilon=args.epsilon)
