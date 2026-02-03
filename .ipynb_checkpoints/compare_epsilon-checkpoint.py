"""
对比不同 epsilon 值下的 vMF 扰动效果
"""

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from qwenvl_vmf import wrap_qwenvl_with_vmf
import torch

# 加载模型和处理器
print("=" * 80)
print("加载 Qwen3-VL-8B 模型...")
print("=" * 80)

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

# 测试消息
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

# 准备输入
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
inputs = {k: v.to(base_model.device) for k, v in inputs.items()}

# 测试不同的 epsilon 值
epsilon_values = [
    ("无扰动", None),
    ("极强隐私 (ε=0.01)", 0.01),
    ("强隐私 (ε=0.1)", 0.1),
    ("强隐私 (ε=0.2)", 0.2),
    ("强隐私 (ε=0.3)", 0.3),
    ("强隐私 (ε=0.4)", 0.4),
    ("中等隐私 (ε=0.5)", 0.5),
    ("弱隐私 (ε=1.0)", 1.0),
    ("极弱隐私 (ε=5.0)", 5.0),
]

results = []

for name, epsilon in epsilon_values:
    print("\n" + "=" * 80)
    print(f"测试: {name}")
    print("=" * 80)

    if epsilon is None:
        # 无扰动：直接使用原始模型
        model = base_model
        print("[无扰动模式]")
    else:
        # 添加 vMF 扰动
        model = wrap_qwenvl_with_vmf(
            base_model,
            epsilon=epsilon,
            beta=1.0,
            alpha=2.0,
            perturb_visual=True,
            perturb_text=True
        )

    # 生成输出
    print("\n生成中...")
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    results.append({
        "name": name,
        "epsilon": epsilon,
        "output": output_text[0]
    })

    print("\n" + "-" * 80)
    print("输出:")
    print("-" * 80)
    print(output_text[0])
    print("-" * 80)

# 输出对比总结
print("\n" + "=" * 80)
print("对比总结")
print("=" * 80)

for i, result in enumerate(results):
    print(f"\n【{result['name']}】")
    print(f"输出长度: {len(result['output'])} 字符")
    print(f"前100字: {result['output'][:100]}...")

    # 与无扰动版本对比
    if i > 0:
        original = results[0]['output']
        current = result['output']

        # 计算相似度（简单的字符级别）
        common_chars = sum(1 for a, b in zip(original, current) if a == b)
        similarity = common_chars / max(len(original), len(current)) * 100

        print(f"与无扰动版本的字符相似度: {similarity:.1f}%")

print("\n" + "=" * 80)
print("实验完成！")
print("=" * 80)
print("\n观察要点:")
print("1. 角度偏差越大，输出差异可能越大")
print("2. epsilon 越小（隐私保护越强），扰动越大")
print("3. 即使有扰动，模型仍应能理解图像内容")
print("4. 过小的 epsilon 可能导致输出质量下降")
