"""
QwenVL-8B 使用示例 - 集成隐私保护
"""

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from models import wrap_qwenvl


def main():
    # 加载模型
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
    )

    # 添加隐私保护（一行代码）
    model = wrap_qwenvl(
        model,
        mechanism='vmf',      # 扰动机制: 'vmf', 'gaussian', 'laplace', 'norm_preserving'
        epsilon=0.5,          # 隐私预算，越小隐私保护越强
        perturb_visual=True,  # 扰动视觉 embedding
        perturb_text=True,    # 扰动文本 embedding
        verbose=True          # 输出详细信息
    )

    # 可以动态调整
    # model.set_epsilon(0.3)  # 更强的隐私保护
    # model.disable()         # 临时禁用扰动
    # model.enable()          # 重新启用

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
    inputs = {k: v.to(model.model.device) for k, v in inputs.items()}

    # 生成（扰动自动应用）
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    print("\n" + "=" * 60)
    print("输出:")
    print("=" * 60)
    print(output_text[0])

    # 查看统计
    stats = model.get_stats_summary()
    print("\n统计摘要:", stats)


if __name__ == "__main__":
    main()
