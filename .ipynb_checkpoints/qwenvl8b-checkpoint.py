"""
QwenVL-8B 使用示例 - 集成 vMF 隐私保护扰动
"""

from modelscope import Qwen3VLForConditionalGeneration, AutoProcessor
from qwenvl_vmf import wrap_qwenvl_with_vmf

# ============================================================
# 使用 wrap_qwenvl_with_vmf 即插即用
# ============================================================

# 加载原始模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# 一行代码添加 vMF 扰动
model = wrap_qwenvl_with_vmf(
    model,
    epsilon=1,          # 隐私预算，越小隐私保护越强
    beta=1.0,             # 调节系数
    alpha=1.0,            # 文本通道的非对称分配因子
    perturb_visual=True,  # 扰动视觉 embedding
    perturb_text=True     # 扰动文本 embedding
)

# 可以动态调整隐私预算
# model.set_epsilon(0.5)  # 更强的隐私保护

# 可以临时禁用/启用扰动
# model.disable()  # 禁用扰动
# model.enable()   # 启用扰动

# We recommend enabling flash_attention_2 for better acceleration and memory saving
# model = Qwen3VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen3-VL-8B-Instruct",
#     dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

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

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

# 将输入移到模型所在的设备（GPU）
inputs = {k: v.to(model.model.device) for k, v in inputs.items()}

# Inference: Generation of the output (扰动会自动应用到 embedding)
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
