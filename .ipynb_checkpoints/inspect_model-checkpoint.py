"""检查 Qwen3-VL 模型结构"""

from modelscope import Qwen3VLForConditionalGeneration

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

print("=" * 60)
print("查找 merger 层的权重维度")
print("=" * 60)

# 查找 merger
for name, module in model.named_modules():
    if 'merger' in name.lower() and 'deepstack' not in name.lower():
        print(f"\n{name}: {type(module).__name__}")
        for param_name, param in module.named_parameters():
            print(f"    {param_name}: {param.shape}")

print("\n" + "=" * 60)
print("查找 language_model embed_tokens")
print("=" * 60)

for name, module in model.named_modules():
    if 'embed_tokens' in name:
        print(f"{name}: {type(module).__name__}")
        if hasattr(module, 'weight'):
            print(f"    weight shape: {module.weight.shape}")
