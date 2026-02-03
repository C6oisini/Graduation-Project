# VLM 隐私保护框架

基于差分隐私的视觉语言模型 (VLM) 隐私保护框架，支持在 embedding 层进行扰动。

## 项目结构

```
Graduation-Project/
├── privacy/                    # 隐私保护机制
│   ├── __init__.py
│   ├── vmf.py                 # vMF 扰动（推荐）
│   ├── gaussian.py            # 高斯机制
│   ├── laplace.py             # 拉普拉斯机制
│   └── norm_preserving.py     # 范数保持高斯
│
├── models/                     # 模型包装器
│   ├── __init__.py
│   └── qwenvl_wrapper.py      # QwenVL 隐私包装器
│
├── experiments/                # 实验脚本
│   ├── __init__.py
│   ├── compare_mechanisms.py  # 机制对比实验
│   └── qwenvl_comparison.py   # QwenVL 上的对比
│
├── examples/                   # 使用示例
│   ├── __init__.py
│   └── qwenvl_demo.py         # QwenVL 示例
│
├── utils/                      # 工具函数
│   └── __init__.py
│
└── docs/                       # 文档
    ├── vMF.md
    └── 大纲.md
```

## 快速开始

### 安装依赖

```bash
uv sync
```

### 基本使用

```python
from modelscope import Qwen3VLForConditionalGeneration
from models import wrap_qwenvl

# 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# 一行代码添加隐私保护
model = wrap_qwenvl(model, mechanism='vmf', epsilon=0.5)

# 正常使用，扰动自动应用
output = model.generate(**inputs)
```

### 支持的机制

| 机制 | 说明 | 范数保持 | 隐私类型 |
|------|------|----------|----------|
| `vmf` | vMF 扰动 | ✓ | 几何隐私 |
| `gaussian` | 高斯机制 | ✗ | (ε,δ)-DP |
| `laplace` | 拉普拉斯机制 | ✗ | 纯 ε-DP |
| `norm_preserving` | 范数保持高斯 | ✓ | 近似 DP |

### 运行实验

```bash
# 机制对比（纯算法）
uv run experiments/compare_mechanisms.py

# QwenVL 上对比
uv run experiments/qwenvl_comparison.py --epsilon 0.5

# 示例
uv run examples/qwenvl_demo.py
```

## API 参考

### wrap_qwenvl

```python
wrap_qwenvl(
    model,                      # QwenVL 模型
    mechanism='vmf',            # 扰动机制
    epsilon=1.0,                # 隐私预算
    delta=1e-5,                 # δ 参数（高斯机制）
    beta=1.0,                   # vMF 调节系数
    perturb_visual=True,        # 扰动视觉 embedding
    perturb_text=True,          # 扰动文本 embedding
    verbose=False               # 输出详细信息
)
```

### 动态调整

```python
model.set_epsilon(0.3)  # 调整隐私预算
model.disable()         # 禁用扰动
model.enable()          # 启用扰动
model.get_stats_summary()  # 获取统计信息
```

## 参数选择指南

| epsilon | 隐私强度 | 角度偏差 | 适用场景 |
|---------|----------|----------|----------|
| 0.1-0.2 | 强 | 70-80° | 高敏感数据 |
| 0.3-0.5 | 中 | 60-70° | 平衡隐私和效用 |
| 0.5-1.0 | 弱 | 45-60° | 一般场景 |
| >1.0 | 极弱 | <45° | 效用优先 |

## License

MIT
