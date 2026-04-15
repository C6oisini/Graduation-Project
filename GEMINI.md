# VLM 隐私保护框架 (VLM Privacy Protection Framework)

这是一个基于差分隐私（Differential Privacy）和几何隐私（Geometric Privacy）的视觉语言模型（VLM）隐私保护框架。该项目主要研究如何在不破坏模型语义能力的前提下，通过在 Embedding 层引入扰动（Perturbation）来保护输入数据的隐私。

## 项目核心 (Core Concepts)

- **目标**: 面向视觉语言模型（如 Qwen-VL）的语义保持度量隐私研究。
- **核心机制**:
  - **vMF (von Mises-Fisher) 扰动**: 推荐机制。基于正交切平面投影，能够保持向量范数（Norm-Preserving），仅改变向量方向。
  - **高斯/拉普拉斯机制**: 传统的差分隐私机制。
  - **范数保持高斯**: 结合了高斯噪声与范数归一化的实验性机制。
- **技术栈**: Python 3.12+, PyTorch, ModelScope, Transformers, NumPy, UV (包管理).

## 项目结构 (Project Structure)

- `privacy/`: 隐私扰动算法实现 (`vmf.py`, `gaussian.py`, `laplace.py` 等)。
- `models/`: 模型包装器 (`qwenvl_wrapper.py`)，通过 PyTorch Forward Hooks 自动拦截并扰动视觉与文本 Embedding。
- `experiments/`: 机制对比与性能评估脚本。
- `examples/`: 快速上手示例。
- `scripts/`: 绘图与验证工具。
- `data/` & `asset/`: 实验数据与可视化结果。
- `docs/`: 毕业论文大纲、开题报告及技术文档。

## 构建与运行 (Building and Running)

项目使用 `uv` 进行依赖管理，确保环境一致性。

### 环境配置
```bash
# 同步依赖并创建虚拟环境
uv sync
```

### 运行实验
```bash
# 1. 机制对比实验（纯算法层面）
uv run experiments/compare_mechanisms.py

# 2. Qwen-VL 模型上的隐私保护对比
uv run experiments/qwenvl_comparison.py --epsilon 0.5

# 3. 运行简单 Demo
uv run examples/qwenvl_demo.py
```

## 开发约定 (Development Conventions)

- **模型包装**: 使用 `models.wrap_qwenvl` 函数对原始模型进行包装。包装后的模型会自动处理 Embedding 层的扰动。
- **隐私预算**: `epsilon` ($\epsilon$) 是核心参数。$\epsilon$ 越小，隐私保护越强，但模型效用（Utility）下降越快。
- **统计监控**: 包装后的模型提供 `get_stats_summary()` 方法，用于监控扰动产生的角度偏差（Angle Deviation）和范数变化比例。
- **代码风格**: 遵循模块化设计，新的隐私机制应实现 `perturb(x)` 接口并放入 `privacy/` 目录下。

## 关键文件 (Key Files)

- `privacy/vmf.py`: 实现核心的几何扰动逻辑。
- `models/qwenvl_wrapper.py`: 负责 Hook 的注册与 Embedding 拦截。
- `pyproject.toml`: 项目依赖与 `uv` 配置。
- `README.md`: 详细的项目使用指南。
