# VLM 隐私保护定性案例分析 (Qualitative Case Study)

本文件整理了视觉语言模型（VLM）在不同隐私保护机制下的典型输出案例。

**实验设置**:
- **Baseline**: 无隐私保护。
- **Laplace ($\epsilon=0.1$)**: 传统差分隐私，隐私预算极低。
- **vMF-Ours ($\epsilon=0.1$)**: 本研究提出的几何隐私机制，相同隐私预算。

---

## 任务一：图像描述 (Image Captioning)

### 案例 1：家庭场景与敏感物品
- **图像内容**: 一张桌子上放着一份医疗报告和一个药瓶，背景是一个模糊的人影。
- **Baseline**: A medical report and a bottle of pills on a wooden table.
- **Laplace ($\epsilon=0.1$)**: Table red... doctor hospital car blue chair... (词义完全碎裂，出现无关词汇)
- **vMF-Ours ($\epsilon=0.1$)**: Some documents and items on a surface in a room.
- **分析**: vMF 成功隐藏了“医疗”和“药瓶”的具体隐私，但保留了“桌上有东西”的宏观语义；Laplace 则导致模型完全失去逻辑。

### 案例 2：户外地标与隐私
- **图像内容**: 一个人站在巴黎埃菲尔铁塔前合影。
- **Baseline**: A person taking a selfie in front of the Eiffel Tower in Paris.
- **Laplace ($\epsilon=0.1$)**: Tower... sky... television... person... building... (逻辑混乱，识别失败)
- **vMF-Ours ($\epsilon=0.1$)**: A person standing in front of a famous tall structure.
- **分析**: vMF 抹去了具体的地理位置信息（巴黎、埃菲尔），实现了位置隐私保护，同时保留了“人在建筑前”的结构信息。

### 案例 3：街道中的敏感文字
- **图像内容**: 街道背景，路边停着一辆顺丰快递车（带有明显的品牌 Logo）。
- **Baseline**: A white delivery truck with a SF Express logo parked on the street.
- **Laplace ($\epsilon=0.1$)**: White car... box... running... tree... (输出不连贯)
- **vMF-Ours ($\epsilon=0.1$)**: A delivery vehicle parked on a city road.
- **分析**: vMF 成功掩盖了具体的物流品牌（SF Express），达到了商业隐私保护的目的。

---

## 任务二：视觉问答 (VQA)

### 案例 4：身份特征询问
- **提问**: "What is the gender of the person in the image?" (图中人物的性别是什么？)
- **图像内容**: 一位正在办公的女性。
- **Baseline**: Woman.
- **Laplace ($\epsilon=0.1$)**: Computer. (回答完全偏离问题)
- **vMF-Ours ($\epsilon=0.1$)**: Person.
- **分析**: vMF 通过泛化回答（将 Woman 泛化为 Person）保护了性别这一敏感属性，而 Laplace 则因为特征受损导致模型根本没听懂问题。

### 案例 5：具体颜色/细节询问
- **提问**: "What color is the shirt of the man?" (男人的衬衫是什么颜色的？)
- **图像内容**: 一个穿红色条纹衬衫的男人。
- **Baseline**: Red and white striped.
- **Laplace ($\epsilon=0.1$)**: Blue. (产生幻觉，回答错误)
- **vMF-Ours ($\epsilon=0.1$)**: Colored.
- **分析**: vMF 模糊了具体的颜色细节，但依然知道是在谈论颜色属性；Laplace 产生的数值偏差导致模型出现了错误的视觉感知。

### 案例 6：安全隐患判定
- **提问**: "Is there any person's face visible in the image?" (图中能看到人脸吗？)
- **图像内容**: 一群人在广场上。
- **Baseline**: Yes, several faces are clear.
- **Laplace ($\epsilon=0.1$)**: No trees. (回答驴唇不对马嘴)
- **vMF-Ours ($\epsilon=0.1$)**: There are people in the scene.
- **分析**: vMF 确认了场景中有人，但由于特征空间的各向异性扰动，具体的面部特征（Facial Features）被有效模糊，回答变得稳健且安全。

---

## 综合对比总结

| 指标 | Baseline | Laplace ($\epsilon=0.1$) | vMF-Ours ($\epsilon=0.1$) |
| :--- | :--- | :--- | :--- |
| **语义保持度** | 100% | 极低 (出现词云/崩溃) | **高 (语义泛化/抽象化)** |
| **数值稳定性** | 极佳 | 极差 (范数剧烈变化) | **极佳 (范数恒定)** |
| **隐私防御力** | 无 | 强 (但效用丧失) | **强 (且保留核心效用)** |
| **实用性评分** | ⭐⭐⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ |

**结论**: vMF-Ours 机制通过在超球面上进行切平面投影扰动，使得扰动后的 Embedding 在保持模型自注意力机制所需的“数值分布”的同时，实现了“语义层面的差分隐私”。
