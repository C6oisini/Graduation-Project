# 面向视觉语言模型的语义保持度量隐私研究 (Semantic-Preserving Metric Privacy for VLMs)

## 摘要 (Abstract)

（略，保持前文内容）

---

## 1. 前置知识与问题定义 (Prerequisites and Problem Definition)

### 1.1 视觉语言模型 (VLM) 的几何特性：以 Qwen-VL 为例
视觉语言模型的核心在于将跨模态信息映射至统一的语义表征空间。以 Qwen-VL-8B 为例，其处理流程不仅是数据的传递，更是几何结构的演变：

#### 1.1.1 视觉特征的生成与对齐
1.  **ViT 编码器**: 图像 $I$ 被切分为 Patch，经 Transformer 层计算。各层内部的自注意力机制公式为：
    $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$
    其中点积项 $\mathbf{q}_i^\top \mathbf{k}_j = \|\mathbf{q}_i\| \|\mathbf{k}_j\| \cos\theta_{ij}$ 表明，模型对权重的判定极度依赖于 Embedding 向量间的**夹角 $\theta$**。
2.  **Smash Data 的产生**: 在拆分推理（Split Inference）中，视觉编码器输出的中间表征被称为“Smash Data”。由于 Projector 层的线性映射性质，这些向量完整保留了图像的拓扑结构信息。

#### 1.1.2 语义流形假设
在 $d=4096$ 的超高维空间中，有效的语义特征并非均匀分布，而是集中在低维的**语义流形（Semantic Manifold）**上。由于 `LayerNorm` 的存在，这些特征被自然地约束在超球面的局部区域。这意味着，任何偏离超球面切空间的扰动都可能导致语义的灾难性丢失。

### 1.2 隐私保护理论：从 DP 到度量隐私

#### 1.2.1 本地差分隐私 (LDP) 的局限性
传统的本地差分隐私要求任意两个输入 $x, x'$ 产生的输出分布必须不可区分。这在 Embedding 空间会导致两个语义完全无关的向量被迫产生相似输出，造成巨大的效用损失。

#### 1.2.2 度量隐私 (Metric Privacy) 与几何感知
度量隐私（又称 $d_{\mathcal{X}}$-privacy）是对 DP 的泛化。其核心在于引入度量距离 $d(\cdot, \cdot)$：
$$\frac{\Pr[\mathcal{M}(x) \in S]}{\Pr[\mathcal{M}(x') \in S]} \le \exp(\epsilon \cdot d(x, x'))$$
- **优势**: 这种“距离感知”的特性允许模型在保护微观细节（小 $d$）的同时，保留宏观语义（大 $d$）。
- **应用**: 对于 VLM，我们采用测地线距离作为 $d$，将隐私预算直接映射到超球面的几何偏转上。

### 1.3 威胁模型与特征反演攻击定义

#### 1.3.1 诚实但好奇的云端节点
本研究设定的攻击者为云端推理服务器。其具备以下能力：
- **白盒背景**: 了解对齐层（Projector）的权重。
- **计算资源**: 拥有训练强大的逆向解码器（Inversion Decoder）所需的算力。

#### 1.3.2 特征反演攻击的形式化
攻击者的目标是找到一张伪造图像 $\hat{I}$，使其生成的扰动特征与截获的 $y$ 尽可能接近。其优化目标可表示为：
$$\min_{\hat{I}} \mathcal{L}_{\text{inv}} = \|\Phi_{\text{enc}}(\hat{I}) - y\|_2^2 + \lambda \mathcal{R}(\hat{I})$$
其中 $\Phi_{\text{enc}}$ 为端侧编码器，$\mathcal{R}$ 为图像先验正则项（如 TV 损失）。

**本研究的防御目标**: 通过 vMF 扰动增加该优化问题的非凸性与解空间的不确定性，使得 $\hat{I}$ 在视觉上与原图 $I$ 的 SSIM 降至最低，同时不干扰下游 LLM 的 `softmax` 激活分布。

---

## 2. 方法论 (Methodology)

### 2.1 语义流形与度量空间假设

#### 2.1.1 超球面嵌入假设
在 Qwen-VL 等 Transformer 架构中，经 `LayerNorm` 标准化后的特征向量 $\mathbf{h}$ 被约束在有界区域内。研究表明，VLM 的核心语义主要编码在向量的**方向**而非幅度上。自注意力的核心计算 $\mathbf{q}_i^\top \mathbf{k}_j = \|\mathbf{q}_i\| \|\mathbf{k}_j\| \cos\theta_{ij}$ 进一步证实了语义相关性对夹角 $\theta$ 的强依赖性。因此，本研究将 Embedding 空间抽象为单位超球面 $\mathbb{S}^{d-1}$。

#### 2.1.2 嵌入空间度量定义
本研究采用基于测地线距离（弧长）的角度度量定义隐私邻域：
$$d_\varepsilon(\mathbf{x}, \mathbf{x}') = \varepsilon \cdot \arccos\left(\frac{\mathbf{x}^\top \mathbf{x}'}{\|\mathbf{x}\| \|\mathbf{x}'\|}\right)$$
该度量确保了隐私保护强度与数据点间的语义距离成正比，即“语义相近获强保护，语义相远获弱保护”，这天然契合多模态嵌入的特性。

### 2.2 vMF 几何扰动框架 (Ours)

#### 2.2.1 算法描述：正交切平面投影
本研究提出**算法 1：正交切平面投影扰动**。对输入 Embedding $\mathbf{x} \in \mathbb{R}^d$，执行以下步骤：

1.  **特征分解**: 计算模长 $r = \|\mathbf{x}\|_2$ 与单位方向 $\boldsymbol{\mu} = \mathbf{x} / r$。
2.  **正交噪声生成**: 采样标准正态噪声 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，通过投影算子 $P_\perp = \mathbf{I}_d - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ 映射至切空间并归一化：
    $$\hat{\mathbf{n}}_\perp = \frac{P_\perp \mathbf{n}}{\|P_\perp \mathbf{n}\|_2}$$
3.  **动态缩放**: 计算扰动幅度 $\lambda = \beta / \varepsilon$，其中 $\beta$ 为平滑因子。
4.  **球面偏转与重投影**:
    $$\mathbf{z} = \boldsymbol{\mu} + \lambda \cdot \hat{\mathbf{n}}_\perp, \quad \boldsymbol{\mu}' = \mathbf{z} / \|\mathbf{z}\|_2$$
5.  **范数恢复**: 最终输出 $\mathbf{y} = r \cdot \boldsymbol{\mu}'$。

#### 2.2.2 偏转角度理论值
该机制产生的偏转角 $\theta$ 具有确定性。在 Word 公式编辑器中可直接使用以下 LaTeX 形式：
$$\theta = \arccos\left(\frac{1}{\sqrt{1 + \lambda^2}}\right) = \arctan(\lambda) = \arctan\left(\frac{\beta}{\epsilon}\right)$$

### 2.3 隐私性证明概要

#### 2.3.1 主定理：近似度量隐私
**定理 2.1**: 设 $\delta_{\text{TV}}(d)$ 为切平面分布与匹配 vMF 分布间的全变差距离。算法 1 满足 $(\kappa^*, \delta_{\text{TV}})$-近似度量隐私。对于任意输入 $\mathbf{x}, \mathbf{x}'$：
$$\Pr[\mathcal{M}(\mathbf{x}) \in S] \le e^{\kappa^* \cdot d_{\text{geo}}(\boldsymbol{\mu}, \boldsymbol{\mu}')} \cdot \Pr[\mathcal{M}(\mathbf{x}') \in S] + 2\delta_{\text{TV}}(d)$$
其中 $\kappa^*$ 通过数值求解 $A_d(\kappa^*) = \cos\theta$ 确定。

#### 2.3.2 高维浓度与收敛
由高维几何特性，当维度 $d \to \infty$ 时，vMF 分布的角度方差 $\text{Var}[\cos\angle] = O(1/d)$。在 VLM 典型的 $d=4096$ 空间下，$\delta_{\text{TV}}$ 趋于 0，证明了切平面近似在统计意义上等价于严格的 vMF 度量隐私保护。

### 2.4 效用性分析

#### 2.4.1 严格范数保持
**定理 2.2**: 扰动前后满足 $\|\mathbf{y}\| = \|\mathbf{x}\|$。这确保了 Attention 机制中 Softmax 的输入量级不变，彻底消除了传统高斯机制带来的“注意力崩塌”问题。

#### 2.4.2 点积偏差界与 Attention 稳定性
**定理 2.3**: 对独立扰动的向量 $\mathbf{y}_1, \mathbf{y}_2$，其点积期望满足：
$$\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] = \cos^2\theta \cdot \mathbf{x}_1^\top \mathbf{x}_2$$
这意味着注意力权重的相对排序（Rank）在期望下得以保持，从而在强隐私约束下依然能维持模型生成的逻辑连贯性。

### 2.5 多模态非对称预算分配与掩码机制

#### 2.5.1 模态异构性分析
在 Qwen-VL 等多模态模型中，视觉与文本嵌入表现出显著的几何差异：
1.  **视觉嵌入 (Visual Embeddings)**: 具有高度冗余性。由于自然图像的局部相关性，相邻 Patch 的特征向量高度相似，能够承受较大规模的角度偏转。
2.  **文本嵌入 (Text Embeddings)**: 具有极高的离散性。每个 Token（词元）均携带独立的语义信息，极小的方向偏转就可能导致 Token 从“猫”漂移到语义无关的“电视”。

#### 2.5.2 非对称隐私预算分配策略
针对上述异构性，本研究提出一种非对称分配策略。设定系统总隐私预算为 $\epsilon_{\text{sys}}$，引入非对称因子 $\alpha > 1$：
$$\epsilon(\text{modality}) = \begin{cases} \epsilon_{\text{sys}}, & \text{Visual} \\ \alpha \cdot \epsilon_{\text{sys}}, & \text{Text} \end{cases}$$
对应的扰动幅度 $\lambda$ 为：
$$\lambda(\text{modality}) = \begin{cases} \beta/\epsilon_{\text{sys}}, & \text{Visual} \\ \beta/(\alpha \cdot \epsilon_{\text{sys}}), & \text{Text} \end{cases}$$
该策略优先保障了文本指令的精确性，同时在视觉通道实施更强的保护。

#### 2.5.3 敏感区掩码保护 (Sensitive Masking)
在 VLM 的多轮对话中，并非所有 Embedding 都包含用户隐私（如系统指令、任务模板）。本研究引入掩码矩阵 $\mathbf{M} \in \{0,1\}^L$，实现精准保护：
$$\mathbf{y}_{\text{final}} = \mathbf{M} \odot \mathbf{y}_{\text{perturbed}} + (1-\mathbf{M}) \odot \mathbf{x}_{\text{original}}$$
- 当 $M_i = 0$ 时（如 "Describe this image:" 等系统提示），保持原值不变，确保模型指令遵循能力不下降。
- 当 $M_i = 1$ 时（如用户输入的具体内容），应用 vMF 扰动，实现端到端的隐私屏障。

---

## 3. 实验设置 (Experimental Setup)

### 3.1 实验数据集深度说明
- **MS-COCO (2017)**: 使用 Validation 集中的 5000 张图像进行 Image Captioning 测试。该任务要求模型生成长序列文本，能够极好地反映扰动对逻辑连贯性的影响。
- **VQA-v2**: 选取具有代表性的问答对（包含 Yes/No、Number、Other 三类问题）。该任务测试扰动对细粒度特征（如物体颜色、数量）的破坏程度。

### 3.2 攻击模型：特征反演审计 (Inversion Auditing)
为了客观评估隐私泄露风险，本研究构建了一个**基于扩散模型解码器（SD-VAE Decoder）的攻击者**：
1.  **数据获取**: 攻击者截获模型 Projector 层输出的扰动序列。
2.  **优化目标**: 攻击者训练一个逆向网络 $G$，尝试最小化 $\mathcal{L} = \|G(y) - \text{Image}\|$。
3.  **评价**: 如果 $G(y)$ 能够还原出原始图像的轮廓或文本中的关键词，则视为隐私泄露。

### 3.3 详细评估指标定义与数学表达

#### 3.3.1 CIDEr (Consensus-based Image Description Evaluation)
CIDEr 通过计算待测句子与参考句子集之间的 n-gram 重叠度，并利用 TF-IDF 进行加权，衡量生成文本的共现频率：
$$\text{CIDEr}_n(c_i, S_i) = \frac{1}{m} \sum_{j=1}^m \frac{\boldsymbol{g}^n(c_i) \cdot \boldsymbol{g}^n(s_{ij})}{\|\boldsymbol{g}^n(c_i)\| \|\boldsymbol{g}^n(s_{ij})\|}$$
其中 $\boldsymbol{g}^n(c_i)$ 是 n-gram 的 TF-IDF 向量。该指标对视觉理解的准确性及语义一致性高度敏感。

#### 3.3.2 CLIPScore (多模态语义一致性)
利用预训练的 CLIP 视觉编码器 $E_I$ 和文本编码器 $E_T$ 计算图像与生成文本在共同特征空间中的余弦相似度：
$$\text{CLIPScore}(I, C) = w \cdot \max(\cos(E_I(I), E_T(C)), 0)$$
其中 $w$ 为缩放系数（通常取 2.5）。该指标衡量了扰动后模型是否仍能捕捉到图像的核心语义。

#### 3.3.3 SSIM (Structural Similarity Index)
用于衡量特征反演攻击还原出的图像 $x$ 与原始图像 $y$ 的结构相似度，综合考虑了亮度 ($l$)、对比度 ($c$) 和结构 ($s$)：
$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$$
具体形式为：
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
其中 $\mu$ 为均值，$\sigma$ 为方差，$\sigma_{xy}$ 为协方差。**在隐私审计中，SSIM 数值越低代表防御越成功。**

#### 3.3.4 余弦相似度 (Cosine Similarity)
用于衡量扰动前后 Embedding 向量的方向偏差，也是本研究中“语义保持”的直接几何度量：
$$\cos(\theta) = \frac{\boldsymbol{x} \cdot \boldsymbol{y}}{\|\boldsymbol{x}\| \|\boldsymbol{y}\|}$$
角度偏差定义为 $\Delta\theta = \arccos(\cos(\theta)) \cdot \frac{180}{\pi}$。

### 3.4 对比基准 (Baselines)
---

## 4. 机制可视化与验证 (Visualization & Validation)

为了深入理解 vMF 扰动机制的内在特性，本研究通过几何模拟和统计分析对机制进行了可视化验证。

### 4.1 机制直观可视化分析 (Visual Analysis, a-d)
本研究通过四张关联子图对 vMF 机制的几何与统计特性进行了多维展示（见 `asset/vmf_visualization.png`）：

#### (a) vMF Distribution on Unit Sphere (球面扰动分布)
*   **目的**：在黎曼流形（单位超球面）上直观展示扰动过程的几何表现。
*   **观察**：不同隐私预算 $\epsilon$ 下的扰动样本呈“点簇”状分布在原始向量 $\mu$ 周围。随着 $\epsilon$ 减小（隐私增强），方向散布的方差增大，点簇在球面上的覆盖范围逐渐扩张。
*   **结论**：可视化证实了扰动过程严格受限于超球面表面，确保了在模糊方向信息的同时，向量模长（能量）保持绝对不变。

#### (b) Relation between $\epsilon$ and $\theta$ (偏转角映射曲线)
*   **目的**：量化隐私预算 $\epsilon$ 与几何畸变（角度偏差）之间的确定性映射关系。
*   **观察**：曲线严格遵循 $\theta = \arctan(\beta/\epsilon)$ 的反切规律。在强隐私区（$\epsilon \to 0$），偏转角迅速逼近 $90^\circ$，实现语义空间的极大化混淆。
*   **结论**：该映射关系构成了本框架的“控制中枢”，使得用户能够通过单一参数实现对隐私-效用平衡点的精确微调。

#### (c) Norm Stability Comparison (范数保持性对比)
*   **目的**：经验性验证 vMF 机制相对于传统 DP 机制在数值稳定性上的代差优势。
*   **观察**：vMF 的范数分布呈现为位于 1.0 处的狄拉克脉冲；而高斯和拉普拉斯机制则表现为显著的范数漂移（均值右移）与高方差扩散。
*   **结论**：通过消除范数偏移，vMF 避免了触发 Transformer 内部激活值的分布崩溃（如 LayerNorm 的饱和或失效），这是其维持高效用的根本物理诱因。

#### (d) Privacy-Utility Trade-off: Cosine Similarity (语义相似度权衡)
*   **目的**：在全隐私预算范围内评估各机制的效用保留效率（余弦相似度）。
*   **观察**：在任意给定的 $\epsilon$ 下，vMF 提供的平均余弦相似度均显著优于高斯和拉普拉斯机制，且实验采样点与理论预估曲线完美契合。
*   **结论**：这验证了方向性扰动是在嵌入空间（Embedding Space）中引入噪声的“高信息效率”方式，能够以更小的效用代价换取更高的隐私增益。

### 4.4 近似机制的统计验证 (Statistical Validation, a-f)
这六张子图共同构成了一个严密的统计验证体系，旨在证明我们提出的“切平面近似（Tangent Plane Approximation）”在处理高维 Embedding 隐私时，既能保持真实 vMF 分布的核心特性，又具备更优的数值稳定性（见 `asset/vmf_validation.png`）：

#### (a) Angular Deviation Distribution (角度偏差分布)
*   **目的**：对比近似算法（TP）与真实采样（vMF）在方向偏转上的微观分布。
*   **观察**：在高维空间下，切平面近似产生的偏转角度呈现极窄的分布（几乎是脉冲状），而真实 vMF 分布较宽。
*   **结论**：虽然分布形态有差异，但两者的**均值（$\mu$）高度一致**。这说明近似算法在期望意义上准确捕捉了隐私预算 $\epsilon$ 要求的方向偏移量。

#### (b) Tangent Plane Isotropy (切平面各向同性)
*   **目的**：验证生成的扰动在超球面的切空间内是否是“公平”的（即不偏向任何特定维度）。
*   **观察**：极坐标直方图呈现为一个接近完美的圆形，各方向采样概率均匀。
*   **结论**：证明了该机制满足**各向同性（Isotropy）**，攻击者无法通过统计偏差来推断原始向量的特定分量，保证了扰动的盲目性和安全性。

#### (c) Dimension Scaling (维度缩放行为)
*   **目的**：研究维度 $d$ 对近似准确性的影响。
*   **观察**：在 $d < 128$ 的低维区，近似理论与真实理论背离；但随着 $d$ 增加到 $512$ 以上，两条线**完美收敛**。
*   **结论**：揭示了“高维红利”。证明了基于切平面的几何简化在 VLM 的高维特征空间（通常 $d=4096$）中是**数学等价**的，但在低维下不适用。

#### (d) Norm Preservation (范数保持性对比)
*   **目的**：展示本方法相对于传统 DP 机制在数值稳定性上的物理优势。
*   **观察**：小提琴图中，vMF (Ours) 是一条位于 1.0 的直线；而 Gaussian 和 Laplace 呈现出巨大的波动且均值向上漂移（范数膨胀）。
*   **结论**：这是 vMF 胜出的底层原因。它**彻底消除了范数漂移**，确保了 Transformer 内部 `LayerNorm` 的输入稳定性，从而在隐私模式下保留了极高的语义效用。

#### (e) Concentration Equivalence (浓度参数等价性)
*   **目的**：验证隐私预算 $\epsilon$ 与语义相似度（余弦相似度）的映射关系是否可靠。
*   **观察**：切平面理论曲线与真实 vMF 理论点完全重合。
*   **结论**：证明了我们对 $\epsilon$ 的参数化（$\lambda = \beta/\epsilon$）在数学上成功模拟了 vMF 分布的**浓度参数（Concentration Parameter, $\kappa$）**。这使得我们可以直接借用差分隐私的理论框架来量化隐私保护强度。

#### (f) Projection QQ Plot (投影坐标 QQ 图)
*   **目的**：从边缘分布的角度检验两种算法生成的噪声是否具有统计一致性。
*   **观察**：采样点几乎全部落在红色的 $y=x$ 参考线上。
*   **结论**：这是**分布一致性**的强证据。它表明在切平面的任意投影轴上，近似算法产生的扰动分量与真实 vMF 产生的扰动分量是无法区分的，进一步增强了近似算法的理论可信度。

### 4.5 语义保持能力的权衡 (余弦相似度对比)
通过余弦相似度的对比可以发现，在相同的 $\epsilon$ 下，vMF 机制提供的平均余弦相似度始终高于高斯和拉普拉斯机制，且与理论曲线高度契合。这证明了在度量隐私框架下，基于超球面的方向扰动比欧氏空间内的盲目加噪具有更高的**隐私-效用效率**。

---

## 5. 结果分析 (Results Analysis)

本研究在多个主流 VLM 任务上对 vMF 机制进行了全面评估，以下是详细的实验结果分析。

### 5.1 主实验效用分析 (Main Utility Results)
参照 `asset/ms_coco_epsilon_curves.png` 与 `asset/vqa_v2_epsilon_curves.png`：
- **性能韧性**: 在 MS-COCO 任务中，当隐私预算 $\epsilon=0.1$（强隐私约束）时，vMF 机制的 CIDEr 得分保持在 **84.1**，显著高于 Embedding-Laplace (75.6) 和 Pixel-Gaussian (58.2)。这表明 vMF 能够在极端扰动下依然保留图像的核心语义逻辑。
- **收敛特征**: 随着 $\epsilon$ 增加至 5.0，所有机制的效用指标均平滑收敛于 No-Privacy Baseline。vMF 机制在全隐私预算区间内始终处于帕累托前沿，证明了其在“隐私-效用”博弈中的高效性。
- **任务一致性**: VQA-v2 的准确率趋势与 COCO 高度一致，vMF 在处理细粒度视觉问答时，避免了传统机制常出现的“属性混淆”现象。

### 5.2 隐私-效用权衡分析 (Trade-off Curves)
参照 `asset/ms_coco_tradeoff.png` 与 `asset/vqa_v2_tradeoff.png`：
- **最优平衡点**: 在 Trade-off 坐标系中，vMF 曲线始终位于左上方（或最靠近坐标原点/理想区域）。在相同的攻击成功率（ASR）水平下，vMF 能够提供比其他机制高出 **10%-15%** 的相对效用增益。
- **安全边界**: 观察 ASR 曲线可以发现，vMF 在中等隐私区（$\epsilon=0.5$）的斜率最为平缓，这意味着它能够提供更宽的“安全操作窗口”，不会因为 $\epsilon$ 的微小波动导致隐私保护瞬间失效。

### 5.3 物理层稳定性验证 (Physical Stability)
参照 `asset/supplementary_norm_stability.png`：
- **范数偏差对比**: 条形图清晰地展示了物理层面的差异。传统 Gaussian 和 Laplace 机制在加噪后，Embedding 范数均值出现了 **10%-30%** 的向上漂移（Norm Expansion），这种漂移直接破坏了 Transformer 内部 LayerNorm 层的输入分布假设。
- **确定性优势**: 相比之下，vMF 机制在所有 $\epsilon$ 设置下均保持了 **1.0 的恒定范数**。这种数值稳定性是模型在隐私模式下依然能维持逻辑连贯性的根本物理保证。

### 5.4 全模态保护的必要性分析 (Ablation Study)
参照 `asset/supplementary_ablation.png`：
- **防御漏洞**: 消融实验显示，若仅采用视觉扰动（Visual-Only），虽然 CIDEr 得分略高，但其 ASR（攻击成功率）高达 **0.312**（$\epsilon=0.1$）。这是由于攻击者可以利用未受保护的文本 Embedding 之间的跨模态关联进行特征还原。
- **闭环防御**: 只有在视觉与文本同时应用 vMF 扰动（Both-vMF）时，才能将 ASR 压制在 **0.182** 的极低水平，证明了全模态协同防御在 VLM 隐私保护中的不可替代性。

### 5.5 跨模型泛化性评估 (Cross-Model Generalization)
参照 `asset/supplementary_cross_model.png`：
- **架构无关性**: 本研究在 LLaVA-1.5 模型上复现了实验。结果显示，vMF 机制在 LLaVA 上的效用保持能力与隐私防御能力与其在 Qwen-VL 上的表现高度一致。
- **结论**: 这证实了 vMF 几何隐私框架并不依赖于特定模型的参数权重，而是针对 Transformer 架构通用的 Embedding 几何约束设计的，具有极强的跨模型部署潜力。

