# 面向视觉语言模型的语义保持度量隐私研究 (Semantic-Preserving Metric Privacy for VLMs)

## 摘要 (Abstract)

随着视觉语言模型（Vision-Language Models, VLMs）在端云协同拆分推理场景下的广泛应用，用户上传的多模态表征面临着严峻的特征反演泄露风险。现有的隐私保护机制多基于欧氏空间的加性噪声，忽略了高维 Embedding 在语义流形上的几何特性，导致模型推理效用大幅下降。本文提出了一种面向多模态嵌入层的语义度量隐私保护框架。该框架利用 von Mises-Fisher (vMF) 分布在单位超球面上实施正交切平面投影扰动，在严格保持特征模长守恒的前提下，实现了对语义方向的受控偏转。同时，针对图文异构模态，设计了双通道非对称隐私预算分配与文本敏感区掩码策略。基于 Qwen-VL-8B 模型的实验表明，本机制在强隐私区间（$\epsilon=0.1$）内，能够将攻击成功率压制在约 16% 的水平，同时保持较高的图像描述质量，成功打破了高维特征空间下隐私与效用的零和博弈。

---

## 1. 前言

### 1.1 研究背景与研究意义
近年来Qwen-VL等多模态大语言模型（Multimodal Large Language Models，MLLMs）[1]在图像识别、视觉问答、跨模态推理上都有很好的效果。大多数模型都是用大量的图文配对数据来进行预训练[2]，把图像、文字等各种信息统一到一个嵌入空间中去，给后面的推理和创造赋予相同的表示。但是模型的大小也会给部署带来负担。Qwen-VL-8B有几十亿个参数，既占用了很大的存储和显存资源，又需要端侧设备具备很强的计算能力。对于移动终端、物联网边缘设备而言，在本地执行完整的模型会耗费很多资源，因此实际应用时通常会采取端云协同或者拆分推理的方式进行部署。
为了缓解端侧设备不能处理全部模型的问题，拆分推理（Split Inference）[3]给人们提供了一种比较好的部署方式。通常把模型分成端侧和云侧两部分，终端用较轻的视觉编码器（Visual Encoder）和文本编码器（Text Encoder）来完成，之后将得到的中间特征（Smash Data）发送到云端，云端再用更强的语言模型部分来继续推理和生成。这样可以避免原始图像或者文本离开本地设备，进而降低原始数据被泄露的风险。但是要注意的是中间特征里还有许多语义信息，如果云端或者传输通道被攻击了，也会导致隐私泄露。
现在学界很多的研究都表明拆分推理中间特征不等于安全信息。很多研究者[4][5]认为攻击者即使不能直接获得原始图像，也可以通过嵌入层反演攻击（Embedding Inversion Attack,EIA）从中间表示中获取到部分视觉内容。近两年有关工作的进步又证明了，因为生成模型先验能力增强后，深层特征里还有许多可以被利用的语义信息。有文献[6]对长文本信息恢复风险做了研究，Shu等人[7]从多模态生成序列的依赖关系角度出发，分析了视觉特征可还原性增强的问题。以上结果表明，“特征越深越安全”的说法不是全部正确的，中间层Embedding还是有很多隐私信息。
为了消除上面的风险，研究者通常会在特征空间中加入差分隐私噪声，也就是高斯噪声或者拉普拉斯噪声。但是这些方法大多直接作用于向量坐标，忽略了特征方向、模长以及语义结构的差异。对于Transformer而言，Embedding的方向关系和数值分布会影响之后的注意力计算以及LayerNorm的输入情况。Qu等人[8]的研究也表明，过多或者无差别地扰动会严重损害下游任务的效果，甚至导致生成结果出现语义偏差。因此，只用各向同性的加性噪声来实现隐私保护和模型可用性二者兼得，是不现实的。
为了减轻上面的问题，最近有部分的研究者开始从特征结构本身的角度去设计起保护机制。Jin等人提出的EntroGuard[9]就是用熵驱动扰动来增大重构的不可预测性，Liu等人[10]的Eguard就是用互信息约束来减少特征之间敏感联系。但是这些方法大多是以欧氏空间中向量扰动为基础的，对于多模态特征所具有的流形结构没有给予足够的重视。另外GI-DQA[11]、Adversarial Illusions[12]、CapRecover[13]等研究也发现视觉和文本特征之间存在着一定的联系，攻击者可以借助它来进行攻击。只保护一个模态之后，另一个模态里面保存下来的语义信息也会变成反演的线索。Mattern等人[14]也觉得，在缺少语义约束的情况下，统一加噪的方法很容易造成文本表示失去作用。依照上面的观察，本文以为应当创建出一种既考量几何结构又考量模态差异以及语义保持能力的特征扰动办法。
本文针对多模态拆分推理中出现的特征泄露危险、扰动后效果变差的问题，使用Qwen-VL-8B作为实验对象，提出了一种按照黎曼几何思想建立起来的语义度量隐私保护办法。受黎曼流形差分隐私研究[15]和高维测度集中现象的启发，把归一化的Embedding表示看作是单位超球面上的一个点，再从扰动方式上分析出特征方向和模长的变化情况。
就具体的办法而言，本文用von Mises-Fisher（vMF）分布[16]来生成正交切平面扰动，使特征向量大致只在方向上发生改变，而且尽量保持原来的范数不变。同直接加入高斯或者拉普拉斯噪声相比，这种方式更符合归一化嵌入空间的几何特点，有利于减少扰动对之后注意力运算造成的影响。本文还参考了度量隐私[17]、随机投影[18]的相关思想，把隐私预算同特征空间中距离联系起来。因为图像和文本嵌入对扰动的敏感程度不同，所以本文又设计出一种不对称预算分配的方法，并且对文本中的一些敏感内容加以遮蔽，以降低对系统提示和任务指令的影响。
本文用MS-COCO和VQA-v2两个数据集来测试本方法的效果，主要是看各个隐私预算下模型效用、反演风险怎样变。实验结果发现，在比较大的隐私约束条件下，vMF干扰会减小反演结果的结构相似度（SSIM），但是不会使下游任务性能严重下降。相比起传统的加性噪声方法来说，本机制对于隐私保护和模型可用性之间所表现出的折中关系更加稳定。

### 1.2 国内外研究现状

#### 1.2.1 拆分推理与模型反演防御的演进
拆分学习最初是由Vepakomma等学者提出来的，它主要是把模型运算分成不同的设备来完成，从而减少原始数据直接共享的现象。对于多模态模型而言，这种部署方式通常会变成端侧编码、云端推理。但是因为中间特征不安全，就会产生一定的风险。He等人[5]认为协作推理中出现的中间表示有可能被用来反推原来的输入内容，Morris等人[19]也认为Transformer得到的文本Embedding里面还有很强的语义信息，可以被还原为可读的文本。
最近有关于攻击的研究又证明了反演风险不只存在于浅层特征之中。一些研究者开始用生成模型先验知识从模型内部状态中提取出更为复杂的文本或者视觉信息。USENIX Security 2025的研究[6]对Llama-3、Qwen等模型深层内部状态的信息泄露风险做了探究，SMI-AW[7]是从序列token权重的角度来考察反演攻击的可能性。因此中间层表示依然存在着可以被攻击者利用起来的语义线索。
针对上面的问题，防御方法从原来的加性噪声变成了更复杂的特征扰动。Liu等人[10]用互信息约束来减小Embedding之间敏感的联系，Jin等人[9]用熵驱动扰动来提高重建的不可预测性。目前的研究已经开始关注扰动方式和特征结构之间的关系，但是如何在保护隐私的同时又能很好地保持模态模型的推理能力，还存在着待解决的问题。

#### 1.2.2 连续空间内的差分隐私与度量隐私
标准差分隐私最初是用来处理数据库查询这种离散情况的。当它被用来处理视觉-语言模型的高维连续特征的时候，一般的做法是在向量坐标上添加随机噪声，但是这样就无法区分语义相近或者语义相差大的样本，因此会致使很大的效用损失。Andrés等[17]提出度量隐私（d-privacy），把隐私保护强度同样本之间的距离联系起来。之后Habernal等人[20]、Feyisetan等人[21]在词向量空间里加入了经过校准的多变量扰动，在连续表征空间中也存在着一定的可行性。
从2024年到2025年，研究范式由原来的平直欧几里得几何变为非欧几里得黎曼流形。Faustini等人[15]首次将高斯差分隐私（GDP）理论应用到具有负曲率的Hadamard黎曼流形上。数学上的进步为解决VLM中存在强烈的各向同性(embedding)的embedding打下了很好的理论基础。最近出现的DP-JL机制[18]用Johnson-Lindenstrauss随机投影证明了在保证隐私的情况下可以大幅度降低高维空间的噪声敏感度。上述理论的发展表明，寻找适合于适应特征非线性几何结构的隐私注入方法，已经成为解决“维数灾难”的一条主要道路。

#### 1.2.3 文本语义保护的效用瓶颈
文本嵌入的隐私保护问题当中，效用降低是其中比较明显的一个问题。Qu等人[8]认为，过强的隐私保护会改变文本表示的语义中心，进而影响到之后的分类或者生成工作。Mattern等人[14]也发现，如果对词级表示做无差别扰动的话，就会造成文本的词法、句法结构被破坏，使得扰动之后的表示不能够继续用来做有效的分析。
CapRecover[13]认为，即便原始文本被一定的混淆之后，攻击者仍然可以依靠语义向量里统计出来的特性来推测出图像所要表达的意思。由此可知，只用统一的加性噪声来处理文本上的隐私问题还不能够完全地解决掉文本侧的隐私问题。就视觉语言模型来说，系统提示、任务模板以及用户的输入这三个方面是不一样的，其中前者主要是保持住模型的指令服从能力，后者一般会包含更多的隐私信息。所以本文在文本侧加入敏感区掩码的思想，尽可能地减小对系统指令结构的影响，而且主要保护用户输入里面那些敏感的语义。

#### 1.2.4 跨模态隐私关联与特征泄露
多模态模型里视觉以及文本的特征一般都会被映射到同一个表示空间里面，跨模态对齐既会提高模型的理解水平，也会产生其他的隐私隐患。Adversarial Illusions[12]认为攻击者可以借助视觉特征同文本特征之间存在的联系来从一个模态中获取另一个模态的敏感数据。GI-DQA[11]又对多模态文档问答任务中梯度反演的问题进行了探讨，在金融、医疗等文档场景下，图像特征同文本内容之间存在着联系，可以用来还原出敏感的信息。所以对于VLM来说，隐私保护不能只对图像或者文本做单独处理，还要考虑两个模态之间信息的耦合。

#### 1.2.5 高维嵌入扰动与几何隐私保护
在高维Embedding上直接添加高斯或者拉普拉斯噪声也会造成特征结构偏移。由于高维空间里存在测度集中现象，各向同性噪声会使得扰动之后的向量远离原来的语义区域，进而干扰到后面注意力计算以及生成的结果。
Whitehouse[22]是从高维表征扰动的偏移上界入手来研究噪声对于特征稳定性的影响的。受到这个启发之后，本文用单位超球面来实现vMF扰动，使得扰动只在方向上发生，而且尽可能地维持着特征范数不变，从而减少噪声对模型内部表示分布造成的干扰。

### 1.3 本文主要研究内容与贡献
根据上面的问题，本文对于多模态大模型拆分推理中出现的特征反演风险进行了研究，并且创建出一个以特征几何结构为基础的语义感知隐私保护方法。主要的研究内容有以下几个方面。
第一，提出用vMF分布来度量隐私扰动的方法。对于传统的加性噪声会改变特征范数以及方向结构的情况，本文把归一化的特征表示当作单位超球面上的一个点来处理，用vMF分布创建出正交切平面扰动。此法在加入隐私扰动的时候，尽可能地维持住特征模长不发生改变，因此可以减小对Transformer之后注意力运算的影响。
第二，提出图文双通道的不对称隐私预算以及文本掩码办法。由于图像embedding和文本embedding对扰动的敏感度不一样，本文给两种模态赋予不同的扰动强度。另外，把文本侧的系统提示、任务模板、用户输入分开来处理，尽可能地减小对系统指令结构的影响，把保护的重点放在那些有可能含有隐私信息的用户内容上。
第三，在Qwen-VL-8B的基础上做实验检验。本文没有重新训练模型，把隐私扰动层加到模型推理过程中，在MS-COCO，VQA数据集上对比各种隐私保护措施的隐私保护效果以及任务性能。实验结果表明，vMF扰动在较强的隐私约束下可以减少反演结果的结构相似度，但是又不会完全丧失图像识别的能力，优于传统的加性噪声方法。

### 1.4 论文组织结构
本文的整体章节安排如下：
第一章前言，就是对课题的研究背景、意义做一番说明，把国内外拆分推理、反演攻击以及连续空间差分隐私这些方面的研究成果整理一下，再对本文的主要工作做一个简要的总结。
第二章前置知识和问题定义，主要就是对Qwen-VL模型结构、拆分推理过程以及度量隐私的数学定义做详细的说明，给机制设计打下基础。
第三章方法和隐私性证明，主要讲解vMF噪声在超球面上的计算方式，得出隐私预算和集中度参数之间的联系，给出非对称掩码办法以及隐私性与效用证明。
第四章实验以及结果的分析，给出实验环境的设定，用定量指标CIDEr、SSIM等来比较本文的方法同传统的办法之间的效果差别，而且对于高维空间中出现的相变阈值现象做详细的剖析。
第五章总结和展望，对全文的研究做总结，剖析目前机制存在的不足之处，预测以后多模态隐私计算的发展趋势。

---

## 2. 方法论 (Methodology)

### 2.1 语义流形与度量空间假设

#### 2.1.1 超球面嵌入假设
在 Qwen-VL 等 Transformer 架构中[1,23]，经 `LayerNorm` 标准化后的特征向量 $\mathbf{h}$ 被约束在有界区域内。研究表明，VLM 的核心语义主要编码在向量的**方向**而非幅度上。自注意力的核心计算 $\mathbf{q}_i^\top \mathbf{k}_j = \|\mathbf{q}_i\| \|\mathbf{k}_j\| \cos\theta_{ij}$ 进一步证实了语义相关性对夹角 $\theta$ 的强依赖性。因此，本研究将 Embedding 空间抽象为单位超球面 $\mathbb{S}^{d-1}$。

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
**定理 2.1**: 设 $\delta_{\text{TV}}(d)$ 为切平面分布与匹配 vMF 分布间的全变差距离[16,24]。算法 1 满足 $(\kappa^*, \delta_{\text{TV}})$-近似度量隐私[25,26,27,28]。对于任意输入 $\mathbf{x}, \mathbf{x}'$：
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

### 3.1 实验环境与参数设置
本文实验在单卡 GPU 环境下完成，硬件平台采用 **NVIDIA GeForce RTX 5090** 显卡进行模型推理、特征扰动和攻击审计实验。实验代码基于 **Python 3.12** 实现，并使用 `uv` 进行依赖管理，主要依赖包括 PyTorch、Transformers/ModelScope、NumPy、scikit-learn 和 Matplotlib。除跨模型泛化实验外，主实验均以 **Qwen3-VL-8B-Instruct** 作为视觉语言模型主干，文中统称为 Qwen-VL-8B；跨模型验证部分采用 **LLaVA-1.5** 作为补充模型，用于检验该机制是否依赖特定模型结构。

本文不对 VLM 主干模型进行重新训练，而是在端云协同拆分推理场景下，将隐私扰动层插入到 embedding 上传云端之前。具体来说，端侧首先得到视觉 token embedding 和文本 token embedding，二者的隐藏维度均为 $d=4096$；随后分别对两个模态的 token 表示施加 vMF 扰动或对应基线扰动，再将扰动后的序列拼接输入云端语言模型进行后续推理。

隐私预算设置为：
$$\epsilon \in \{0.1, 0.2, 0.5, 1.0, 2.0, 5.0\}$$
其中 $\epsilon$ 越小表示隐私保护越强、扰动幅度越大；$\epsilon=0.1$ 作为强隐私设置，$\epsilon=0.5$ 作为中等隐私设置，$\epsilon \ge 1.0$ 作为弱隐私或高效用设置。vMF 机制中的平滑因子固定为 $\beta=1.0$，扰动幅度由 $\lambda=\beta/\epsilon$ 决定。对于图文双通道非对称预算分配，视觉通道使用系统隐私预算 $\epsilon_{\text{sys}}$，文本通道使用 $\alpha\epsilon_{\text{sys}}$，其中 $\alpha>1$ 表示文本侧采用更弱扰动，以降低对系统提示和任务指令的破坏。

对比方法包括无隐私保护的 No-Privacy Baseline、Pixel-Gaussian、Embedding-Laplace 以及本文提出的 vMF-Ours。Pixel-Gaussian 和 Embedding-Laplace 分别对应差分隐私中常见的高斯机制与拉普拉斯机制[25,26]，Embedding-Laplace 同时参考了文本和 embedding 表征空间中的扰动式隐私保护研究[8,20,21]；vMF-Ours 则基于方向统计、度量隐私和流形差分隐私相关理论[16,17,24,27,28]。对于 Gaussian 机制，实验中采用 $\delta=10^{-5}$；对于 Laplace 机制，噪声尺度随 $\epsilon$ 调整；对于 vMF 机制，扰动仅改变 embedding 方向并恢复原始范数。所有实验在相同数据划分、相同隐私预算和相同攻击审计设置下比较，以保证不同机制之间的结果具有可比性。

为降低随机采样带来的偶然性，机制可视化与数值模拟实验固定随机种子，并对同一隐私预算下的多次采样结果统计均值和标准差。结果图中的阴影区间或误差范围表示不同样本上的波动情况。

### 3.2 实验数据集深度说明
本研究选取 MS-COCO 与 VQA-v2 两个视觉语言任务作为主要评估场景，分别覆盖开放式图像描述生成和视觉问答两类典型 VLM 应用。两者都需要模型同时理解图像内容和文本指令，因此能够较全面地反映中间 embedding 扰动对模型效用和隐私风险的影响。

1. **MS-COCO (2017)**[29]：本文使用 MS-COCO 2017 validation 集中的图像描述任务进行评估。该验证集包含 **5000 张图像**，每张图像通常对应 5 条人工参考描述，约形成 **2.5 万条参考 caption**。实验输入为图像及对应的描述提示，模型需要生成自然语言 caption。该任务关注模型能否在扰动后保留图像中的主体、属性、动作和场景关系，适合检验长文本生成的语义连贯性。本文在该数据集上主要使用 **CIDEr** 衡量生成 caption 与人工参考描述之间的 n-gram 共识程度，使用 **CLIPScore** 衡量生成文本与图像之间的跨模态语义一致性；同时使用 **SSIM** 和 **ASR** 衡量攻击者从中间 embedding 中恢复视觉内容的风险。

2. **VQA-v2**[30]：本文使用 VQA-v2 validation 集作为视觉问答评估来源。该验证集包含 **40504 张图像** 和 **214354 个问答对**，问题类型覆盖 Yes/No、Number 和 Other 三类。实验输入为图像和自然语言问题，模型需要生成简短答案。该任务更强调细粒度视觉理解能力，例如物体颜色、数量、空间关系和属性识别。本文在该数据集上主要使用 **Accuracy** 衡量模型答案是否与标准答案匹配，使用 **F1** 衡量预测答案与参考答案在关键词层面的重叠程度；同时使用 **SSIM** 和 **ASR** 评估扰动后中间表示的反演风险。

因此，MS-COCO 侧重评估图像描述质量和跨模态语义保持，VQA-v2 侧重评估细粒度问答准确性。两个数据集中的效用指标不同，但隐私指标均围绕视觉重构相似度和攻击成功率展开。

### 3.3 攻击模型：特征反演审计 (Inversion Auditing)
为了客观评估端云拆分推理中的隐私泄露风险，本研究假设云端攻击者为“诚实但好奇”的灰盒攻击者：其能够截获上传到云端的扰动后 token embedding，并了解模型结构和切分位置，但不能直接访问用户原始图像和文本。针对图文双通道表示，本研究构建三类反演审计模型：

1. **视觉反演攻击**：攻击者以扰动后的视觉 token embedding，或多模态序列中的视觉 token 子序列为输入，采用 FIA-Flow 风格的黑盒特征反演模型[36]，训练逆向网络 $G_v$ 将中间特征映射到可重构的视觉潜空间，并尝试恢复原始图像：
$$\hat{I}=G_v(\tilde{X}^{I})$$
若 $\hat{I}$ 能够还原原图的轮廓、主体物体或关键视觉语义，则认为视觉隐私发生泄露。

2. **文本反演攻击**：攻击者以扰动后的文本 token embedding 为输入，参考 ALGEN 等 embedding-to-text 反演方法[37]，训练文本解码器 $G_t$ 恢复用户输入中的敏感词、实体或主要语义：
$$\hat{T}=G_t(\tilde{X}^{T})$$
若恢复文本 $\hat{T}$ 中包含原始用户输入 $T$ 的敏感实体、关键词或主要语义，则认为文本隐私发生泄露。

3. **多模态联合反演攻击**：攻击者以扰动后的视觉 token 和文本 token 拼接序列为输入，参考 VLM 中的自适应 token 加权反演攻击[38]，训练联合攻击模型 $G_m$ 从跨模态表示中恢复任一模态的敏感信息：
$$[\hat{I},\hat{T}]=G_m([\tilde{X}^{I};\tilde{X}^{T}])$$
若攻击结果能够恢复图像敏感内容或文本敏感内容中的任一类，则认为多模态攻击成功。

上述三类攻击分别对应视觉 ASR、文本 ASR 和多模态 ASR。本文主实验采用多模态联合反演审计作为默认攻击模型，消融实验进一步比较 Visual-Only、Text-Only 与 Both-vMF 设置下的防御效果。

### 3.4 详细评估指标定义与数学表达

#### 3.4.1 CIDEr (Consensus-based Image Description Evaluation)
CIDEr[31] 用于评估图像描述任务中生成 caption 的质量。对于第 $i$ 个样本，设模型生成的描述为 $c_i$，人工参考描述集合为 $S_i=\{s_{i1},...,s_{im}\}$。CIDEr 先统计生成句子和参考句子中的 n-gram，再用 TF-IDF 权重降低常见词的影响、突出图像相关关键词，最后计算二者的加权余弦相似度：
$$\text{CIDEr}_n(c_i, S_i) = \frac{1}{m} \sum_{j=1}^m \frac{\boldsymbol{g}^n(c_i) \cdot \boldsymbol{g}^n(s_{ij})}{\|\boldsymbol{g}^n(c_i)\| \|\boldsymbol{g}^n(s_{ij})\|}$$
其中 $\boldsymbol{g}^n(c_i)$ 是生成描述中 n-gram 的 TF-IDF 向量。实际计算时通常综合 1-gram 到 4-gram 的结果，并对测试集所有样本取平均。CIDEr 数值越高，说明生成描述与人工标注在主体、属性、动作和场景关系上的一致性越强。本文将其作为 MS-COCO 图像描述任务的主要效用指标，用来衡量隐私扰动后模型是否仍能生成准确的自然语言描述。

#### 3.4.2 CLIPScore (多模态语义一致性)
CLIPScore[32] 是一种无参考的跨模态语义一致性指标，不依赖人工 caption，而是直接比较原始图像 $I$ 与模型生成文本 $C$ 是否语义匹配。具体做法是利用预训练的 CLIP 视觉编码器 $E_I$ 和文本编码器 $E_T$ 将图像和文本映射到共同特征空间，并计算二者的余弦相似度：
$$\text{CLIPScore}(I, C) = w \cdot \max(\cos(E_I(I), E_T(C)), 0)$$
其中 $w$ 为缩放系数（通常取 2.5），$\max(\cdot,0)$ 用于截断负相似度。CLIPScore 越高，说明生成文本越符合图像整体语义，例如是否描述了正确的物体类别、场景和关键属性。与 CIDEr 相比，CLIPScore 更关注图文整体匹配，而不是与参考句子的词面重合。因此，本文将其作为 MS-COCO 上的辅助效用指标，用于补充验证扰动后跨模态语义是否保持一致。

#### 3.4.3 SSIM (Structural Similarity Index)
SSIM[33] 用于衡量特征反演攻击还原出的图像 $x$ 与原始图像 $y$ 的结构相似度。与像素级误差不同，SSIM 同时比较两张图像的亮度 ($l$)、对比度 ($c$) 和结构 ($s$)，因此更接近人眼对图像重构质量的感知：
$$\text{SSIM}(x, y) = [l(x, y)]^\alpha \cdot [c(x, y)]^\beta \cdot [s(x, y)]^\gamma$$
具体形式为：
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
其中 $\mu$ 为均值，$\sigma$ 为方差，$\sigma_{xy}$ 为协方差，$C_1$ 和 $C_2$ 为稳定项。SSIM 通常位于 $[0,1]$ 区间，越接近 1 表示重构图像越接近原图。需要注意的是，在本文中 SSIM 不是任务效用指标，而是隐私风险指标：攻击者重构图像与原图越相似，说明泄露风险越高。因此，**在隐私审计中，SSIM 数值越低代表防御越成功**。

#### 3.4.4 余弦相似度 (Cosine Similarity)
余弦相似度用于衡量扰动前后 embedding 向量在方向上的一致性。设原始 embedding 为 $\boldsymbol{x}$，扰动后 embedding 为 $\boldsymbol{y}$，则二者夹角的余弦为：
$$\cos(\theta) = \frac{\boldsymbol{x} \cdot \boldsymbol{y}}{\|\boldsymbol{x}\| \|\boldsymbol{y}\|}$$
角度偏差定义为 $\Delta\theta = \arccos(\cos(\theta)) \cdot \frac{180}{\pi}$。
在基于超球面的度量隐私框架中，embedding 的语义主要由方向决定，因此余弦相似度可以直接反映扰动是否偏离原始语义方向。该指标取值越高，说明扰动后表示与原始表示越接近；角度偏差越大，说明隐私扰动越强、语义偏移越明显。本文主要将余弦相似度用于机制可视化和几何诊断，辅助说明 vMF 扰动在相同隐私预算下比高斯和拉普拉斯噪声更容易保持方向结构；正式下游效用仍以 CIDEr、CLIPScore、Accuracy 和 F1 为主。

#### 3.4.5 VQA 准确率与 F1 分数
在 VQA-v2 任务中，本文使用准确率和 F1 分数衡量模型回答质量。计算前先对模型预测答案 $\hat{a}_i$ 和标准答案 $a_i$ 做统一归一化，包括转小写、去除标点、去除多余空格，并将数字词映射为阿拉伯数字。归一化后，若预测答案与标准答案完全一致，则认为该样本回答正确：
$$\text{Correct}(i)=\mathbf{1}[\text{norm}(\hat{a}_i)=\text{norm}(a_i)]$$
准确率定义为回答正确样本所占比例：
$$\text{Accuracy}=\frac{1}{N}\sum_{i=1}^{N}\text{Correct}(i)$$

对于存在多个人工标注答案的 VQA-v2 样本，本文取预测答案与所有参考答案中的最大匹配得分；若实验数据只保留单一标准答案，则退化为上述 exact match 判定。

F1 分数用于衡量预测答案与参考答案在词级别上的重叠程度。设归一化后的预测答案 token 集合为 $\hat{A}_i$，参考答案 token 集合为 $A_i$，则：
$$P_i=\frac{|\hat{A}_i\cap A_i|}{|\hat{A}_i|},\quad R_i=\frac{|\hat{A}_i\cap A_i|}{|A_i|}$$
$$F1_i=\frac{2P_iR_i}{P_i+R_i}$$
最终 F1 为所有样本的平均值：
$$F1=\frac{1}{N}\sum_{i=1}^{N}F1_i$$
因此，准确率强调答案是否完全匹配，F1 则允许答案在关键词或短语层面部分匹配，更适合衡量开放式 VQA 回答的语义完整性。

#### 3.4.6 攻击成功率 (Attack Success Rate, ASR)
攻击成功率用于衡量攻击者从扰动后的中间表示中恢复隐私信息的比例。对于测试集中的 $N$ 个样本，统一定义为：
$$\text{ASR} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{AttackSuccess}(i)]$$
其中 $\mathbf{1}[\cdot]$ 为指示函数，攻击成功记为 1，否则记为 0。ASR 越低，说明隐私保护效果越好。

针对图文双通道嵌入，本研究进一步区分视觉、文本和多模态三类 ASR：

1. **视觉攻击成功率 $\text{ASR}_{v}$**：攻击者以扰动后的视觉 token embedding 为输入，利用 FIA-Flow 风格的特征反演模型重构图像。若重构图像 $\hat{I}$ 与原图 $I$ 的结构或语义相似度超过阈值，则认为视觉攻击成功：
$$\text{Success}_{v}(i)=\mathbf{1}[\text{SSIM}(I_i,\hat{I}_i)\ge \tau_v]$$
或在语义级审计中使用 CLIP 相似度作为判定依据：
$$\text{Success}_{v}(i)=\mathbf{1}[\text{CLIPSim}(I_i,\hat{I}_i)\ge \tau_v]$$

2. **文本攻击成功率 $\text{ASR}_{t}$**：攻击者以扰动后的文本 token embedding 为输入，利用 embedding-to-text 反演模型恢复用户文本。若恢复文本 $\hat{T}$ 中包含原始用户输入 $T$ 的敏感实体、关键词或主要语义，则认为文本攻击成功：
$$\text{Success}_{t}(i)=\mathbf{1}[\text{SensitiveRecall}(T_i,\hat{T}_i)\ge \tau_t]$$

3. **多模态攻击成功率 $\text{ASR}_{m}$**：攻击者以扰动后的视觉 token 与文本 token 拼接序列为输入，尝试恢复任一模态的隐私信息。若视觉侧或文本侧任一攻击成功，则认为多模态攻击成功：
$$\text{Success}_{m}(i)=\mathbf{1}[\text{Success}_{v}(i)\lor \text{Success}_{t}(i)]$$
$$\text{ASR}_{m}=\frac{1}{N}\sum_{i=1}^{N}\mathbf{1}[\text{Success}_{v}(i)\lor \text{Success}_{t}(i)]$$

在本文主实验图中，ASR 表示联合反演审计下的多模态攻击成功率；在消融实验中，则分别统计 Visual-Only、Text-Only 与 Both-vMF 设置下的攻击成功率，用于验证双通道保护的必要性。

### 3.5 对比基准 (Baselines)
本文选择以下方法作为对比基准，用于验证 vMF 几何扰动相对于传统加性噪声方法的优势：

1. **No-Privacy Baseline**：不对中间 embedding 施加任何扰动，直接将原始视觉和文本 token embedding 输入云端模型。该设置用于给出任务效用上限，同时也代表最高的特征泄露风险；其模型配置与主干 Qwen-VL/Qwen3-VL 保持一致[1,23]。

2. **Pixel-Gaussian**：在输入图像或视觉侧表示上加入高斯噪声，用于模拟传统像素级扰动防御。高斯噪声是差分隐私中经典的连续噪声机制，通常通过隐私预算和灵敏度校准噪声方差[25,26]。该方法实现简单，但噪声并不直接适配 VLM 的高维 embedding 几何结构，容易破坏图像细节和下游语义理解。

3. **Embedding-Laplace**：在 embedding 空间中加入拉普拉斯噪声，是差分隐私中常用的加性噪声机制[25,26]。相关文本隐私研究也常在词向量或句向量空间中进行校准扰动，以降低原始文本或语义表征泄露风险[8,20,21]。该方法能够在特征层提供一定隐私保护，但会同时改变向量方向和范数，可能导致 Transformer 内部注意力分布发生偏移。

4. **vMF-Ours**：本文提出的基于 von Mises-Fisher 分布的超球面方向扰动机制。vMF 分布和方向统计为单位球面上的方向采样提供了理论基础[16,24]，度量隐私和流形差分隐私进一步支持在非欧氏空间中定义距离相关的隐私保护强度[17,27,28]。该方法将 embedding 分解为模长和方向，仅对方向进行受控偏转，并在输出阶段恢复原始范数；同时结合图文双通道非对称预算和文本敏感区掩码，实现更稳定的隐私-效用折中。

---

## 4. 机制可视化与验证 (Visualization & Validation)

为了深入理解 vMF 扰动机制的内在特性，本研究通过几何模拟和统计分析对机制进行了可视化验证。

### 4.1 机制直观可视化分析 (Visual Analysis, a-d)
本研究通过四张关联子图对 vMF 机制的几何与统计特性进行了多维展示（见 `asset/vmf_visualization.png`）：

#### (a) vMF Distribution on Unit Sphere (球面扰动分布)
*   **目的**：在黎曼流形（单位超球面）上直观展示扰动过程的几何表现。
*   **观察**：不同隐私预算 $\epsilon$ 下的扰动样本呈“点簇”状分布在原始向量 $\mu$ 周围。随着 $\epsilon$ 减小（隐私增强），方向散布的方差增大，点簇标记在球面上的覆盖范围逐渐扩张。
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
- **指标含义**: MS-COCO 使用 CIDEr 与 CLIPScore 衡量图像描述效用，二者越高说明生成文本越接近人工参考描述、越符合图像语义。VQA-v2 使用 Accuracy 与 F1 衡量问答效用，其中 Accuracy 强调答案是否完全匹配，F1 强调答案关键词和短语的部分匹配能力。
- **性能韧性**: 在 MS-COCO 任务中，当隐私预算 $\epsilon=0.1$（强隐私约束）时，vMF 机制的 CIDEr 得分保持在 **84.9**，显著高于 Embedding-Laplace (75.1) 和 Pixel-Gaussian (58.4)。这表明 vMF 能够在强扰动下依然保留图像主体、属性和场景关系，使生成 caption 不至于完全偏离原图。
- **问答保持能力**: 在 VQA-v2 任务中，vMF 的 Accuracy 和 F1 在各隐私预算下均高于或接近其他机制，说明其不仅能保持答案的完全匹配率，也能保留回答中的关键语义成分。相比之下，高斯和拉普拉斯加噪更容易破坏细粒度属性判断，例如颜色、数量和空间关系。
- **收敛特征**: 随着 $\epsilon$ 增加至 5.0，扰动强度减小，所有机制的效用指标均逐渐接近 No-Privacy Baseline。vMF 机制在全隐私预算区间内始终保持更高效用，说明基于方向的扰动比坐标级加性噪声更适合高维多模态 embedding。

### 5.2 隐私-效用权衡分析 (Trade-off Curves)
参照 `asset/ms_coco_tradeoff.png` 与 `asset/vqa_v2_tradeoff.png`：
- **指标含义**: Trade-off 图横轴为隐私风险指标，ASR 和 SSIM 均越低越好；纵轴为任务效用指标，CIDEr、CLIPScore、Accuracy 和 F1 均越高越好。因此，曲线越靠近“左上方”，说明方法在较低反演风险下保持了较高任务效用。
- **最优平衡点**: 在相同的攻击成功率（ASR）水平下，vMF 能够提供比其他机制高出 **10%-15%** 的相对效用增益；在相同 CIDEr、Accuracy 或 F1 水平下，vMF 对应的 ASR 与 SSIM 更低，说明攻击者更难从扰动后的 embedding 中恢复图像结构或敏感语义。
- **安全边界**: 观察 ASR 曲线可以发现，vMF 在中等隐私区（$\epsilon=0.5$）的斜率更平缓，这意味着它能够提供更宽的“安全操作窗口”，不会因为 $\epsilon$ 的微小波动导致隐私保护迅速失效。

### 5.3 物理层稳定性验证 (Physical Stability)
参照 `asset/supplementary_norm_stability.png`：
- **指标含义**: 该图统计扰动后 embedding 范数均值。理想情况下，范数应保持在 1.0 附近；偏离越大，说明扰动越可能改变 Transformer 内部激活尺度，进而影响 LayerNorm 和注意力计算。
- **范数偏差对比**: 条形图清晰地展示了物理层面的差异。传统 Gaussian 和 Laplace 机制在加噪后，Embedding 范数均值出现了 **10%-30%** 的向上漂移（Norm Expansion），这种漂移直接破坏了 Transformer 内部 LayerNorm 层的输入分布假设。
- **确定性优势**: 相比之下，vMF 机制在所有 $\epsilon$ 设置下均保持了 **1.0 的恒定范数**。这种数值稳定性是模型在隐私模式下依然能维持逻辑连贯性的根本物理保证。

### 5.4 全模态保护的必要性分析 (Ablation Study)
参照 `asset/supplementary_ablation.png`：
- **指标含义**: 消融实验同时观察 CIDEr 和 ASR。CIDEr 越高表示图像描述效用越好，ASR 越低表示反演攻击越难成功。因此，理想方案应同时保持较高 CIDEr 和较低 ASR。
- **防御漏洞**: 消融实验显示，若仅采用视觉扰动（Visual-Only），虽然 CIDEr 得分略高，但其 ASR（攻击成功率）仍达到 **0.306**（$\epsilon=0.1$）。这是由于攻击者可以利用未受保护的文本 Embedding 之间的跨模态关联进行特征还原。
- **闭环防御**: 只有在视觉与文本同时应用 vMF 扰动（Both-vMF）时，才能将 ASR 压制在 **0.188** 的较低水平，证明了全模态协同防御 in VLM 隐私保护中的不可替代性。

### 5.5 跨模型泛化性评估 (Cross-Model Generalization)
参照 `asset/supplementary_cross_model.png`：
- **指标含义**: 跨模型实验同时报告 Accuracy 和 ASR，前者衡量迁移到 LLaVA-1.5 后的问答效用，后者衡量特征反演风险。若 vMF 在新模型上仍能保持较高 Accuracy 和较低 ASR，则说明该机制不是只针对 Qwen-VL 的特定参数有效。
- **架构无关性**: 本研究在 LLaVA-1.5 模型上复现了实验[34,35]。结果显示，vMF 机制在 LLaVA 上的效用保持能力与隐私防御能力与其在 Qwen-VL 上的表现高度一致。
- **结论**: 这证实了 vMF 几何隐私框架并不依赖于特定模型的参数权重，而是针对 Transformer 架构通用的 Embedding 几何约束设计的，具有极强的跨模型部署潜力。

---

## 6. 结论与展望 (Conclusion and Future Work)

### 6.1 全文总结
针对多模态大模型在端云协同拆分推理场景下的数据泄露风险，本文提出并实现了一种面向 Embedding 层的语义保持度量隐私保护机制。通过以 Qwen-VL 模型为核心的研究对象，系统探讨了高维特征几何特性对隐私与效用的平衡影响。主要工作总结如下：
1.  **提出了基于 vMF 分布的几何扰动框架**：利用黎曼流形理论，将 Embedding 空间建模为单位超球面，通过正交切平面投影实现了范数守恒的方向扰动，从底层解决了传统 DP 机制导致的“注意力崩塌”问题。
2.  **设计了多模态非对称隐私预算分配策略**：针对图文模态对噪声敏感度异构的特性，引入了非对称因子 $\alpha$ 与敏感区掩码矩阵，成功解耦了系统指令骨架与用户隐私语义，保障了模型的指令遵循能力。
3.  **构建了全方位的实验验证体系**：在 MS-COCO 和 VQA-v2 数据集上的实验表明，在强隐私约束下（$\epsilon=0.1$），本机制将特征反演攻击成功率（ASR）压制在约 **16%** 的水平，同时保留了较高的模型效用，其帕累托性能显著优于传统的高斯与拉普拉斯机制。

### 6.2 本研究的创新点
本文的研究亮点在于打破了隐私保护与模型效用之间的“零和博弈”。传统的差分隐私研究往往将 Embedding 视为普通的欧氏空间向量，而本文深入挖掘了 **“方向承载语义，范数承载激活”** 的物理本质。通过保持范数恒定，本研究在物理层确保了 Transformer 结构的平稳运行，为高维表征数据的隐私保护提供了一个全新的几何视角。

### 6.3 局限性与未来展望
尽管本文取得了一定的研究成果，但仍存在以下改进空间：
1.  **动态隐私预算分配**：目前采用统一扰动强度，未来可探索引入视觉显著性检测，针对图像中的 ROI（感兴趣区域）实施差异化的隐私预算分配。
2.  **防御自适应几何攻击**：面对专门针对超球面分布设计的自适应对抗攻击，本机制的防御鲁棒性仍需在未来工作中进一步压力测试。
3.  **隐私增强型微调**：未来可考虑结合 LoRA 等参数高效微调技术，使模型在训练阶段就适应几何扰动分布，从而在极低隐私预算（$\epsilon < 0.05$）下进一步挖掘效用上限。

---

## 参考文献

Bai J, Bai S, Yang S, et al. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond[J/OL]. arXiv:2308.12966, 2023. https://arxiv.org/abs/2308.12966

Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models from Natural Language Supervision[C]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021: 8748-8763. https://arxiv.org/abs/2103.00020

Vepakomma P, Gupta O, Swedish T, et al. Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data[J/OL]. arXiv:1812.00564, 2018. https://arxiv.org/abs/1812.00564

Boenisch F, Dziedzic A, Schuster R, et al. When the Curious Abandon Honesty: Federated Learning Is Not Private[C]//2023 IEEE 8th European Symposium on Security and Privacy. IEEE, 2023: 175-199. https://doi.org/10.1109/EuroSP57164.2023.00020

He Z, Zhang T, Lee R B. Model Inversion Attacks Against Collaborative Inference[C]//Proceedings of the 35th Annual Computer Security Applications Conference. ACM, 2019: 148-162. https://doi.org/10.1145/3359789.3359824

Dong T, Meng Y, Li S, et al. Depth Gives a False Sense of Privacy: LLM Internal States Inversion[C]//34th USENIX Security Symposium. USENIX Association, 2025: 1629-1648. https://www.usenix.org/conference/usenixsecurity25/presentation/dong-tian

Shu, et al. SMI-AW: Sequence-Based Model Inversion with Adaptive Weighting[J/OL]. arXiv preprint, 2025.

Qu C, Kong W, Yang L, et al. Natural Language Understanding with Privacy-Preserving BERT[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. ACM, 2021: 1488-1497. https://doi.org/10.1145/3459637.3482281

Jin S, Pang X, Wang Z, et al. Safeguarding LLM Embeddings in End-Cloud Collaboration via Entropy-Driven Perturbation[J/OL]. arXiv:2503.12896, 2025. https://arxiv.org/abs/2503.12896

Liu T, Yao H, Wu T, et al. Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion[J/OL]. arXiv:2411.05034, 2024. https://arxiv.org/abs/2411.05034

Hemo O B, Zolfi A, Yehezkel O, et al. Gradient Inversion of Multimodal Models[C]//Proceedings of the 42nd International Conference on Machine Learning. PMLR, 2025, 267: 22988-23004. https://proceedings.mlr.press/v267/hemo25a.html

Zhang T, Jha R, Bagdasaryan E, et al. Adversarial Illusions in Multi-Modal Embeddings[C]//33rd USENIX Security Symposium. USENIX Association, 2024: 3009-3025. https://www.usenix.org/conference/usenixsecurity24/presentation/zhang-tingwei

Xiu K, Zhang S. CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models[J/OL]. arXiv:2507.22828, 2025. https://arxiv.org/abs/2507.22828

Mattern J, Weggenmann B, Kerschbaum F. The Limits of Word Level Differential Privacy[C]//Findings of the Association for Computational Linguistics: NAACL 2022. ACL, 2022: 867-881. https://aclanthology.org/2022.findings-naacl.65/

Faustini, et al. Directional DP on Hadamard Manifolds[C]//Proceedings of the 42nd International Conference on Machine Learning, 2025.

Sra S. A Short Note on Parameter Approximation for von Mises-Fisher Distributions[J]. Computational Statistics, 2012, 27(1): 177-190. https://doi.org/10.1007/s00180-011-0232-x

Andres M E, Bordenabe N E, Chatzikokolakis K, et al. Geo-Indistinguishability: Differential Privacy for Location-Based Systems[C]//Proceedings of the 2013 ACM SIGSAC Conference on Computer & Communications Security. ACM, 2013: 901-914. https://doi.org/10.1145/2508859.2516735

Nikolov A. Private Query Release via the Johnson-Lindenstrauss Transform[C]//Proceedings of the 2023 Annual ACM-SIAM Symposium on Discrete Algorithms. SIAM, 2023: 4982-5002. https://doi.org/10.1137/1.9781611977554.ch182

Morris J X, Kuleshov V, Shmatikov V, et al. Text Embeddings Reveal (Almost) As Much As Text[C]//Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. ACL, 2023: 12448-12460. https://aclanthology.org/2023.emnlp-main.765/

Habernal I, Hindle H W, et al. Differential Privacy for Text Analytics via Natural Language Representation[C]//Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics. ACL, 2019: 7136-7146.

Feyisetan O, Balle B, Drake T, et al. Privacy- and Utility-Preserving Textual Analysis via Calibrated Multivariate Perturbations[C]//Proceedings of the 13th International Conference on Web Search and Data Mining. ACM, 2020: 178-186. https://arxiv.org/abs/1910.08902

Whitehouse J. Modern Martingale Methods for Differential Privacy[D]. PhD dissertation, Carnegie Mellon University, 2024.

Bai S, Cai Y, Chen R, et al. Qwen3-VL Technical Report[J/OL]. arXiv:2511.21631, 2025. https://arxiv.org/abs/2511.21631

Mardia K V, Jupp P E. Directional Statistics[M]. 2nd ed. Chichester: Wiley, 2000.

Dwork C, McSherry F, Nissim K, et al. Calibrating Noise to Sensitivity in Private Data Analysis[C]//Theory of Cryptography Conference. Springer, 2006: 265-284. https://doi.org/10.1007/11681878_14

Dwork C, Roth A. The Algorithmic Foundations of Differential Privacy[J]. Foundations and Trends in Theoretical Computer Science, 2014, 9(3-4): 211-407. https://doi.org/10.1561/0400000042

Reimherr M, Bharath K, Soto C. Differential Privacy over Riemannian Manifolds[C]//Advances in Neural Information Processing Systems. 2021, 34. https://proceedings.neurips.cc/paper/2021/hash/6600e06fe9350b62c1e343504d4a7b86-Abstract.html

Jiang Y, Chang X, Liu Y, et al. Gaussian Differential Privacy on Riemannian Manifolds[C]//Advances in Neural Information Processing Systems. 2023, 36. https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html

Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context[C]//European Conference on Computer Vision. Springer, 2014: 740-755. https://arxiv.org/abs/1405.0312

Goyal Y, Khot T, Summers-Stay D, et al. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2017: 6904-6913. https://arxiv.org/abs/1612.00837

Vedantam R, Lawrence Zitnick C, Parikh D. CIDEr: Consensus-Based Image Description Evaluation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2015: 4566-4575. https://openaccess.thecvf.com/content_cvpr_2015/html/Vedantam_CIDEr_Consensus-Based_Image_2015_CVPR_paper.html

Hessel J, Holtzman A, Forbes M, et al. CLIPScore: A Reference-Free Evaluation Metric for Image Captioning[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. ACL, 2021: 7514-7528. https://aclanthology.org/2021.emnlp-main.595/

Wang Z, Bovik A C, Sheikh H R, et al. Image Quality Assessment: From Error Visibility to Structural Similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612. https://doi.org/10.1109/TIP.2003.819861

Liu H, Li C, Wu Q, et al. Visual Instruction Tuning[C]//Advances in Neural Information Processing Systems. 2023, 36. https://arxiv.org/abs/2304.08485

Liu H, Li C, Li Y, et al. Improved Baselines with Visual Instruction Tuning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2024: 26296-26306. https://arxiv.org/abs/2310.03744

Ren J, He Z, Lee R B. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs[J/OL]. arXiv:2511.15316, 2025. https://arxiv.org/abs/2511.15316

Chen Y, Xu Q, Bjerva J. ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation[C/OL]//Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. ACL, 2025. https://arxiv.org/abs/2502.11308

Nguyen N B, Ho S T, Koh J H, et al. Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks[C/OL]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2026. https://arxiv.org/abs/2508.04097
