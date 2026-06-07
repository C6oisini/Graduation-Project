# 陈凡-毕业论文

![图](../asset/docx_media/image1.jpeg)

本 科 毕 业 设 计（论文）

| 题目 | 面向视觉语言模型的语义保持度量隐私研究 |
| --- | --- |
| 作者 | 陈凡 |
| 学院 | 计算机科学与工程学院 |
| 专业 | 信息安全 |
| 学号 | 2205030208 |
| 校内指导老师 | 张少波 |
| 校外指导老师 | 赵雄伟 |

二〇二 六 年 5 月 20 日

## 摘 要

在端云协同的拆分推理场景中，多模态大模型通常需要将中间层特征从终端传输到云端，这使特征反演攻击成为一个需要关注的隐私风险。传统差分隐私方法直接在特征上加入噪声，虽然能够增强隐私保护，但也容易破坏高维嵌入空间中的方向信息和范数分布，进而影响模型的视觉理解和文本生成效果。针对这一问题，本文设计了一种面向多模态拆分推理的语义度量隐私机制。该机制将归一化后的特征表示放在单位超球面上处理，并利用 von Mises-Fisher（vMF）分布构造正交切平面扰动，使特征在保持模长基本不变的情况下发生方向偏转。进一步地，本文结合图像和文本嵌入的差异，设置了非对称隐私预算，并对文本中的敏感区域进行掩码处理，以减少对系统提示和任务指令的干扰。基于 Qwen3-VL-8B-Instruct 的实验结果表明，该方法在较强隐私约束下仍能保留一定的任务性能，同时能够降低特征反演攻击带来的信息泄露风险。

关键词：多模态大模型；拆分推理；度量隐私；von Mises-Fisher 分布；语义保持

## ABSTRACT

To address the risks of intermediate feature inversion leakage in Multimodal Large Models (MLMs) within end-cloud collaborative split inference architectures, as well as the collapse of model utility caused by the destruction of high-dimensional semantic structures in traditional differential privacy mechanisms, this paper proposes a Semantic Metric Privacy mechanism tailored for multimodal split inference. By modeling the high-dimensional feature space as a unit hypersphere on a Riemannian manifold, this mechanism pioneers the introduction of the von Mises-Fisher (vMF) distribution to implement orthogonal tangent plane projection perturbations. It achieves statistical indistinguishability in high-dimensional metric spaces by performing controlled vector direction deflections while strictly maintaining norm conservation. Furthermore, considering the heterogeneous nature of image and text modalities, a dual-channel asymmetric privacy budget allocation and a text-sensitive masking strategy are designed to precisely decouple system instruction skeletons from user privacy semantics. Plug-and-play deployment experiments on the Qwen3-VL-8B-Instruct model demonstrate that the proposed mechanism exhibits significant semantic phase-transition advantages in strong privacy regimes. While effectively defending against pixel-level feature reconstruction attacks, it maintains high performance in visual understanding and text generation, successfully breaking the zero-sum game between privacy protection and the utility of large model inference in high-dimensional feature spaces.

Keywords: Multimodal Large Models; Split Inference; Metric Privacy; von Mises-Fisher Distribution; Semantic-Preserving

## 目 录

- 第一章 前 言 - 1 -

  - 1.1 研究背景与研究意义 - 1 -

  - 1.2 国内外研究现状 - 2 -

    - 1.2.1 拆分推理与模型反演防御的演进 - 2 -

    - 1.2.2 连续空间内的差分隐私与度量隐私 - 3 -

    - 1.2.3 文本语义保护的效用瓶颈 - 3 -

    - 1.2.4 跨模态隐私关联与特征泄露 - 4 -

    - 1.2.5 高维嵌入扰动与几何隐私保护 - 4 -

  - 1.3 本文主要研究内容与贡献 - 4 -

  - 1.4 论文组织结构 - 5 -

- 第二章 前置知识与问题定义 - 6 -

  - 2.1 视觉语言模型基础：以 Qwen3-VL 为例 - 6 -

  - 2.2 隐私保护理论基础 - 9 -

    - 2.2.1 差分隐私 - 9 -

    - 2.2.2 度量隐私 - 10 -

    - 2.2.3 von Mises-Fisher分布 - 10 -

    - 2.2.4测地线距离 - 10 -

  - 2.3 问题定义与威胁模型 - 10 -

    - 2.3.1 拆分推理场景 - 10 -

    - 2.3.2 诚实但好奇的攻击者 - 11 -

- 第三章 方法与隐私效用证明 - 12 -

  - 3.1 语义流形与度量空间假设 - 12 -

    - 3.1.1 超球面嵌入假设 - 12 -

    - 3.1.2 嵌入空间度量定义 - 12 -

  - 3.2 vMF 几何扰动框架 - 12 -

    - 3.2.1 算法1：正交切平面投影 - 12 -

    - 3.2.2 算法2：多模态非对称预算分配与掩码机制 - 12 -

    - 3.2.3 算法3：敏感区掩码保护 - 13 -

    - 3.2.4 其他命题与引理 - 13 -

  - 3.3 隐私性证明 - 14 -

    - 3.3.1 主定理：近似度量隐私 - 14 -

    - 3.3.2 辅助引理 - 15 -

    - 3.3.3 主定理证明 - 15 -

  - 3.4 效用性分析 - 17 -

    - 3.4.1 范数保持 - 17 -

    - 3.4.2 余弦相似度 - 17 -

    - 3.4.3 点积偏差 - 18 -

- 第四章 实验结果 - 19 -

  - 4.1 实验环境与参数设置 - 19 -

  - 4.2 实验数据集说明 - 19 -

  - 4.3 攻击模型：特征反演审计 - 20 -

  - 4.4 详细评估指标定义与数学表达 - 21 -

    - 4.4.1 CIDEr - 21 -

    - 4.4.2 CLIPScore - 21 -

    - 4.4.3 SSIM - 22 -

    - 4.4.4 余弦相似度 - 22 -

    - 4.4.5 VQA 准确率与 F1 分数 - 22 -

    - 4.4.6 攻击成功率 - 23 -

  - 4.5 对比基准 - 24 -

  - 4.6 结果分析 - 24 -

    - 4.6.1 主实验效用分析 - 24 -

    - 4.6.2 隐私-效用权衡分析 - 26 -

    - 4.6.3 稳定性验证 - 27 -

    - 4.6.4 全模态保护的必要性分析 - 28 -

    - 4.6.5 跨模型泛化性评估 - 28 -

- 第五章 结论与展望 - 30 -

- 参 考 文 献 - 31 -

- 致 谢 - 34 -

# 第一章 前 言

## 1.1 研究背景与研究意义

近年来Qwen-VL等多模态大语言模型（Multimodal Large Language Models，MLLMs）[1]在图像识别、视觉问答、跨模态推理上都有很好的效果。大多数模型都是用大量的图文配对数据来进行预训练[2]，把图像、文字等各种信息统一到一个嵌入空间中去，给后面的推理和创造赋予相同的表示。但是模型太大也会给端侧设备部署带来负担。Qwen3-VL-8B-Instruct有几十亿个参数，既占用了很大的存储和显存资源，又需要端侧设备具备很强的计算能力。对于移动终端、物联网边缘设备而言，在本地执行完整的模型会耗费很多资源，因此实际应用时通常会采取端云协同或者拆分推理的方式进行部署。

为了缓解端侧设备不能处理全部模型的问题，拆分推理（Split Inference）[3]给人们提供了一种比较好的部署方式。通常把模型分成端侧和云侧两部分，终端用较轻的视觉编码器（Visual Encoder）和文本编码器（Text Encoder）来完成，之后将得到的中间特征（Smash Data）发送到云端，云端再用更强的语言模型部分来继续推理和生成。这样可以避免原始图像或者文本离开本地设备，进而降低原始数据被泄露的风险。但是要注意的是中间特征里还有许多语义信息，如果云端或者传输通道被攻击了，也会存在隐私泄露的风险。

现在学术界很多的研究都表明拆分推理中间特征容易遭受攻击。很多研究者[4][5]认为攻击者即使不能直接获得原始图像，也可以通过嵌入层反演攻击（Embedding Inversion Attack,EIA）从中间表示中获取到部分视觉内容。近两年有关工作的进步又证明了，因为生成模型先验能力增强后，深层特征里还有许多可以被利用的语义信息。有文献[6]对长文本信息恢复风险做了研究，Shu等人[7]从多模态生成序列的依赖关系角度出发，分析了视觉特征可还原性增强的问题。以上结果表明，“特征越深越安全”的说法不是全部正确的，中间层Embedding还是有很多隐私信息。

为了消除上面的风险，研究者通常会在特征空间中加入差分隐私噪声，也就是高斯噪声或者拉普拉斯噪声。但是这些方法大多直接作用于向量坐标，忽略了特征方向、模长以及语义结构的差异。对于Transformer而言，Embedding的方向关系和数值分布会影响之后的注意力计算以及LayerNorm的输入情况。Qu等人[8]的研究也表明，过多或者无差别地扰动会严重损害下游任务的效果，甚至导致生成结果出现语义偏差。因此，只用各向同性的加性噪声来实现隐私保护和模型可用性二者兼得，是不现实的。

为了减轻上面的问题，最近有部分的研究者开始从特征结构本身的角度去设计起保护机制。Jin等人提出的EntroGuard[9]就是用熵驱动扰动来增大重构的不可预测性，Liu等人[10]的Eguard就是用互信息约束来减少特征之间敏感联系。但是这些方法大多是以欧氏空间中向量扰动为基础的，对于多模态特征所具有的流形结构没有给予足够的重视。另外GI-DQA[11]、Adversarial Illusions[12]、CapRecover[13]等研究也发现视觉和文本特征之间存在着一定的联系，攻击者可以借助它来进行攻击。只保护一个模态之后，另一个模态里面保存下来的语义信息也会变成反演的线索。Mattern等人[14]也觉得，在缺少语义约束的情况下，统一加噪的方法很容易造成文本表示失去作用。依照上面的观察，本文以为应当提出一种既考量几何结构又考量模态差异以及语义保持能力的特征扰动办法。

本文针对多模态拆分推理中出现的特征泄露危险、扰动后效果变差的问题，使用Qwen3-VL-8B-Instruct作为实验对象，提出了一种按照黎曼几何思想建立起来的语义度量隐私保护办法。受黎曼流形差分隐私研究[15]和高维测度集中现象的启发，把归一化的Embedding表示看作是单位超球面上的一个点，再从扰动方式上分析出特征方向和模长的变化情况。

就具体的办法而言，本文用von Mises-Fisher（vMF）分布[16]来生成正交切平面扰动，使特征向量大致只在方向上发生改变，而且尽量保持原来的范数不变。同直接加入高斯或者拉普拉斯噪声相比，这种方式更符合归一化嵌入空间的几何特点，有利于减少扰动对之后注意力运算造成的影响。本文还参考了度量隐私[17]、随机投影[18]的相关思想，把隐私预算同特征空间中距离联系起来。因为图像和文本嵌入对扰动的敏感程度不同，所以本文又设计出一种不对称预算分配的方法，并且对文本中的一些敏感内容加以遮蔽，以降低对系统提示和任务指令的影响。

本文用MS-COCO和VQA-v2两个数据集来测试本方法的效果，主要是看各个隐私预算下模型效用、反演风险怎样变。实验结果发现，在比较大的隐私约束条件下，vMF干扰会减小反演结果的结构相似度（SSIM），但是不会使下游任务性能严重下降。相比起传统的加性噪声方法来说，本机制对于隐私保护和模型可用性之间所表现出的折中关系更加稳定。

## 1.2 国内外研究现状

### 1.2.1 拆分推理与模型反演防御的演进

拆分学习最初是由Vepakomma等学者提出来的，它主要是把模型运算分成不同的设备来完成，从而减少原始数据直接共享的现象。对于多模态模型而言，这种部署方式通常会变成端侧编码、云端推理。但是因为中间特征不安全，就会产生一定的风险。He等人[5]认为协作推理中出现的中间表示有可能被用来反推原来的输入内容，Morris等人[19]也认为Transformer得到的文本Embedding里面还有很强的语义信息，可以被还原为可读的文本。

最近有关于攻击的研究又证明了反演风险不只存在于浅层特征之中。一些研究者开始用生成模型先验知识从模型内部状态中提取出更为复杂的文本或者视觉信息。USENIX Security 2025的研究[6]对Llama-3、Qwen等模型深层内部状态的信息泄露风险做了探究，SMI-AW[7]是从序列token权重的角度来考察反演攻击的可能性。国内关于多模态大语言模型安全性的综述也指出，隐私泄露已经成为视觉-语言模型安全评估中的重要问题之一[42]。因此中间层表示依然存在着可以被攻击者利用起来的语义线索。

针对上面的问题，防御方法从原来的加性噪声变成了更复杂的特征扰动。Liu等人[10]用互信息约束来减小Embedding之间敏感的联系，Jin等人[9]用熵驱动扰动来提高重建的不可预测性。目前的研究已经开始关注扰动方式和特征结构之间的关系，但是如何在保护隐私的同时又能很好地保持模态模型的推理能力，还存在着待解决的问题。

### 1.2.2 连续空间内的差分隐私与度量隐私

标准差分隐私最初是用来处理数据库查询这种离散情况的。，国内已有综述对其噪声校准、隐私预算和机器学习场景中的应用进行了系统梳理[39,40]当它被用来处理视觉-语言模型的高维连续特征的时候，一般的做法是在向量坐标上添加随机噪声，但是这样就无法区分语义相近或者语义相差大的样本，因此会致使很大的效用损失。高维数据发布相关研究也指出，维度增长会显著放大差分隐私机制中的噪声敏感性，需要结合降维、随机投影等方法缓解效用损失[41]。Andrés等[17]提出度量隐私（d-privacy），把隐私保护强度同样本之间的距离联系起来。之后Habernal等人[20]、Feyisetan等人[21]在词向量空间里加入了经过校准的多变量扰动，在连续表征空间中也存在着一定的可行性。

从2023年到2025年，研究范式由原来的平直欧几里得几何变为非欧几里得黎曼流形。Faustini等人[15]首次将高斯差分隐私（GDP）理论应用到具有负曲率的Hadamard黎曼流形上。数学上的进步为解决VLM中存在强烈的各向同性(embedding)的embedding打下了很好的理论基础。最近出现的DP-JL机制[18]用Johnson-Lindenstrauss随机投影证明了在保证隐私的情况下可以大幅度降低高维空间的噪声敏感度。上述理论的发展表明，寻找适合于适应特征非线性几何结构的隐私注入方法，已经成为解决“维数灾难”的一条主要道路。

### 1.2.3 文本语义保护的效用瓶颈

文本嵌入的隐私保护问题当中，效用降低是其中比较明显的一个问题。Qu等人[8]认为，过强的隐私保护会改变文本表示的语义中心，进而影响到之后的分类或者生成工作。Mattern等人[14]也发现，如果对词级表示做无差别扰动的话，就会造成文本的词法、句法结构被破坏，使得扰动之后的表示不能够继续用来做有效的分析。

CapRecover[13]认为，即便原始文本被一定的混淆之后，攻击者仍然可以依靠语义向量里统计出来的特性来推测出图像所要表达的意思。由此可知，只用统一的加性噪声来处理文本上的隐私问题还不能够完全地解决掉文本侧的隐私问题。就视觉语言模型来说，系统提示、任务模板以及用户的输入这三个方面是不一样的，其中前者主要是保持住模型的指令服从能力，后者一般会包含更多的隐私信息。所以本文在文本侧加入敏感区掩码的思想，尽可能地减小对系统指令结构的影响，而且主要保护用户输入里面那些敏感的语义。

### 1.2.4 跨模态隐私关联与特征泄露

多模态模型里视觉以及文本的特征一般都会被映射到同一个表示空间里面，跨模态对齐既会提高模型的理解水平，也会产生其他的隐私隐患。Adversarial Illusions[12]认为攻击者可以借助视觉特征同文本特征之间存在的联系来从一个模态中获取另一个模态的敏感数据。GI-DQA[11]又对多模态文档问答任务中梯度反演的问题进行了探讨，在金融、医疗等文档场景下，图像特征同文本内容之间存在着联系，可以用来还原出敏感的信息。所以对于VLM来说，隐私保护不能只对图像或者文本做单独处理，还要考虑两个模态之间信息的耦合。

### 1.2.5 高维嵌入扰动与几何隐私保护

在高维Embedding上直接添加高斯或者拉普拉斯噪声也会造成特征结构偏移。由于高维空间里存在测度集中现象，各向同性噪声会使得扰动之后的向量远离原来的语义区域，进而干扰到后面注意力计算以及生成的结果。

Whitehouse[22]是从高维表征扰动的偏移上界入手来研究噪声对于特征稳定性的影响的。受到这个启发之后，本文用单位超球面来实现vMF扰动，使得扰动只在方向上发生，而且尽可能地维持特征范数不变，从而减少噪声对模型内部表示分布造成的干扰。

## 1.3 本文主要研究内容与贡献

根据上面的问题，本文对于多模态大模型拆分推理中出现的特征反演风险进行了研究，并且创建出一个以特征几何结构为基础的语义感知隐私保护方法。主要的研究内容有以下几个方面。

第一，提出用vMF分布来度量隐私扰动的方法。对于传统的加性噪声会改变特征范数以及方向结构的情况，本文把归一化的特征表示当作单位超球面上的一个点来处理，用vMF分布创建出正交切平面扰动。此法在加入隐私扰动的时候，尽可能地维持住特征模长不发生改变，因此可以减小对Transformer之后注意力运算的影响。

第二，提出图文双通道的不对称隐私预算以及文本掩码办法。由于图像embedding和文本embedding对扰动的敏感度不一样，本文给两种模态赋予不同的扰动强度。另外，把文本侧的系统提示、任务模板、用户输入分开来处理，尽可能地减小对系统指令结构的影响，把保护的重点放在那些有可能含有隐私信息的用户内容上。

第三，在Qwen3-VL-8B-Instruct的基础上做实验检验。本文没有重新训练模型，把隐私扰动层加到模型推理过程中，在MS-COCO，VQA数据集上对比各种隐私保护措施的隐私保护效果以及任务性能。实验结果表明，vMF扰动在较强的隐私约束下可以减少反演结果的结构相似度，但是又不会完全丧失图像识别的能力，明显优于传统的加性噪声方法。

## 1.4 论文组织结构

本文的整体章节安排如下：

第一章前言，就是对课题的研究背景、意义做一番说明，把国内外拆分推理、反演攻击以及连续空间差分隐私这些方面的研究成果进行整理，再对本文的主要工作做一个简要的总结。

第二章前置知识和问题定义，主要就是对Qwen-VL模型结构、拆分推理过程以及度量隐私的数学定义做详细的说明，给机制设计打下基础。

第三章方法和隐私性证明，主要讲解vMF噪声在超球面上的计算方式，得出隐私预算和集中度参数之间的联系，给出非对称掩码办法以及隐私性与效用证明。

第四章实验以及结果的分析，给出实验环境的设定，用定量指标CIDEr、SSIM等来比较本文的方法同传统的办法之间的效果差别，而且对于高维空间中出现的相变阈值现象做详细的剖析。

第五章总结和展望，对全文的研究做总结，剖析目前机制存在的不足之处，预测以后多模态隐私计算的发展趋势。

# 第二章 前置知识与问题定义

## 2.1 视觉语言模型基础：以 Qwen3-VL 为例

视觉语言模型旨在实现图像与文本特征的深度对齐。以本研究采用的 Qwen3-VL-8B-Instruct 为例，其核心架构由三个关键组件构成：

- 视觉编码器 (Visual Encoder：采用预训练的 ViT (Vision Transformer) 结构：把原始图像 $I∈R^{H×W×C}$ 切分为固定大小的补丁（Patches），这些补丁经过线性投影并结合位置编码（Positional Embedding），被映射为视觉特征向量 $X_{vis}={v_{1},v_{2},…,v_{n}}$，经过多层自注意力（Self-Attention）计算，每个向量 $v_{i}$ 包含了其对应区域及其周围的丰富语义信息。

- 对齐投影层 (Visual-Language Adapter/Projector)：因为视觉编码器给出的特征空间和语言模型的语义空间在维度以及分布上不一致，所以对齐投影层起着连接的作用，投影层一般是由带有激活函数的线性层或者轻量级MLP构成的，把视觉特征从原来的维度映射到和语言模型基座一样的维度空间，投影过程保证了视觉特征在保持拓扑结构的情况下，数值分布可以符合语言模型对于Token Embedding的分布期望。本文抓住并执行隐私扰动的主要环节就是处在该投影层输出之后的中间层表示（Smash Data）。Qwen-VL把视觉Embedding当作特殊的“视觉Token”，同文本Token一起进入Transformer块。

- 语言模型基座 (LLM Backbone): 语言模型基座是 VLM 进行逻辑推理与文本生成的最后一步。它接收由投影后的视觉序列 $\hat{X}vis$ 与文本词元序列 $Xtxt$ 拼接而成的统一输入流。模型通过多头自注意力（Multi-Head Attention）计算模态间交互最后输出文本Token，并循环进行直到结束。

![图](../asset/docx_media/image2.jpeg)

图2.1：Qwen3-VL框架示意图

Qwen3-VL的目标就是使模型不仅可以看见图片或者视频，而且可以真正认识这个世界、明白事情、作出反应。因此Qwen3-VL在各个主要的能力方面都进行了全面的改进，希望把视觉大模型由“感知”变成“认知”，由“识别”发展到“推理和执行”。它的主要能力如下

- 视觉智能体（Visual Agent），Qwen3-VL可以操控电脑以及手机的界面，辨识GUI元素，明白按钮的作用，使用工具，完成工作，在OS World等benchmark上取得世界一流的成果，可以借助调用工具来提高对于细粒度感知任务的成效。

- 纯文本能力可以和顶级语言模型媲美，Qwen3-VL在预训练阶段就加入了文本和视觉模态一起训练，文本能力不断加强，在纯文本任务上可以达到Qwen3-VL-235B-A22B-2507纯文本旗舰模型的水平，是真正意义上的文本基础牢固、多模态全面的新型视觉语言模型。

- 视觉编码能力大为提高，可以生成图像的代码和视频的代码，看见设计图之后就可以生成http://Draw.io/HTML/CSS/JS代码，达到“所见即所得”的效果。

- 空间感知能力大大增强，2D grounding由原来的绝对坐标转变为相对坐标，可以判断出物体的位置、方向、角度以及遮挡情况，可以完成3D grounding，给复杂的环境推理以及具身环境打下基础。

- 长上下文支持以及长视频理解，全系列模型都自带256K token的上下文长度，可以达到100万token的程度。因此不管是一百多页的技术文档、一整本的教材，还是两个小时的视频，都可以全部输入进去、全程记住、准确查找，而且可以精确地找到视频中每一个秒的时间点。

- 多模态思考能力大大加强，Thinking模型主要改善了STEM和数学推理的能力。对于专业的学科问题，模型可以抓住细节、拆解问题、找出原因、得出结论，并且给出有条理、有根据的回答，在MathVision、MMMU、MathVista等权威评价里处于领先地位。

- 视觉感知和识别能力得到了很大的提高，经过改善了预训练数据的质量和范围之后，现在的模型可以识别出更多的物体种类，包括名人、动漫人物、商品、地标、动植物等等，可以满足日常生活中以及专业领域里所有的“万物识别”的要求。

- OCR可以支持更多的语言和复杂的环境，可以支持的语言数量由原来的10种增加到现在的32种，可以涵盖更多的国家和地域，在复杂的光线、模糊、倾斜等实拍困难的环境下仍然保持良好的稳定性，对于生僻字、古籍字、专业术语的识别准确率也有明显提高，超长文档的理解以及精细结构的还原能力也得到了改善。

本文用通义千问团队最近发布出来的Qwen3-VL-8B-Instruct做为主要的实验模型。它是目前开源多模态大语言模型（MLLMs）当中较好的一个，它在性能上和结构上都是很好的例子。它的主要优点有下面几个方面。

![图](../asset/docx_media/image3.png)

图2.2：Qwen3-VL中等模型性能图

1）卓越的多模态感知和细粒度解析能力

依照官方基准测试数据来看，Qwen3-VL-8B-Instruct在文字识别（OCR）和文档理解任务上表现得十分出色。它在OCR Bench上得了896分的好成绩，在DocVQA任务上准确率达到了96.1%。

隐私研究有关联性，高精度的OCR能力使该模型对于图像里敏感的文字（证件号、财务报表）有很强的提取能力，这就更加突出了在本研究当中使用Embedding层隐私扰动来抵抗特征反演攻击的必要性。

2）有很强的逻辑推理和视觉数学才能

该模型对于复杂的视觉推理任务有较大的提升。MathVista（视觉数学基准）取得77.2分的成绩，比同级别其他竞争者（GPT-5 Mini等）好很多。MMLU综合知识测评得分为80.7分。

隐私研究有关联性就是说模型的高推理上限给隐私机制赋予了更大的效用保持的要求。本文的目的就是检验vMF机制是否可以在不破坏它精密逻辑推理链的基础上，达到语义层面的模糊化。

3）兼顾部署效率和推理效用

Qwen3-VL-8B-Instruct 是一个有80亿参数的中等大小模型，在性能方面比以前的大尺寸模型（Qwen2.5-VL-72B）好很多，并且在HRBench4K/8K这样的高清感知任务上表现得也比较出色。8B规模的模型就是目前端云协同、拆分推理等场景里最好的端侧或者边缘侧的候选者，在它上面研究出隐私保护的方法，有很强的工程实现价值和实际的指导作用。

![图](../asset/docx_media/image4.png)

图2.3：Qwen3-VL同家族与GPT/Claude性能对比图

## 2.2 隐私保护理论基础

### 2.2.1 差分隐私

差分隐私通过在计算过程中引入随机噪声，确保查询结果对单个样本的变动不敏感。

$(ϵ,δ)$-DP 定义: 一个随机机制 $M$ 满足隐私预算 $ϵ$ 和松弛参数 $δ$，若对于任意相邻数据集 $D,D^{'}$，有：

$Pr(M(D)∈S)≤e^{ϵ}⋅Pr(M(D^{'})∈S)+δ$ (2.1)

传统的本地差分隐私要求任意两个输入 $x,x^{'}$ 产生的输出分布必须不可区分。这在 Embedding 空间会导致两个语义完全无关的向量被迫产生相似输出，造成巨大的推理效用损失。

### 2.2.2 度量隐私

在超高维空间$d=4096$里，有效的语义特征不是均匀分布的，而是在低维的语义流形上。因为有LayerNorm的存在，所以这些特征就被迫限制到超球面的局部范围内。因此，任何偏离超球面切空间的扰动都会造成语义的灾难性丢失。

对于Embedding这样的高维特征来说，本文使用的是度量隐私框架。它的主要思想就是把隐私保证的程度同语义的距离联系起来$d(⋅,⋅)$

$Pr(M(x)∈S)≤e^{d(x,x^{'})}⋅Pr(M(x^{'})∈S)$ (2.2)

该种定义可以使得我们在特征空间里达成语义相近但是不能被分辨的目的，在保证了细节的情况下还保持了宏观上的逻辑关系。对于VLM来说，用测地线距离来表示隐私预算，把隐私预算直接映射到超球面的几何偏移上。

### 2.2.3 von Mises-Fisher分布

$d$ 维单位超球面 $S^{d-1}$ 上的 vMF 分布 $vMF(μ,κ)$ 的密度函数为：

$f_{vMF}(z;μ,κ)=C_{d}(κ)exp(κμ^{⊤}z)$ (2.3)

其中 $μ∈S^{d-1}$ 为均值方向，$κ≥0$ 为集中度，$C_{d}(κ)=κ^{d/2-1}/((2π)^{d/2}I_{d/2-1}(κ))$ 为归一化常数，$I_{μ}$ 为第一类修正 Bessel 函数。

性质1。$κ = 0$ 时退化为球面均匀分布；$κ→∞$ 时集中于 $μ$ 处的点质量。

性质2。$z∼vMF(μ,κ)$ 时，$E(z)=A_{d}(κ)μ$，其中 $A_{d}(κ)=I_{d/2}(κ)/I_{d/2-1}(κ)$。

### 2.2.4测地线距离

对 $u,v∈S^{d-1}$，测地线距离为：

$d_{geo}(u,v)=arccos(u^{⊤}v)$ (2.4)

即 $S^{d-1}$ 上的大圆弧长。满足度量公理（非负性、同一性、对称性、三角不等式）。

## 2.3 问题定义与威胁模型

### 2.3.1 拆分推理场景

为了解决端侧算力约束和数据安全问题，本文认为应该用端侧编码加云端推理的方法来处理：

- 端侧（Client）执行视觉编码器、token嵌入，得到初步的embedding序列。

- 切分点（Cut Layer）就是数据经过投影层之后被截断，并且加入到隐私扰动里面去。

- 云端接收扰动之后的Embedding，用LLM基座来执行下游的任务。

### 2.3.2 诚实但好奇的攻击者

本研究假设云端服务器是“诚实但好奇”的：

- 诚实: 服务器会严格按照协议执行 VLM 推理并返回结果。

- 好奇: 服务器会尝试利用其拥有的强大算力（及预训练的逆向解码器），从接收到的 Embedding中重构用户的原始视觉细节（如人脸、文字）或推理用户的敏感属性。

攻击目标: 攻击者旨在最大化重构图像与原图的相似度（SSIM）以及还原出用户输入的原始文本。

防御目标: 在保证 VLM 输出质量（Utility）的前提下，将 ASR 和 SSIM 压制到随机水平以下。

# 第三章 方法与隐私效用证明

## 3.1 语义流形与度量空间假设

### 3.1.1 超球面嵌入假设

在 Qwen-VL 等 Transformer 架构[1][23]中，经过LayerNorm标准化之后得到的特征向量被限制在一个有界的范围内。有关研究表明，VLM的主要语义是用向量的方向来表示的而不是幅度。Self Attention的计算结果$q_{i}^{⊤}k_{j}=(q_{i})(k_{j})cosθ_{ij}θS^{d-1}$也说明了语义相关的角度具有很强的依赖关系。所以本文把Embedding空间抽象成单位超球面$S^{d-1}$。

### 3.1.2 嵌入空间度量定义

本研究采用基于测地线距离（弧长）的角度度量定义隐私邻域：

$d_{ε}(x,x^{'})=ε⋅arccos(\frac{x^{⊤}x^{'}}{(x)(x^{'})})$ (3.1)

该度量确保了隐私保护强度与数据点间的语义距离成正比，即“语义相近获强保护，语义相远获弱保护”，这天然契合多模态嵌入的特性。

## 3.2 vMF 几何扰动框架

### 3.2.1 算法1：正交切平面投影

本研究提出的正交切平面投影扰动。对输入 Embedding $x∈R^{d}$，执行以下步骤：

- 特征分解: 计算模长 $r=(x)_{2}$ 与单位方向 $μ=x/r$。

- 正交噪声生成: 采样标准正态噪声 $n∼N(0,I_{d})$，通过投影算子 $P_{⊥}=I_{d}-μμ^{⊤}$ 映射至切空间并归一化：

$\hat{n_{⊥}}=\frac{P_{⊥}n}{(P_{⊥}n)_{2}}$ (3.2)

- 动态缩放: 计算扰动幅度 $λ= β/ ε$，其中 $β$ 为平滑因子。

- 球面偏转与重投影:

$z=μ+λ⋅\hat{n_{⊥}}, μ^{'}=z/(z)_{2}$ (3.3)

- 范数恢复: 最终输出 $y=r⋅μ^{'}$。

### 3.2.2 算法2：多模态非对称预算分配与掩码机制

在 Qwen-VL 等多模态模型中，视觉与文本嵌入表现出显著的几何差异：

- 视觉嵌入 (Visual Embeddings): 具有高度冗余性。由于自然图像的局部相关性，相邻 Patch 的特征向量高度相似，能够承受较大规模的角度偏转。

- 文本嵌入 (Text Embeddings)： 具有极高的离散性。每个 Token（词元）均携带独立的语义信息，极小的方向偏转就可能导致 Token 从“猫”漂移到语义无关的“电视”。

针对上述异构性，本研究提出一种非对称分配策略。设定系统总隐私预算为 $ϵ_{sys}$，引入非对称因子 $α> 1$：

$ϵ(modality)=(&ϵ_{sys}, &&Visual&α⋅ϵ_{sys}, &&Text &&)$ (3.4)

对应的扰动幅度 $λ$ 为：

$λ(modality)=(&β/ϵ_{sys}, &&Visual&β/(α⋅ϵ_{sys}), &&Text &&)$ (3.5)

该策略优先保障了文本指令的精确性，同时在视觉通道实施更强的保护。

### 3.2.3 算法3：敏感区掩码保护

在 VLM 的多轮对话中，并非所有 Embedding 都包含用户隐私（如系统指令、任务模板）。本研究引入掩码矩阵 $M∈{0,1}^{L}$，实现精准保护：

$y_{final}=M⊙y_{perturbed}+(1-M)⊙x_{original}$ (3.6)

当 $M_{i}=0$ 时（如 "Describe this image:" 等系统提示），保持原值不变，确保模型指令遵循能力不下降。

当 $M_{i}=1$ 时（如用户输入的具体内容），应用 vMF 扰动，实现端到端的隐私屏障。

### 3.2.4 其他命题与引理

命题 3.1（偏转角度）。

$θ=arccos(\frac{1}{\sqrt{1+λ^{2}}})=arctan(λ)=arctan(\frac{β}{ε})$ (3.7)

证明。$\hat{n_{⊥}}⊥μ$，故 $(z)^{2}=1+λ^{2}$，从而 $cosθ=μ^{⊤}z/(z)=1/\sqrt{1+λ^{2}}$。

引理 3.1（切平面投影）。投影算子 $P_{⊥}=I_{d}-μμ^{⊤}$ 将 $R^{d}$ 投影到 $μ^{⊥}={v:v^{⊤}μ=0}$。

证明。(i) 幂等性：$P_{⊥}^{2}=P_{⊥}$（利用 $μ^{⊤}μ=1$）。(ii) 正交性：$μ^{⊤}P_{⊥}v=0$。(iii) 像空间维度为 $d-1$。

引理 3.2（方向均匀性）。若 $n∼N(0,I_{d})$，则 $\hat{n_{⊥}}=P_{⊥}n/‖P_{⊥}n$ 在 $S^{d-2}⊂μ^{⊥}$ 上均匀分布。

证明。高斯分布的旋转不变性保证 $P_{⊥}n$ 在 $μ^{⊥}$上各向同性。选取标准正交基 ${e_{1},…,e_{d-1}}$，投影坐标$n_{i}=e_{i}^{⊤}n~N(0,1)$。由球面对称性，归一化后在 $S^{d-2}$ 上均匀分布。

![图](../asset/docx_media/image5.png)

图3.1：vMF算法示意图

(a) 展示了扰动 Embedding 在 3D 单位超球面上的空间分布，直观体现了不同隐私预算$ϵ$下方向特征的散射程度；

(b) 量化了隐私预算$ϵ$与偏转角$δ$之间遵循反切规律的确定性映射关系，说明了这个机制对于隐私强度的精确控制。

(c) 通过范数分布直方图对比了不同机制的数值稳定性，证明了vMF方法比传统的DP方法更有利于去除范数膨胀现象，保证了Transformer结构输入的稳定性

(d) 从全隐私的角度出发来考察各个机制的语义保持效率（余弦相似度），证明了本文所提出的几何扰动框架比其他的更有利于提高信息效率。

## 3.3 隐私性证明

### 3.3.1 主定理：近似度量隐私

主定理: 设 $d$ 为嵌入维度，$δ_{TV}(d)$ 为切平面分布与匹配 vMF 分布间的全变差距离[16][24]。算法 1 的扰动机制 $M$ 满足 $(κ^{*},δ_{TV})$-近似度量隐私[25-28]：对所有 $x,x^{'}∈R^{d}∖{0}$ 和可测集 $S$：

$Pr(M(x)∈S)≤e^{κ^{*}⋅d_{geo}(μ,μ^{'})}⋅Pr(M(x^{'})∈S)+2δ_{TV}(d)$ (3.8)

其中 $κ^{*}$ 通过数值求解 $A_{d}(κ^{*})=cosδ$ 确定。

其中 $μ=x/(x)，μ^{'}=x^{'}/‖x^{'}‖$，$κ^{*}$ 由 4.3 步骤 B 确定。在 $d = 4096$下 $δ_{TV}$ 可忽略。

### 3.3.2 辅助引理

引理 3.3（vMF 密度比界）。对$μ_{1},μ_{2}∈S^{d-1}$和任意 $z∈S^{d-1}$：

$\frac{f_{vMF}(z;μ_{1},κ)}{f_{vMF}(z;μ_{2},κ)}=exp(κ(μ_{1}-μ_{2})^{⊤}z)≤exp(κ(μ_{1}-μ_{2}))$ (3.9)

证明。归一化常数$C_{d}(κ)$消去后，密度比为$exp(κ(μ_{1}-μ_{2})^{⊤}z)$。由 Cauchy-Schwarz 不等式及 $(z)=1$得上界。

引理 3.4（弦距离-测地线距离）。对$μ_{1},μ_{2}∈S^{d-1}$：

$(μ_{1}-μ_{2})=2sin(\frac{d_{geo}(μ_{1},μ_{2})}{2})≤d_{geo}(μ_{1},μ_{2})$ (3.10)

证明。设$θ=arccos(μ_{1}^{⊤}μ_{2})$，则$(μ_{1}-μ_{2})^{2}=2(1-cosθ)=4sin^{2}(θ/2)$。由 $sin(θ/2)≤θ/2$ 得结论。

引理 3.5（TV 距离的隐私传递）。若机制 $M_{1}$ 满足 $ϵ$-度量隐私，且对所有输入 $x$ 有 $d_{TV}(M_{1}(x),M_{2}(x))≤δ$，则$M_{2}$ 满足 $(ϵ,2δ)$-近似度量隐私。

证明。对任意可测集 $S$ 和输入 $x,x^{'}$：

$Pr(M_{2}(x)∈S)≤Pr(M_{1}(x)∈S)+δ≤e^{ε⋅d(x,x^{'})}Pr(M_{1}(x^{'})∈S)+δ≤e^{ϵ⋅d(x,x^{'})}(Pr(M_{2}(x^{'})∈S)+δ)+δ≤e^{ϵ⋅d(x,x^{'})}Pr(M_{2}(x^{'})∈S)+2δ$ (3.11)

### 3.3.3 主定理证明

算法1分解为方向扰动 $μ↦μ^{'}$ 和模长恢复 $μ^{'}↦rμ^{'}$。模长恢复是确定性后处理，不影响隐私保证。

步骤 A：输出分布刻画。

扰动方向 $μ^{'}=(μ+λ\hat{n_{⊥}})/‖μ+λ\hat{n_{⊥}}‖$，其中 $\hat{n_{⊥}}$ 在 $μ^{⊥}$ 的单位球面上均匀分布（引理 3.2）。偏转角度为确定值 $θ=arctan(λ)$（命题 3.1），故 $μ^{'}$ 均匀分布在球环 $C(μ,θ)$ 上。记该分布为$P_{TP}(∙|μ)$。

步骤 B：构造 vMF 参考机制。

定义参考机制 $M_{vMF}$：从 $vMF(μ,κ^{*})$ 采样，其中 $κ^{*}$ 通过数值求解确定：

$A_{d}(κ^{*})=\frac{I_{d/2}(κ^{*})}{I_{d/2-1}(κ^{*})}=cosθ=\frac{1}{\sqrt{1+β^{2}/ε^{2}}}$ (3.12)

注记 3.1。 $κ^{*}$ 与 $ε$ 的关系是隐式的，需数值求解。例如 $d=4096$，$ε=0.5$，$β=1$ 时 $κ^{*}≈2290$。

由引理 3.3 和 3.4，$M_{vMF}$ 满足严格 $κ^{*}$-度量隐私：

$\frac{f_{vMF}(z;μ_{1},κ^{*})}{f_{vMF}(z;μ_{2},κ^{*})}≤exp(κ^{*}⋅d_{geo}(μ_{1},μ_{2}))$ (3.13)

步骤 C：高维浓度与隐私传递。

切平面分布 $P_{TP}$ 与 vMF 分布 $P_{vMF}$ 的全变差距离为$δ_{TV}=d_{TV}(P_{TP}(∙|μ)),P_{vMF}(∙|μ)$。

由高维浓度现象，vMF 角度方差随维度衰减：$Var_{vMF}(cos∠(μ,μ^{'}))=O(1/d)$。当 $d →∞$ 时 vMF 集中于角度 $θ$ 处，与切平面的 delta 分布趋于一致，$δ_{TV}(d)→0$。

由引理 3.5，切平面机制满足 $(κ^{*},2δ_{TV})$-近似度量隐私。

注记 3.2（实验验证）。 $d = 4096$ 下的数值实验：

![图](../asset/docx_media/image7.png)

图3.2 :vMF分布统计验证体系图

根据图3.2，可以证明我们提出的“切平面近似（Tangent Plane Approximation）”在处理高维 Embedding 隐私时，既能保持真实 vMF 分布的核心特性，又具备更优的数值稳定性，接下来分图进行说明方法的正确性：

(a) Angular Deviation Distribution (角度偏差分布)：用近似算法（TP）和真实的采样（vMF）来比较方向偏移量的微观分布。在高维空间里，切平面近似的偏转角度是极窄的（几乎成脉冲状），但是真实的vMF分布比较宽。尽管分布形状不同，但是它们的平均值是一致的。由此可知近似算法对于期望而言可以很好地满足出隐私预算的要求。

(b) Tangent Plane Isotropy (切平面各向同性)：这个图可以证明出所得到的扰动是在超球面切空间里是不是公平的（也就是不偏袒任何一个方向）。极坐标直方图看起来很像一个完美的圆，各个方向上的采样概率都是一样的。证明了该机制具有各向同性（Isotropy），攻击者不能用统计上的偏差去推测出原始向量中某一个分量的情况，从而保证了扰动的无差别性和安全性。

(c) Dimension Scaling (维度缩放行为)：此图研究维度 $d$ 对近似准确性的影响。在 $d < 128$ 的低维区，近似理论与真实理论不同；但随着 $d$ 增加到 512 以上，两条线完全地趋于一致。证明了基于切平面的几何简化在 VLM 的高维特征空间（通常 $d=4096$）中是数学等价的，但在低维下不适用。

(d) Norm Preservation (范数保持性对比)：即本方法和传统的DP机制相比，在数值稳定方面有较好的表现。小提琴图上，vMF（我们的）是一条在1.0处的直线，Gaussian和Laplace都存在着很大的波动，并且它们的均值向上升起（范数膨胀）。vMF取胜的原因就是它完全抹去了范数漂移，保证了Transformer内部LayerNorm的输入稳定，因此在隐私模式下可以很好地保留语义效果。

(e) Concentration Equivalence (浓度参数等价性)：此图验证隐私预算 $ϵ$ 与语义相似度（余弦相似度）的映射关系是否可靠。切平面理论曲线与真实 vMF 理论点完全一致。证明了我们对 $ϵ$ 的参数化（$λ= β/ϵ$）在数学上成功模拟了 vMF 分布的浓度参数（Concentration Parameter, $κ$）。这使得我们可以直接借用差分隐私的理论框架来量化隐私保护强度。

(f) Projection QQ Plot (投影坐标 QQ 图)：用边缘分布来检验两个算法所得到的噪声是不是有统计上的相同性。采样点大都处在红色的参考线上面。这是分布一致性的有力证明。说明在切平面任何一条投影轴上，近似的算法得到的扰动分量同真实的vMF得到的扰动分量不能区分，从而加强了近似算法的理论可靠性。

## 3.4 效用性分析

### 3.4.1 范数保持

定理 1（严格范数保持）。 $(y)=‖x‖$。

证明。 $(y)=(rμ^{'})=r(μ^{'})=r⋅1=‖x‖$。

推论 1（Attention 信号强度保持）。$(y_{i})⋅(y_{j})=(x_{i})⋅‖x_{j}‖$，Softmax 输入量级不变，避免注意力权重的系统性偏移。

### 3.4.2 余弦相似度

定理 2（期望余弦相似度）。

$E(\frac{x^{⊤}y}{(x)(y)})=cos(arctan\frac{β}{ε})=\frac{1}{\sqrt{1+β^{2}/ε^{2}}}$ (3.14)

证明。 由范数保持性，余弦相似度 $=μ^{⊤}μ^{'}$。偏转角度 $θ=arctan(β/ε)$ 是确定性的，故 $E(μ^{⊤}μ^{'})=cosδ$。

### 3.4.3 点积偏差

定理 3 （点积偏差界）。对独立扰动的 $y_{1}=M(x_{1})，y_{2}=M(x_{2})$：

$(E(y_{1}^{⊤}y_{2})-x_{1}^{⊤}x_{2})≤(x_{1})(x_{2})(1-cos^{2}θ)$ (3.15)

证明。将 $μ_{k}^{'}=cosθ⋅μ_{k}+sinθ⋅\hat{n_{⊥,k}}$ 展开点积期望。由引理 3.2，$E(\hat{n_{⊥,k}})=0$，独立性保证交叉项为零：

$E(μ_{1}^{'}^{⊤}μ_{2}^{'})=cos^{2}θ⋅μ_{1}^{⊤}μ_{2}$ (3.16)

故 $E(y_{1}^{⊤}y_{2})=cos^{2}θ⋅x_{1}^{⊤}x_{2}$，偏差 $=(1-cos^{2}θ)(x_{1}^{⊤}x_{2})≤r_{1}r_{2}sin^{2}δ$。$ϵ$ 较大时 $sin^{2}θ≈β^{2}/ϵ^{2}$，偏差很小，Attention 权重的相对排序在期望下保持。

# 第四章 实验结果

## 4.1 实验环境与参数设置

本文实验在单卡 GPU 环境下完成，硬件平台采用 NVIDIA GeForce RTX 5090显卡进行模型推理、特征扰动和攻击审计实验。实验代码基于Python 3.12 实现，并使用 uv进行依赖管理，主要依赖包括 PyTorch、Transformers/ModelScope、NumPy、scikit-learn 和 Matplotlib。除跨模型泛化实验外，主实验均以 Qwen3-VL-8B-Instruct 作为视觉语言模型主干，文中统称为 Qwen-VL-8B；跨模型验证部分采用 LLaVA-1.5[34][35] 作为补充模型，用于检验该机制是否依赖特定模型结构。

本文不对 VLM 主干模型进行重新训练，而是在端云协同拆分推理场景下，将隐私扰动层插入到 embedding 上传云端之前。具体来说，端侧首先得到视觉 token embedding 和文本 token embedding，二者的隐藏维度均为 $d=4096$；随后分别对两个模态的 token 表示施加 vMF 扰动或对应基线扰动，再将扰动后的序列拼接输入云端语言模型进行后续推理。

隐私预算设置为：$ϵ∈{0.1, 0.2, 0.5, 1.0, 2.0, 5.0}$，其中 $ϵ$ 越小表示隐私保护越强、扰动幅度越大；$ϵ=0.1$ 作为强隐私设置，$ϵ=0.5$ 作为中等隐私设置，$ϵ≥1.0$ 作为弱隐私或高效用设置。vMF 机制中的平滑因子固定为 $β=1.0$，扰动幅度由 $λ=β/ϵ$ 决定。对于图文双通道非对称预算分配，视觉通道使用系统隐私预算 $ϵ_{sys}$，文本通道使用 $αϵ_{sys}$，其中 $α>1$ 表示文本侧采用更弱扰动，以降低对系统提示和任务指令的破坏。

对比方法包括无隐私保护的 No-Privacy Baseline、Pixel-Gaussian、Embedding-Laplace 以及本文提出的 vMF-Ours。Pixel-Gaussian 和 Embedding-Laplace 分别对应差分隐私中常见的高斯机制与拉普拉斯机制[25,26,39,40]，Embedding-Laplace 同时参考了文本和 embedding 表征空间中的扰动式隐私保护研究[8,20,21]；vMF-Ours 则基于方向统计、度量隐私和流形差分隐私相关理论[16,17,24,27,28]。由于 VLM embedding 属于典型高维连续表征，本文也参考高维差分隐私研究中关于维度增长、降维与随机投影的讨论[41]。对于 Gaussian 机制，实验中采用 $δ=10^{-5}$；对于 Laplace 机制，噪声尺度随 $ϵ$ 调整；对于 vMF 机制，扰动仅改变 embedding 方向并恢复原始范数。所有实验在相同数据划分、相同隐私预算和相同攻击审计设置下比较，以保证不同机制之间的结果具有可比性。

为降低随机采样带来的偶然性，机制可视化与数值模拟实验固定随机种子，并对同一隐私预算下的多次采样结果统计均值和标准差。结果图中的阴影区间或误差范围表示不同样本上的波动情况。

## 4.2 实验数据集说明

本研究选取 MS-COCO[29] 与 VQA-v2[30] 两个视觉语言任务作为主要评估场景，分别覆盖开放式图像描述生成和视觉问答两类典型 VLM 应用。两者都需要模型同时理解图像内容和文本指令，因此能够较全面地反映中间 embedding 扰动对模型效用和隐私风险的影响。

MS-COCO (2017): 本文使用 MS-COCO 2017 validation 集中的图像描述任务进行评估。该验证集包含5000 张图像，每张图像通常对应 5 条人工参考描述，约形成 2.5 万条参考 caption。实验输入为图像及对应的描述提示，模型需要生成自然语言 caption。该任务关注模型能否在扰动后保留图像中的主体、属性、动作和场景关系，适合检验长文本生成的语义连贯性。本文在该数据集上主要使用 CIDEr 衡量生成 caption 与人工参考描述之间的 n-gram 共识程度，使用 CLIPScore 衡量生成文本与图像之间的跨模态语义一致性；同时使用 SSIM 和 ASR 衡量攻击者从中间 embedding 中恢复视觉内容的风险。

VQA-v2: 本文使用 VQA-v2 validation 集作为视觉问答评估来源。该验证集包含 40504 张图像 和 214354 个问答对，问题类型覆盖 Yes/No、Number 和 Other 三类。实验输入为图像和自然语言问题，模型需要生成简短答案。该任务更强调细粒度视觉理解能力，例如物体颜色、数量、空间关系和属性识别。本文在该数据集上主要使用 Accuracy 衡量模型答案是否与标准答案匹配，使用 F1 衡量预测答案与参考答案在关键词层面的重叠程度；同时使用 SSIM和ASR 评估扰动后中间表示的反演风险。

## 4.3 攻击模型：特征反演审计

为了客观评估端云拆分推理中的隐私泄露风险，本研究假设云端攻击者为“诚实但好奇”的灰盒攻击者：其能够截获上传到云端的扰动后 token embedding，并了解模型结构和切分位置，但不能直接访问用户原始图像和文本。针对图文双通道表示，本研究构建三类反演审计模型：

视觉反演攻击：攻击者以扰动后的视觉 token embedding，或多模态序列中的视觉 token 子序列为输入，采用 FIA-Flow 风格的黑盒特征反演模型[36]，训练逆向网络 $G_{v}$ 将中间特征映射到可重构的视觉潜空间，并尝试恢复原始图像：

$\hat{I}=G_{v}(\hat{X^{I}})$ (4.1)

若 $\hat{I}$ 能够还原原图的轮廓、主体物体或关键视觉语义，则认为视觉隐私发生泄露。

文本反演攻击：攻击者以扰动后的文本 token embedding 为输入，参考 ALGEN 等 embedding-to-text 反演方法[37]，训练文本解码器 $G_{t}$ 恢复用户输入中的敏感词、实体或主要语义：

$\hat{T}=G_{t}(\hat{X^{T}})$ (4.2)

若恢复文本 $\hat{T}$ 中包含原始用户输入 $T$ 的敏感实体、关键词或主要语义，则认为文本隐私发生泄露。

多模态联合反演攻击：攻击者以扰动后的视觉 token 和文本 token 拼接序列为输入，参考 VLM 中的自适应 token 加权反演攻击[38]，训练联合攻击模型 $G_{m}$ 从跨模态表示中恢复任一模态的敏感信息：

$(\hat{I},\hat{T})=G_{m}((\hat{X^{I}};\hat{X^{T}}))$ (4.3)

若攻击结果能够恢复图像敏感内容或文本敏感内容中的任一类，则认为多模态攻击成功。

上述三类攻击分别对应视觉 ASR、文本 ASR 和多模态 ASR。本文主实验采用多模态联合反演审计作为默认攻击模型，消融实验进一步比较 Visual-Only、Text-Only 与 Both-vMF 设置下的防御效果。

## 4.4 详细评估指标定义与数学表达

### 4.4.1 CIDEr

CIDEr[31]用于评估图像描述任务中生成 caption 的质量。对于第 $i$ 个样本，设模型生成的描述为 $c_{i}$，人工参考描述集合为 $S_{i}={s_{i1},…,s_{im}}$。CIDEr 先统计生成句子和参考句子中的 n-gram，再用 TF-IDF 权重降低常见词的影响、突出图像相关关键词，最后计算二者的加权余弦相似度：

$CIDEr_{n}(c_{i},S_{i})=\frac{1}{m}∑_{j=1}^{m}\frac{g^{n}(c_{i})⋅g^{n}(s_{ij})}{(g^{n}(c_{i}))(g^{n}(s_{ij}))}$ (4.4)

其中 $g^{n}(c_{i})$ 是生成描述中 n-gram 的 TF-IDF 向量。实际计算时通常综合 1-gram 到 4-gram 的结果，并对测试集所有样本取平均。CIDEr 数值越高，说明生成描述与人工标注在主体、属性、动作和场景关系上的一致性越强。本文将其作为 MS-COCO 图像描述任务的主要效用指标，用来衡量隐私扰动后模型是否仍能生成准确的自然语言描述。

### 4.4.2 CLIPScore

CLIPScore[32]是一种无参考的跨模态语义一致性指标，不依赖人工 caption，而是直接比较原始图像 $I$ 与模型生成文本 $C$ 是否语义匹配。具体做法是利用预训练的 CLIP 视觉编码器 $E_{I}$ 和文本编码器 $E_{T}$ 将图像和文本映射到共同特征空间，并计算二者的余弦相似度：

$CLIPScore(I,C)=w⋅max(cos(E_{I}(I),E_{T}(C)),0)$ (4.5)

其中 $w$ 为缩放系数（通常取 2.5），$max(⋅,0)$ 用于截断负相似度。CLIPScore 越高，说明生成文本越符合图像整体语义，例如是否描述了正确的物体类别、场景和关键属性。与 CIDEr 相比，CLIPScore 更关注图文整体匹配，而不是与参考句子的词面重合。因此，本文将其作为 MS-COCO 上的辅助效用指标，用于补充验证扰动后跨模态语义是否保持一致。

### 4.4.3 SSIM

SSIM[33]用于衡量特征反演攻击还原出的图像 $x$ 与原始图像 $y$ 的结构相似度。与像素级误差不同，SSIM 同时比较两张图像的亮度 ($l$)、对比度 ($c$) 和结构 ($s$)，因此更接近人眼对图像重构质量的感知：

$SSIM(x,y)=(l(x,y))^{α}⋅(c(x,y))^{β}⋅(s(x,y))^{γ}$ (4.6)

具体形式为：

$SSIM(x,y)=\frac{(2μ_{x}μ_{y}+C_{1})(2σ_{xy}+C_{2})}{(μ_{x}^{2}+μ_{y}^{2}+C_{1})(σ_{x}^{2}+σ_{y}^{2}+C_{2})}$ (4.7)

其中 $μ$ 为均值，$σ$ 为方差，$σ_{xy}$ 为协方差，$C_{1}$ 和 $C_{2}$ 为稳定项。SSIM 通常位于 $(0,1)$ 区间，越接近 1 表示重构图像越接近原图。需要注意的是，在本文中 SSIM 不是任务效用指标，而是隐私风险指标：攻击者重构图像与原图越相似，说明泄露风险越高。因此，在隐私审计中，SSIM 数值越低代表防御越成功。

### 4.4.4 余弦相似度

余弦相似度用于衡量扰动前后 embedding 向量在方向上的一致性。设原始 embedding 为 $x$，扰动后 embedding 为 $y$，则二者夹角的余弦为：

$cos(θ)=\frac{x⋅y}{(x)(y)}$ (4.8)

角度偏差定义为 $Δθ=arccos(cos(θ))⋅\frac{180}{π}$。

在基于超球面的度量隐私框架中，embedding 的语义主要由方向决定，因此余弦相似度可以直接反映扰动是否偏离原始语义方向。该指标取值越高，说明扰动后表示与原始表示越接近；角度偏差越大，说明隐私扰动越强、语义偏移越明显。本文主要将余弦相似度用于机制可视化和几何诊断，辅助说明 vMF 扰动在相同隐私预算下比高斯和拉普拉斯噪声更容易保持方向结构；正式下游效用仍以 CIDEr、CLIPScore、Accuracy 和 F1 为主。

### 4.4.5 VQA 准确率与 F1 分数

在 VQA-v2 任务中，本文使用准确率和 F1 分数衡量模型回答质量。计算前先对模型预测答案 $\hat{a_{i}}$ 和标准答案 $a_{i}$ 做统一归一化，包括转小写、去除标点、去除多余空格，并将数字词映射为阿拉伯数字。归一化后，若预测答案与标准答案完全一致，则认为该样本回答正确：

$Correct(i)=1(norm(\hat{a_{i}})=norm(a_{i}))$ (4.9)

准确率定义为回答正确样本所占比例：

$Accuracy=\frac{1}{N}∑_{i=1}^{NCorrect}(i)$ (4.10)

对于存在多个人工标注答案的 VQA-v2 样本，本文取预测答案与所有参考答案中的最大匹配得分；若实验数据只保留单一标准答案，则退化为上述 exact match 判定。

F1 分数用于衡量预测答案与参考答案在词级别上的重叠程度。设归一化后的预测答案 token 集合为 $\hat{A_{i}}$，参考答案 token 集合为 $A_{i}$，则：

$P_{i}=\frac{(\hat{A_{i}}∩A_{i})}{(\hat{A_{i}})}, R_{i}=\frac{(\hat{A_{i}}∩A_{i})}{(A_{i})}$ (4.11)

$F1_{i}=\frac{2P_{i}R_{i}}{P_{i}+R_{i}}$ (4.12)

最终 F1 为所有样本的平均值：

$F1=\frac{1}{N}∑_{i=1}^{N}F1_{i}$ (4.13)

因此，准确率强调答案是否完全匹配，F1 则允许答案在关键词或短语层面部分匹配，更适合衡量开放式 VQA 回答的语义完整性。

### 4.4.6 攻击成功率

攻击成功率用于衡量攻击者从扰动后的中间表示中恢复隐私信息的比例。对于测试集中的 $N$ 个样本，统一定义为：

$ASR=\frac{1}{N}∑_{i=1}^{N}1(AttackSuccess(i))$ (4.14)

其中 $1(⋅)$ 为指示函数，攻击成功记为 1，否则记为 0。ASR 越低，说明隐私保护效果越好。

针对图文双通道嵌入，本研究进一步区分视觉、文本和多模态三类 ASR：

1)视觉攻击成功率 $ASR_{v}$：攻击者以扰动后的视觉 token embedding 为输入，利用 FIA-Flow 风格的特征反演模型重构图像。若重构图像 $\hat{I}$ 与原图 $I$ 的结构或语义相似度超过阈值，则认为视觉攻击成功：

$Success_{v}(i)=1(SSIM(I_{i},\hat{I_{i}})≥τ_{v})$ (4.15)

或在语义级审计中使用 CLIP 相似度作为判定依据：

$Success_{v}(i)=1(CLIPSim(I_{i},\hat{I_{i}})≥τ_{v})$ (4.16)

2)文本攻击成功率 $ASR_{t}$：攻击者以扰动后的文本 token embedding 为输入，利用 embedding-to-text 反演模型恢复用户文本。若恢复文本 $\hat{T}$ 中包含原始用户输入 $T$ 的敏感实体、关键词或主要语义，则认为文本攻击成功：

$Success_{t}(i)=1(SensitiveRecall(T_{i},\hat{T_{i}})≥τ_{t})$ (4.17)

3)多模态攻击成功率 $ASR_{m}$：攻击者以扰动后的视觉 token 与文本 token 拼接序列为输入，尝试恢复任一模态的隐私信息。若视觉侧或文本侧任一攻击成功，则认为多模态攻击成功：

$Success_{m}(i)=1(Success_{v}(i)∨Success_{t}(i))$ (4.18)

$ASR_{m}=\frac{1}{N}∑_{i=1}^{N}1(Success_{v}(i)∨Success_{t}(i))$ (4.19)

在本文主实验图中，ASR 表示联合反演审计下的多模态攻击成功率；在消融实验中，则分别统计 Visual-Only、Text-Only 与 Both-vMF 设置下的攻击成功率，用于验证双通道保护的必要性。

## 4.5 对比基准

本文选择以下方法作为对比基准，用于验证 vMF 几何扰动相对于传统加性噪声方法的优势：

No-Privacy Baseline：不对中间 embedding 施加任何扰动，直接将原始视觉和文本 token embedding 输入云端模型。该设置用于给出任务效用上限，同时也代表最高的特征泄露风险；其模型配置与主干 Qwen-VL/Qwen3-VL 保持一致[1,23]。

Pixel-Gaussian：在输入图像或视觉侧表示上加入高斯噪声，用于模拟传统像素级扰动防御。高斯噪声是差分隐私中经典的连续噪声机制，通常通过隐私预算和灵敏度校准噪声方差[25,26]。该方法实现简单，但噪声并不直接适配 VLM 的高维 embedding 几何结构，容易破坏图像细节和下游语义理解。

Embedding-Laplace：在 embedding 空间中加入拉普拉斯噪声，是差分隐私中常用的加性噪声机制[25,26]。相关文本隐私研究也常在词向量或句向量空间中进行校准扰动，以降低原始文本或语义表征泄露风险[8,20,21]。该方法能够在特征层提供一定隐私保护，但会同时改变向量方向和范数，可能导致 Transformer 内部注意力分布发生偏移。

vMF-Ours：本文提出的基于 von Mises-Fisher 分布的超球面方向扰动机制。vMF 分布和方向统计为单位球面上的方向采样提供了理论基础[16,24]，度量隐私和流形差分隐私进一步支持在非欧氏空间中定义距离相关的隐私保护强度[17,27,28]。该方法将 embedding 分解为模长和方向，仅对方向进行受控偏转，并在输出阶段恢复原始范数；同时结合图文双通道非对称预算和文本敏感区掩码，实现更稳定的隐私-效用折中。

## 4.6 结果分析

本研究在多个主流 VLM 任务上对 vMF 机制进行了全面评估，以下是详细的实验结果分析。

### 4.6.1 主实验效用分析

参考图4.1和图4.2在下面几个方面进行分析。

![图](../asset/docx_media/image9.png)

图4.1 :MS-Coco数据集主实验图

![图](../asset/docx_media/image9.png)

图4.1 :MS-Coco数据集主实验图

MS-COCO 使用 CIDEr 与 CLIPScore 衡量图像描述效用，二者越高说明生成文本越接近人工参考描述、越符合图像语义。VQA-v2 使用 Accuracy 与 F1 衡量问答效用，其中 Accuracy 强调答案是否完全匹配，F1 强调答案关键词和短语的部分匹配能力。

在 MS-COCO 任务中，当隐私预算 $ϵ=0.1$（强隐私约束）时，vMF 机制的 CIDEr 得分保持在 84.9，显著高于 Embedding-Laplace (75.1) 和 Pixel-Gaussian (58.4)。这表明 vMF 能够在强扰动下依然保留图像主体、属性和场景关系，使生成 caption 不至于完全偏离原图。

在 VQA-v2 任务中，vMF 的 Accuracy 和 F1 在各隐私预算下均高于或接近其他机制，说明其不仅能保持答案的完全匹配率，也能保留回答中的关键语义成分。相比之下，高斯和拉普拉斯加噪更容易破坏细粒度属性判断，例如颜色、数量和空间关系。

随着 $ϵ$ 增加至 5.0，扰动强度减小，所有机制的效用指标均逐渐接近 No-Privacy Baseline。vMF 机制在全隐私预算区间内始终保持更高效用，说明基于方向的扰动比坐标级加性噪声更适合高维多模态 embedding。

### 4.6.2 隐私-效用权衡分析

参考图4.3和图4.4。

在 Trade-off 坐标系中，vMF 曲线始终位于左上方（或最靠近坐标原点/理想区域）。在相同的攻击成功率（ASR）水平下，vMF 能够提供比其他机制高出 10%-15% 的相对效用增益。

在相同的攻击成功率（ASR）水平下，vMF 能够提供比其他机制高出 10%-15%的相对效用增益；在相同 CIDEr、Accuracy 或 F1 水平下，vMF 对应的 ASR 与 SSIM 更低，说明攻击者更难从扰动后的 embedding 中恢复图像结构或敏感语义。

观察 ASR 曲线可以发现，vMF 在中等隐私区（$ϵ=0.5$）的斜率更平缓，这意味着它能够提供更宽的“安全操作窗口”，不会因为 $ϵ$ 的微小波动导致隐私保护迅速失效。

![图](../asset/docx_media/image11.png)

图4.3 :MS-Coco数据集trade-off实验图

![图](../asset/docx_media/image13.png)

图4.4 :VQA-v2数据集trade-off实验图

### 4.6.3 稳定性验证

参考图4.5。

![图](../asset/docx_media/image15.png)

图4.5 :范数保持对比图

该图统计扰动后 embedding 范数均值。理想情况下，范数应保持在 1.0 附近；偏离越大，说明扰动越可能改变 Transformer 内部激活尺度，进而影响 LayerNorm 和注意力计算。

条形图清晰地展示了物理层面的差异。传统 Gaussian 和 Laplace 机制在加噪后，Embedding 范数均值出现了 10%-30% 的向上漂移（Norm Expansion），这种漂移直接破坏了 Transformer 内部 LayerNorm 层的输入分布假设。

相比之下，vMF 机制在所有 $ϵ$ 设置下均保持了 1.0 的恒定范数。这种数值稳定性是模型在隐私模式下依然能维持逻辑连贯性的根本物理保证。

### 4.6.4 全模态保护的必要性分析

![图](../asset/docx_media/image17.png)

图4.6：模态保护消融图

参考图4.6。

消融实验同时观察 CIDEr 和 ASR。CIDEr 越高表示图像描述效用越好，ASR 越低表示反演攻击越难成功。因此，理想方案应同时保持较高 CIDEr 和较低 ASR。

消融实验显示，若仅采用视觉扰动（Visual-Only），虽然 CIDEr 得分略高，但其 ASR（攻击成功率）高达 0.312（$ϵ=0.1$）。这是由于攻击者可以利用未受保护的文本 Embedding 之间的跨模态关联进行特征还原。

闭环防御: 只有在视觉与文本同时应用 vMF 扰动（Both-vMF）时，才能将 ASR 压制在 0.182 的极低水平，证明了全模态协同防御在 VLM 隐私保护中的不可替代性。

### 4.6.5 跨模型泛化性评估

跨模型实验同时报告 Accuracy 和 ASR，前者衡量迁移到 LLaVA-1.5[34][35] 后的问答效用，后者衡量特征反演风险。若 vMF 在新模型上仍能保持较高 Accuracy 和较低 ASR，则说明该机制不是只针对 Qwen-VL 的特定参数有效。

本研究在 LLaVA-1.5模型上复现了实验。结果显示，vMF 机制在 LLaVA 上的效用保持能力与隐私防御能力与其在 Qwen3-VL-8B-Instruct 上的表现高度一致，表明该方法的有效性与模型架构无关。

这证实了 vMF 几何隐私框架并不依赖于特定模型的参数权重，而是针对 Transformer 架构通用的 Embedding 几何约束设计的，具有极强的跨模型部署潜力。

![图](../asset/docx_media/image19.png)

图4.7：跨模态试验效果图

# 第五章 结论与展望

针对多模态大模型在端云协同拆分推理场景下的数据泄露风险，本文提出并实现了一种面向 Embedding 层的语义保持度量隐私保护机制。通过以Qwen3-VL-8B-Instruct模型为核心的研究对象，系统探讨了高维特征几何特性对隐私与效用的平衡影响。主要工作总结如下：

- 提出了基于 vMF 分布的几何扰动框架：利用黎曼流形理论，将 Embedding 空间建模为单位超球面，通过正交切平面投影实现了范数守恒的方向扰动，从底层解决了传统 DP 机制导致的“注意力崩塌”问题。

- 设计了多模态非对称隐私预算分配策略：针对图文模态对噪声敏感度异构的特性，引入了非对称因子 $α$ 与敏感区掩码矩阵，成功解耦了系统指令骨架与用户隐私语义，保障了模型的指令遵循能力。

- 构建了全方位的实验验证体系：在 MS-COCO 和 VQA-v2 数据集上的实验表明，在强隐私约束下，本机制将特征反演攻击成功率压制在可控水平以下，同时保留了较高的模型效用，其帕累托性能显著优于传统的高斯与拉普拉斯机制。

本文的研究亮点在于打破了隐私保护与模型效用之间的“零和博弈”。传统的差分隐私研究往往将 Embedding 视为普通的欧氏空间向量，而本文深入挖掘了 “方向承载语义，范数承载激活” 的物理本质。通过保持范数恒定，本研究在物理层确保了 Transformer 结构的平稳运行，为高维表征数据的隐私保护提供了一个全新的几何视角。

尽管本文取得了一定的研究成果，但仍存在以下改进空间：

- 动态隐私预算分配：目前采用统一扰动强度，未来可探索引入视觉显著性检测，针对图像中的 ROI（感兴趣区域）实施差异化的隐私预算分配。

- 防御自适应几何攻击：面对专门针对超球面分布设计的自适应对抗攻击，本机制的防御鲁棒性仍需在未来工作中进一步压力测试。

- 隐私增强型微调：未来可考虑结合 LoRA 等参数高效微调技术，使模型在训练阶段就适应几何扰动分布，从而在极低隐私预算下进一步挖掘效用上限。

# 参 考 文 献

Bai J, Bai S, Yang S, et al. Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond[J/OL]. arXiv:2308.12966, 2023. https://arxiv.org/abs/2308. 12966

Radford A, Kim J W, Hallacy C, et al. Learning Transferable Visual Models from Natural Language Supervision[C]//Proceedings of the 38th International Conference on Machine Learning. PMLR, 2021: 8748-8763. https://arxiv.org/abs/2103.00020

Vepakomma P, Gupta O, Swedish T, et al. Split Learning for Health: Distributed Deep Learning without Sharing Raw Patient Data[J/OL]. arXiv:1812.00564, 2018. https://arxiv.org/abs/1812.00564

Boenisch F, Dziedzic A, Schuster R, et al. When the Curious Abandon Honesty: Federated Learning Is Not Private[C]//2023 IEEE 8th European Symposium on Security and Privacy. IEEE, 2023: 175-199. https://doi.org/10.1109/EuroSP57164.2023.00020

He Z, Zhang T, Lee R B. Model Inversion Attacks Against Collaborative Inference[C]//Proceedings of the 35th Annual Computer Security Applications Conference. ACM, 2019: 148-162. https://doi.org /10.1145/3359789.3359824

Dong T, Meng Y, Li S, et al. Depth Gives a False Sense of Privacy: LLM Internal States Inversion[C]//34th USENIX Security Symposium. USENIX Association, 2025: 1629-1648. https://www.usenix.org/conference/usenixsecurity25/presentation/dong-tian

Shu, et al. SMI-AW: Sequence-Based Model Inversion with Adaptive Weighting[J/OL]. arXiv preprint, 2025.

Qu C, Kong W, Yang L, et al. Natural Language Understanding with Privacy-Preserving BERT[C]//Proceedings of the 30th ACM International Conference on Information & Knowledge Management. ACM, 2021: 1488-1497. https://doi.org/10.1145/3459637.3482281

Jin S, Pang X, Wang Z, et al. Safeguarding LLM Embeddings in End-Cloud Collaboration via Entropy-Driven Perturbation[J/OL]. arXiv:2503.12896, 2025. https://arxiv.org/abs/2503.12896

Liu T, Yao H, Wu T, et al. Mitigating Privacy Risks in LLM Embeddings from Embedding Inversion[J/OL]. arXiv:2411.05034, 2024. https://arxiv.org/abs/2411.05034

Hemo O B, Zolfi A, Yehezkel O, et al. Gradient Inversion of Multimodal Models[C]//Proceedings of the 42nd International Conference on Machine Learning. PMLR, 2025, 267: 22988-23004. https: //proceedings.mlr.press/v267/hemo25a.html

Zhang T, Jha R, Bagdasaryan E, et al. Adversarial Illusions in Multi-Modal Embeddings[C]//33rd USENIX Security Symposium. USENIX Association, 2024: 3009-3025. https://www.usenix.org /conference/usenixsecurity24/presentation/zhang-tingwei

Xiu K, Zhang S. CapRecover: A Cross-Modality Feature Inversion Attack Framework on Vision Language Models[J/OL]. arXiv:2507.22828, 2025. https://arxiv.org/abs/2507.22828

Mattern J, Weggenmann B, Kerschbaum F. The Limits of Word Level Differential Privacy[C]//Findings of the Association for Computational Linguistics: NAACL 2022. ACL, 2022: 867-881. https:// aclanthology.org/2022.findings-naacl.65/

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

Dwork C, McSherry F, Nissim K, et al. Calibrating Noise to Sensitivity in Private Data Analysis[C] //Theory of Cryptography Conference. Springer, 2006: 265-284. https://doi.org/10.1007/ 11681878_14

Dwork C, Roth A. The Algorithmic Foundations of Differential Privacy[J]. Foundations and Trends in Theoretical Computer Science, 2014, 9(3-4): 211-407. https://doi.org/10.1561/0400000042

Reimherr M, Bharath K, Soto C. Differential Privacy over Riemannian Manifolds[C]//Advances in Neural Information Processing Systems. 2021, 34. https://proceedings.neurips.cc/paper/2021/hash/ 6600e06fe9350b62c1e343504d4a7b86-Abstract.html

Jiang Y, Chang X, Liu Y, et al. Gaussian Differential Privacy on Riemannian Manifolds[C]//Advances in Neural Information Processing Systems. 2023, 36. https://proceedings.neurips.cc/paper_files/ paper/2023/hash/2f27964513a28d034530bfdd117ea31d-Abstract-Conference.html

Lin T Y, Maire M, Belongie S, et al. Microsoft COCO: Common Objects in Context[C]//European Conference on Computer Vision. Springer, 2014: 740-755. https://arxiv.org/abs/1405.0312

Goyal Y, Khot T, Summers-Stay D, et al. Making the V in VQA Matter: Elevating the Role of Image Understanding in Visual Question Answering[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2017: 6904-6913. https://arxiv.org/abs/1612.00837

Vedantam R, Lawrence Zitnick C, Parikh D. CIDEr: Consensus-Based Image Description Evaluation[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. IEEE, 2015: 4566-4575. https://openaccess.thecvf.com/content_cvpr_2015/html/Vedantam_CIDEr_ Consensus-Based_Image_2015_CVPR_paper.html

Hessel J, Holtzman A, Forbes M, et al. CLIPScore: A Reference-Free Evaluation Metric for Image Captioning[C]//Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. ACL, 2021: 7514-7528. https://aclanthology.org/2021.emnlp-main.595/

Wang Z, Bovik A C, Sheikh H R, et al. Image Quality Assessment: From Error Visibility to Structural Similarity[J]. IEEE Transactions on Image Processing, 2004, 13(4): 600-612. https://doi.org /10.1109/TIP.2003.819861

Liu H, Li C, Wu Q, et al. Visual Instruction Tuning[C]//Advances in Neural Information Processing Systems. 2023, 36. https://arxiv.org/abs/2304.08485

Liu H, Li C, Li Y, et al. Improved Baselines with Visual Instruction Tuning[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2024: 26296-26306. https://arxiv.org/abs/2310.03744

Ren J, He Z, Lee R B. What Your Features Reveal: Data-Efficient Black-Box Feature Inversion Attack for Split DNNs[J/OL]. arXiv:2511.15316, 2025. https://arxiv.org/abs/2511.15316

Chen Y, Xu Q, Bjerva J. ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation[C/OL]//Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics. ACL, 2025. https://arxiv.org/abs/2502.11308

Nguyen N B, Ho S T, Koh J H, et al. Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks[C/OL]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. IEEE, 2026. https://arxiv.org/abs/2508.04097

高志强, 王宇涛. 差分隐私技术研究进展[J]. 通信学报, 2017, 38(Z1): 151-155. https://doi.org/10.11959/j.issn.1000-436x.2017248

谭作文, 张连福. 机器学习隐私保护研究综述[J]. 软件学报, 2020, 31(7): 2127-2156. https://doi.org/10.13328/j.cnki.jos.005815

张兴, 陈昊. 差分隐私的高维数据发布研究综述[J]. 智能系统学报, 2021, 16(6): 989-998. https://doi.org/10.11992/tis.202012013

陈晋音, 席昌坤, 郑海斌, 高铭, 张甜馨. 多模态大语言模型的安全性研究综述[J]. 计算机科学, 2025, 52(7): 315-341. https://www.jsjkx.com/CN/Y2025/V52/I7/315

# 致 谢

行文至此，本科四年时光转瞬即逝，终于在此刻完成了我的毕业论文，在此刻之前，我曾无数次憧憬毕业时候的潇洒自如，但实际是内心平静，又在为明天的事情忙碌，焦虑，奋斗。

首先感谢我自己，从大一什么都不懂的腼腆小男孩成长成现在能够独当一面的小大人，我时常佩服自己的果断，什么事情都是先做了在说，也时常佩服自己的勇气，无论是好是坏都尝试把它走对，更佩服自己的心态，想着无论什么事总有一个兜底的选项。这些性格造就了我今天的成就，所以我非常感谢那个从不放弃，敢闯敢拼的自己。

感谢我的家人，没有他们无条件的支持，理解与包容，我无法无忧无虑的在前行路上奔跑。他们的爱是我最坚实的后盾，我也时常跟自己说，累了就回家吧。

感谢一路上帮助过我的老师们，尤其是张少波老师，王晓亮老师和刘天晶老师，感谢张老师在本科阶段的教学以及学术，学业论文上的指导，感谢王晓亮老师在本科各项竞赛中给予的帮助与支持，感谢刘天晶老师在学生工作上的对我的帮助。当然，不止这三位老师，所有老师们的言传身教，都将令我终身受益。

感谢我的室友，同学以及朋友们，很庆幸能够和你们度过最美好的四年，感谢彼此之间的支持与陪伴，我将永生难忘。

最后我想引用这段来敦促今后的我：“We have only two jobs on this world, The first, to learn, The second, to cope”，希望未来的我，始终不忘学习，也始终拥有面对一切的能力。
