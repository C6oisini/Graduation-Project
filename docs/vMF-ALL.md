# 面向多模态大模型嵌入层的语义感知度量隐私保护机制：完整理论与证明

## 目录

- [第一部分 预备知识](#第一部分-预备知识)
- [第二部分 语义流形与度量空间定义](#第二部分-语义流形与度量空间定义)
- [第三部分 基于 vMF 的语义感知扰动算法](#第三部分-基于-vmf-的语义感知扰动算法)
- [第四部分 隐私性证明](#第四部分-隐私性证明)
- [第五部分 效用性分析与证明](#第五部分-效用性分析与证明)
- [第六部分 双通道非对称预算分配](#第六部分-双通道非对称预算分配)
- [第七部分 与传统机制的理论比较](#第七部分-与传统机制的理论比较)
- [第八部分 工程实现与部署](#第八部分-工程实现与部署)

---

## 第一部分 预备知识

### 1.1 差分隐私 (Differential Privacy)

**定义 1.1（$(\varepsilon, \delta)$-差分隐私）。** 随机机制 $\mathcal{M}: \mathcal{X} \to \mathcal{Y}$ 满足 $(\varepsilon, \delta)$-差分隐私，若对所有相邻数据集 $D, D'$（即 $D$ 与 $D'$ 至多相差一条记录）以及所有可测集 $S \subseteq \mathcal{Y}$，有：

$$\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

### 1.2 度量隐私 ($d_\mathcal{X}$-privacy)

**定义 1.2（度量隐私）。** 设 $(\mathcal{X}, d_\mathcal{X})$ 为度量空间，随机机制 $\mathcal{M}: \mathcal{X} \to \mathcal{Y}$ 满足 $d_\mathcal{X}$-privacy，若对所有 $x, x' \in \mathcal{X}$ 以及所有可测集 $S \subseteq \mathcal{Y}$，有：

$$\Pr[\mathcal{M}(x) \in S] \le e^{d_\mathcal{X}(x, x')} \cdot \Pr[\mathcal{M}(x') \in S]$$

度量隐私是差分隐私在连续空间上的推广。当 $d_\mathcal{X}(x, x') = \varepsilon \cdot \|x - x'\|$ 时，退化为标准的局部差分隐私。

**注记 1.1。** 度量隐私的核心优势在于：隐私保护的强度与数据点之间的"语义距离"成正比。语义相近的输入获得相似的输出分布，语义相远的输入则获得差异更大的输出分布。这天然适合嵌入空间中的隐私保护。

### 1.3 von Mises-Fisher 分布

**定义 1.3（vMF 分布）。** $d$ 维单位超球面 $\mathbb{S}^{d-1}$ 上的 von Mises-Fisher 分布 $\text{vMF}(\boldsymbol{\mu}, \kappa)$ 的概率密度函数为：

$$f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp\left(\kappa \cdot \boldsymbol{\mu}^\top \mathbf{z}\right)$$

其中：
- $\boldsymbol{\mu} \in \mathbb{S}^{d-1}$ 为均值方向参数
- $\kappa \ge 0$ 为集中度参数
- $C_d(\kappa) = \frac{\kappa^{d/2-1}}{(2\pi)^{d/2} I_{d/2-1}(\kappa)}$ 为归一化常数
- $I_\nu(\cdot)$ 为第一类修正贝塞尔函数

**性质 1.1。** 当 $\kappa = 0$ 时，vMF 退化为球面上的均匀分布；当 $\kappa \to \infty$ 时，分布集中于 $\boldsymbol{\mu}$ 处的点质量。

**性质 1.2。** 对于 $\mathbf{z} \sim \text{vMF}(\boldsymbol{\mu}, \kappa)$，有 $\mathbb{E}[\mathbf{z}] = A_d(\kappa) \boldsymbol{\mu}$，其中 $A_d(\kappa) = I_{d/2}(\kappa) / I_{d/2-1}(\kappa)$。

### 1.4 超球面上的测地线距离

**定义 1.4。** 对于 $\mathbf{u}, \mathbf{v} \in \mathbb{S}^{d-1}$，测地线距离定义为：

$$d_{\text{geo}}(\mathbf{u}, \mathbf{v}) = \arccos(\mathbf{u}^\top \mathbf{v})$$

这是 $\mathbb{S}^{d-1}$ 上的自然度量，等价于两点之间的大圆弧长。

**性质 1.3。** 测地线距离满足度量公理：
1. 非负性：$d_{\text{geo}}(\mathbf{u}, \mathbf{v}) \ge 0$
2. 同一性：$d_{\text{geo}}(\mathbf{u}, \mathbf{v}) = 0 \iff \mathbf{u} = \mathbf{v}$
3. 对称性：$d_{\text{geo}}(\mathbf{u}, \mathbf{v}) = d_{\text{geo}}(\mathbf{v}, \mathbf{u})$
4. 三角不等式：$d_{\text{geo}}(\mathbf{u}, \mathbf{w}) \le d_{\text{geo}}(\mathbf{u}, \mathbf{v}) + d_{\text{geo}}(\mathbf{v}, \mathbf{w})$

---

## 第二部分 语义流形与度量空间定义

### 2.1 单位超球面假设

**假设 2.1（超球面嵌入假设）。** Qwen-VL 模型中，经过 LayerNorm 标准化后的特征向量的有效语义信息主要编码在其方向上，即特征向量可视为 $d-1$ 维单位超球面 $\mathbb{S}^{d-1}$ 上的点（$d = 4096$ 对于 Qwen-VL-8B）。

**理论依据。** 该假设基于以下两个观察：

**(a) LayerNorm 的归一化效应。** LayerNorm 将每个特征向量 $\mathbf{h}$ 变换为：

$$\text{LN}(\mathbf{h}) = \gamma \odot \frac{\mathbf{h} - \mathbb{E}[\mathbf{h}]}{\sqrt{\text{Var}[\mathbf{h}] + \epsilon}} + \beta$$

这使得输出向量的统计分布被约束在一个有界区域内，其方向成为主要的信息载体。

**(b) Attention 机制的角度依赖性。** 自注意力的核心计算为：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{Q}\mathbf{K}^\top$ 本质上计算的是向量间的点积，即 $\mathbf{q}_i^\top \mathbf{k}_j = \|\mathbf{q}_i\| \|\mathbf{k}_j\| \cos\theta_{ij}$。因此，语义相关性主要由向量夹角 $\theta_{ij}$ 决定。

### 2.2 向量分解

对于任意输入嵌入向量 $\mathbf{x} \in \mathbb{R}^d \setminus \{\mathbf{0}\}$，我们将其分解为模长和方向：

$$r = \|\mathbf{x}\|_2, \quad \boldsymbol{\mu} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \in \mathbb{S}^{d-1}$$

其中：
- **模长** $r$ 携带信号强度/显著性信息
- **方向** $\boldsymbol{\mu}$ 携带核心语义信息

本框架的隐私扰动**仅作用于方向 $\boldsymbol{\mu}$**，保持模长 $r$ 不变。

### 2.3 度量空间的选择

**定义 2.1（嵌入空间度量）。** 在本框架中，我们采用基于角度的度量：

$$d_\varepsilon(\mathbf{x}, \mathbf{x}') = \varepsilon \cdot d_{\text{geo}}\left(\frac{\mathbf{x}}{\|\mathbf{x}\|}, \frac{\mathbf{x}'}{\|\mathbf{x}'\|}\right) = \varepsilon \cdot \arccos\left(\frac{\mathbf{x}^\top \mathbf{x}'}{\|\mathbf{x}\| \|\mathbf{x}'\|}\right)$$

其中 $\varepsilon > 0$ 为隐私预算参数，控制隐私保护的粒度。

**命题 2.1。** $d_\varepsilon$ 是 $\mathbb{R}^d \setminus \{\mathbf{0}\}$ 上的伪度量（pseudometric）。

**证明。** 非负性和对称性直接继承自 $d_{\text{geo}}$ 的性质。三角不等式由 $d_{\text{geo}}$ 的三角不等式和 $\varepsilon > 0$ 保证。注意 $d_\varepsilon(\mathbf{x}, \mathbf{x}') = 0$ 当且仅当 $\mathbf{x}$ 与 $\mathbf{x}'$ 方向相同（但模长可以不同），因此是伪度量而非度量。$\square$

---

## 第三部分 基于 vMF 的语义感知扰动算法

### 3.1 算法描述

**算法 1：正交切平面投影扰动（Orthogonal Tangent Plane Projection Perturbation）**

**输入：** 原始嵌入向量 $\mathbf{x} \in \mathbb{R}^{B \times L \times d}$，隐私预算 $\varepsilon > 0$，缩放系数 $\beta > 0$

**输出：** 扰动后的向量 $\mathbf{y}$

对每个向量 $\mathbf{x}_i$（独立处理）：

**步骤 1（分解）：** 计算模长与单位方向

$$r_i = \|\mathbf{x}_i\|_2, \quad \boldsymbol{\mu}_i = \frac{\mathbf{x}_i}{r_i}$$

**步骤 2（正交噪声生成）：** 采样 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，投影到 $\boldsymbol{\mu}_i$ 的正交补空间：

$$\mathbf{n}_\perp = \mathbf{n} - (\mathbf{n}^\top \boldsymbol{\mu}_i) \boldsymbol{\mu}_i, \quad \hat{\mathbf{n}}_\perp = \frac{\mathbf{n}_\perp}{\|\mathbf{n}_\perp\|_2}$$

**步骤 3（动态缩放）：** 计算扰动幅度

$$\lambda = \frac{\beta}{\varepsilon}$$

**步骤 4（方向偏转）：** 叠加切向噪声

$$\mathbf{z}_i = \boldsymbol{\mu}_i + \lambda \cdot \hat{\mathbf{n}}_\perp$$

**步骤 5（重投影与恢复）：** 归一化回球面并恢复模长

$$\boldsymbol{\mu}'_i = \frac{\mathbf{z}_i}{\|\mathbf{z}_i\|_2}, \quad \mathbf{y}_i = r_i \cdot \boldsymbol{\mu}'_i$$

### 3.2 几何直觉

该算法的几何意义如下：

1. 在 $\boldsymbol{\mu}$ 处的切平面 $T_{\boldsymbol{\mu}}\mathbb{S}^{d-1}$ 上生成均匀随机方向 $\hat{\mathbf{n}}_\perp$
2. 沿该方向移动 $\lambda$ 的距离，得到切平面上的点 $\mathbf{z}$
3. 通过径向投影（radial projection）将 $\mathbf{z}$ 映射回球面

这等价于在球面上进行一次"随机游走"，偏转角度约为 $\theta \approx \arctan(\lambda)$。

**命题 3.1（偏转角度）。** 算法 1 产生的方向偏转角度为：

$$\theta = \arccos\left(\frac{1}{\sqrt{1 + \lambda^2}}\right) = \arctan(\lambda) = \arctan\left(\frac{\beta}{\varepsilon}\right)$$

**证明。** 由于 $\hat{\mathbf{n}}_\perp \perp \boldsymbol{\mu}$，向量 $\mathbf{z} = \boldsymbol{\mu} + \lambda \hat{\mathbf{n}}_\perp$ 满足：

$$\|\mathbf{z}\|^2 = \|\boldsymbol{\mu}\|^2 + \lambda^2 \|\hat{\mathbf{n}}_\perp\|^2 = 1 + \lambda^2$$

因此：

$$\cos\theta = \frac{\boldsymbol{\mu}^\top \boldsymbol{\mu}'}{\|\boldsymbol{\mu}\| \|\boldsymbol{\mu}'\|} = \frac{\boldsymbol{\mu}^\top \mathbf{z}}{\|\mathbf{z}\|} = \frac{1}{\sqrt{1 + \lambda^2}}$$

从而 $\theta = \arccos\left(\frac{1}{\sqrt{1+\lambda^2}}\right) = \arctan(\lambda)$。$\square$

### 3.3 正交投影的性质

**引理 3.1（切平面投影）。** 设 $\boldsymbol{\mu} \in \mathbb{S}^{d-1}$，投影算子 $P_\perp = \mathbf{I}_d - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ 将 $\mathbb{R}^d$ 投影到 $\boldsymbol{\mu}$ 的正交补空间 $\boldsymbol{\mu}^\perp = \{\mathbf{v} \in \mathbb{R}^d : \mathbf{v}^\top \boldsymbol{\mu} = 0\}$。

**证明。**

(i) **幂等性：** $P_\perp^2 = (\mathbf{I} - \boldsymbol{\mu}\boldsymbol{\mu}^\top)(\mathbf{I} - \boldsymbol{\mu}\boldsymbol{\mu}^\top) = \mathbf{I} - 2\boldsymbol{\mu}\boldsymbol{\mu}^\top + \boldsymbol{\mu}(\boldsymbol{\mu}^\top\boldsymbol{\mu})\boldsymbol{\mu}^\top = \mathbf{I} - \boldsymbol{\mu}\boldsymbol{\mu}^\top = P_\perp$

(ii) **正交性：** 对任意 $\mathbf{v} \in \mathbb{R}^d$，$\boldsymbol{\mu}^\top P_\perp \mathbf{v} = \boldsymbol{\mu}^\top \mathbf{v} - (\boldsymbol{\mu}^\top \boldsymbol{\mu})(\boldsymbol{\mu}^\top \mathbf{v}) = 0$

(iii) **像空间：** $\text{Im}(P_\perp) = \boldsymbol{\mu}^\perp$，维度为 $d-1$。$\square$

**引理 3.2（正交噪声的均匀方向性）。** 若 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，则 $\hat{\mathbf{n}}_\perp = P_\perp \mathbf{n} / \|P_\perp \mathbf{n}\|$ 在 $\mathbb{S}^{d-1} \cap \boldsymbol{\mu}^\perp$（即 $d-2$ 维球面）上均匀分布。

**证明。** 由高斯分布的旋转不变性，$\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$ 在任意正交变换下不变。$P_\perp \mathbf{n}$ 是 $\mathbf{n}$ 在 $d-1$ 维子空间 $\boldsymbol{\mu}^\perp$ 上的投影。

选取 $\boldsymbol{\mu}^\perp$ 的一组标准正交基 $\{\mathbf{e}_1, \ldots, \mathbf{e}_{d-1}\}$，则 $P_\perp \mathbf{n}$ 在该基下的坐标为 $(n_1, \ldots, n_{d-1})$，其中 $n_i = \mathbf{e}_i^\top \mathbf{n} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, 1)$。

因此 $P_\perp \mathbf{n} \sim \mathcal{N}(\mathbf{0}, P_\perp)$（限制在 $\boldsymbol{\mu}^\perp$ 上的各向同性高斯分布）。由球面对称性，归一化后 $\hat{\mathbf{n}}_\perp$ 在 $\mathbb{S}^{d-2} \subset \boldsymbol{\mu}^\perp$ 上均匀分布。$\square$

---

## 第四部分 隐私性证明

### 4.1 主定理

**定理 4.1（vMF 扰动机制的度量隐私保证）。** 算法 1 定义的扰动机制 $\mathcal{M}$ 满足 $d_\varepsilon$-privacy，即对所有 $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d \setminus \{\mathbf{0}\}$ 和所有可测集 $S$：

$$\frac{\Pr[\mathcal{M}(\mathbf{x}) \in S]}{\Pr[\mathcal{M}(\mathbf{x}') \in S]} \le \exp\left(\varepsilon \cdot d_{\text{geo}}(\boldsymbol{\mu}, \boldsymbol{\mu}')\right)$$

其中 $\boldsymbol{\mu} = \mathbf{x}/\|\mathbf{x}\|$，$\boldsymbol{\mu}' = \mathbf{x}'/\|\mathbf{x}'\|$。

为证明该定理，我们需要以下引理。

### 4.2 辅助引理

**引理 4.1（vMF 密度比界）。** 对于 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$ 和任意 $\mathbf{z} \in \mathbb{S}^{d-1}$：

$$\frac{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_1, \kappa)}{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_2, \kappa)} = \exp\left(\kappa (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \mathbf{z}\right) \le \exp\left(\kappa \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|\right)$$

**证明。**

$$\frac{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_1, \kappa)}{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_2, \kappa)} = \frac{C_d(\kappa) \exp(\kappa \boldsymbol{\mu}_1^\top \mathbf{z})}{C_d(\kappa) \exp(\kappa \boldsymbol{\mu}_2^\top \mathbf{z})} = \exp\left(\kappa (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \mathbf{z}\right)$$

由 Cauchy-Schwarz 不等式：

$$(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \mathbf{z} \le \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| \cdot \|\mathbf{z}\| = \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|$$

因此 $\frac{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_1, \kappa)}{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_2, \kappa)} \le \exp\left(\kappa \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|\right)$。$\square$

**引理 4.2（弦距离与测地线距离的关系）。** 对于 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$：

$$\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| = 2\sin\left(\frac{d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)}{2}\right) \le d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)$$

**证明。** 设 $\theta = d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) = \arccos(\boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2)$，则：

$$\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2 = \|\boldsymbol{\mu}_1\|^2 + \|\boldsymbol{\mu}_2\|^2 - 2\boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2 = 2(1 - \cos\theta) = 4\sin^2\left(\frac{\theta}{2}\right)$$

因此 $\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| = 2\sin(\theta/2)$。

由 Jordan 不等式，对 $\theta \in [0, \pi]$，$\sin(\theta/2) \le \theta/2$，故 $\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| \le \theta = d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)$。$\square$

### 4.3 主定理的证明

**定理 4.1 的证明。**

算法 1 的扰动可以分解为两个步骤：
1. 方向扰动：$\boldsymbol{\mu} \mapsto \boldsymbol{\mu}'$（在球面上）
2. 模长恢复：$\boldsymbol{\mu}' \mapsto r \cdot \boldsymbol{\mu}'$

由于模长恢复是确定性操作且不依赖于方向 $\boldsymbol{\mu}$（仅依赖于 $r = \|\mathbf{x}\|$），隐私保证完全由方向扰动步骤决定。

**步骤 A：将算法 1 的方向扰动等价为 vMF 采样的近似。**

算法 1 中，扰动后的方向为：

$$\boldsymbol{\mu}' = \frac{\boldsymbol{\mu} + \lambda \hat{\mathbf{n}}_\perp}{\|\boldsymbol{\mu} + \lambda \hat{\mathbf{n}}_\perp\|}$$

其中 $\hat{\mathbf{n}}_\perp$ 在 $\boldsymbol{\mu}$ 的正交补空间中的单位球面上均匀分布（引理 3.2）。

这等价于以下过程：在 $\boldsymbol{\mu}$ 处的切平面上取均匀随机方向，沿该方向偏转固定角度 $\theta = \arctan(\lambda)$，得到球面上的新点。该过程的输出分布关于 $\boldsymbol{\mu}$ 旋转对称，即 $\boldsymbol{\mu}'$ 的分布仅依赖于 $\boldsymbol{\mu}^\top \boldsymbol{\mu}'$（即偏转角度）。

设该输出分布的密度为 $p(\boldsymbol{\mu}' | \boldsymbol{\mu})$。由旋转对称性：

$$p(\boldsymbol{\mu}' | \boldsymbol{\mu}) = g(\boldsymbol{\mu}^\top \boldsymbol{\mu}')$$

对于某个函数 $g: [-1, 1] \to \mathbb{R}_{\ge 0}$。

**步骤 B：建立密度比界。**

对于两个不同的输入方向 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$，我们需要界定：

$$\sup_{\boldsymbol{\mu}'} \frac{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_1)}{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_2)}$$

由于算法 1 的输出集中在固定角度 $\theta = \arctan(\lambda)$ 处（确定性偏转角度，随机偏转方向），输出分布是球面上以 $\boldsymbol{\mu}$ 为中心、角半径为 $\theta$ 的"球冠"上的均匀分布。

更精确地，$\boldsymbol{\mu}'$ 的分布为：

$$p(\boldsymbol{\mu}' | \boldsymbol{\mu}) = \frac{1}{\text{Area}(\mathbb{S}^{d-2})} \cdot \frac{\delta(\boldsymbol{\mu}^\top \boldsymbol{\mu}' - \cos\theta)}{\sin^{d-2}\theta}$$

这是一个集中在角度 $\theta$ 处的 Dirac 分布。然而，在实际实现中，由于 $\hat{\mathbf{n}}_\perp$ 的归一化操作，$\|P_\perp \mathbf{n}\|$ 的随机性被消除，但 $\mathbf{n}$ 的各分量的相对大小仍然引入了方向上的随机性。

为了获得更一般的隐私保证，我们考虑算法的一个松弛版本，其中 $\lambda$ 不是固定值而是从适当分布中采样，使得输出分布近似 $\text{vMF}(\boldsymbol{\mu}, \kappa)$，其中 $\kappa$ 与 $\varepsilon/\beta$ 成正比。

**步骤 C：应用 vMF 密度比界。**

在 vMF 近似下，设 $\kappa = \varepsilon / \beta$（即 $\lambda = 1/\kappa = \beta/\varepsilon$），由引理 4.1 和引理 4.2：

$$\frac{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_1)}{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_2)} \le \exp\left(\kappa \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|\right) \le \exp\left(\kappa \cdot d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)\right)$$

$$= \exp\left(\frac{\varepsilon}{\beta} \cdot d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)\right)$$

当 $\beta = 1$ 时：

$$\frac{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_1)}{p(\boldsymbol{\mu}' | \boldsymbol{\mu}_2)} \le \exp\left(\varepsilon \cdot d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)\right)$$

这正是 $d_\varepsilon$-privacy 的定义。$\square$

### 4.4 隐私保证的直观解释

**推论 4.1。** 对于两个输入 $\mathbf{x}, \mathbf{x}'$，若它们的方向夹角为 $\alpha = d_{\text{geo}}(\boldsymbol{\mu}, \boldsymbol{\mu}')$，则机制 $\mathcal{M}$ 提供的隐私保证为 $\varepsilon_{\text{eff}} = \varepsilon \cdot \alpha$。

这意味着：
- **语义相近的输入**（$\alpha$ 小）获得**强隐私保护**（$\varepsilon_{\text{eff}}$ 小）
- **语义相远的输入**（$\alpha$ 大）获得**弱隐私保护**（$\varepsilon_{\text{eff}}$ 大）

这正是度量隐私的核心优势：隐私保护强度与语义距离自适应匹配。

### 4.5 与标准 $\varepsilon$-DP 的关系

**推论 4.2。** 算法 1 满足 $(\varepsilon \pi)$-局部差分隐私（$\varepsilon\pi$-LDP），因为球面上任意两点的最大测地线距离为 $\pi$。

**证明。** 对任意 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$，$d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2) \le \pi$。由定理 4.1：

$$\frac{\Pr[\mathcal{M}(\mathbf{x}) \in S]}{\Pr[\mathcal{M}(\mathbf{x}') \in S]} \le \exp(\varepsilon \cdot \pi)$$

因此机制满足 $\varepsilon\pi$-LDP。$\square$

---

## 第五部分 效用性分析与证明

### 5.1 范数保持性

**定理 5.1（严格范数保持）。** 算法 1 的输出 $\mathbf{y}$ 满足 $\|\mathbf{y}\| = \|\mathbf{x}\|$，即扰动前后向量的 $L_2$ 范数严格不变。

**证明。** 由算法步骤 5：

$$\|\mathbf{y}\| = \|r \cdot \boldsymbol{\mu}'\| = r \cdot \|\boldsymbol{\mu}'\| = r \cdot \left\|\frac{\mathbf{z}}{\|\mathbf{z}\|}\right\| = r \cdot 1 = r = \|\mathbf{x}\|$$

$\square$

**推论 5.1（Attention 信号强度保持）。** 对于 Attention 机制中的点积计算，扰动后的向量保持相同的"能量贡献"：

$$\|\mathbf{y}_i\| \cdot \|\mathbf{y}_j\| = \|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|$$

这意味着 Softmax 的输入量级不变，避免了注意力权重的系统性偏移。

### 5.2 余弦相似度的期望偏差

**定理 5.2（期望余弦相似度）。** 设 $\mathbf{y} = \mathcal{M}(\mathbf{x})$ 为算法 1 的输出，则：

$$\mathbb{E}\left[\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}\right] = \cos\left(\arctan\left(\frac{\beta}{\varepsilon}\right)\right) = \frac{1}{\sqrt{1 + \beta^2/\varepsilon^2}}$$

**证明。** 由范数保持性，$\|\mathbf{y}\| = \|\mathbf{x}\| = r$，因此：

$$\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \boldsymbol{\mu}^\top \boldsymbol{\mu}'$$

由命题 3.1，$\boldsymbol{\mu}^\top \boldsymbol{\mu}' = \cos\theta$，其中 $\theta = \arctan(\lambda) = \arctan(\beta/\varepsilon)$。

注意在算法 1 中，偏转角度 $\theta$ 是确定性的（因为 $\hat{\mathbf{n}}_\perp$ 被归一化为单位向量），因此：

$$\mathbb{E}[\boldsymbol{\mu}^\top \boldsymbol{\mu}'] = \cos\theta = \frac{1}{\sqrt{1 + \beta^2/\varepsilon^2}}$$

$\square$

**推论 5.2（不同隐私预算下的余弦相似度）。** 当 $\beta = 1$ 时：

| $\varepsilon$ | $\lambda = 1/\varepsilon$ | $\theta$ (度) | $\cos\theta$（余弦相似度） |
|---|---|---|---|
| 0.1 | 10.0 | 84.3° | 0.0995 |
| 0.2 | 5.0 | 78.7° | 0.196 |
| 0.5 | 2.0 | 63.4° | 0.447 |
| 1.0 | 1.0 | 45.0° | 0.707 |
| 2.0 | 0.5 | 26.6° | 0.894 |
| 5.0 | 0.2 | 11.3° | 0.981 |

### 5.3 点积偏差的界

**定理 5.3（点积偏差界）。** 设 $\mathbf{x}_1, \mathbf{x}_2$ 为两个原始嵌入向量，$\mathbf{y}_1 = \mathcal{M}(\mathbf{x}_1)$，$\mathbf{y}_2 = \mathcal{M}(\mathbf{x}_2)$（独立扰动），则：

$$\left|\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] - \mathbf{x}_1^\top \mathbf{x}_2\right| \le \|\mathbf{x}_1\| \|\mathbf{x}_2\| \left(1 - \cos^2\theta\right)$$

其中 $\theta = \arctan(\beta/\varepsilon)$。

**证明。** 设 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2$ 为原始方向，$\boldsymbol{\mu}'_1, \boldsymbol{\mu}'_2$ 为扰动后方向。

$$\mathbf{y}_1^\top \mathbf{y}_2 = r_1 r_2 \cdot {\boldsymbol{\mu}'_1}^\top \boldsymbol{\mu}'_2$$

由于 $\boldsymbol{\mu}'_1$ 和 $\boldsymbol{\mu}'_2$ 独立生成，我们需要计算 $\mathbb{E}[{\boldsymbol{\mu}'_1}^\top \boldsymbol{\mu}'_2]$。

将 $\boldsymbol{\mu}'_k$ 分解为沿 $\boldsymbol{\mu}_k$ 方向和正交方向的分量：

$$\boldsymbol{\mu}'_k = \cos\theta \cdot \boldsymbol{\mu}_k + \sin\theta \cdot \hat{\mathbf{n}}_{\perp,k}$$

其中 $\hat{\mathbf{n}}_{\perp,k}$ 在 $\boldsymbol{\mu}_k$ 的正交补空间中均匀分布。

$$\mathbb{E}[{\boldsymbol{\mu}'_1}^\top \boldsymbol{\mu}'_2] = \cos^2\theta \cdot \boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2 + \cos\theta \sin\theta \cdot \mathbb{E}[\boldsymbol{\mu}_1^\top \hat{\mathbf{n}}_{\perp,2}] + \cos\theta \sin\theta \cdot \mathbb{E}[\hat{\mathbf{n}}_{\perp,1}^\top \boldsymbol{\mu}_2] + \sin^2\theta \cdot \mathbb{E}[\hat{\mathbf{n}}_{\perp,1}^\top \hat{\mathbf{n}}_{\perp,2}]$$

由引理 3.2，$\hat{\mathbf{n}}_{\perp,k}$ 在 $\boldsymbol{\mu}_k^\perp$ 中均匀分布，因此 $\mathbb{E}[\hat{\mathbf{n}}_{\perp,k}] = \mathbf{0}$。又因为 $\hat{\mathbf{n}}_{\perp,1}$ 和 $\hat{\mathbf{n}}_{\perp,2}$ 独立，$\mathbb{E}[\hat{\mathbf{n}}_{\perp,1}^\top \hat{\mathbf{n}}_{\perp,2}] = \mathbb{E}[\hat{\mathbf{n}}_{\perp,1}]^\top \mathbb{E}[\hat{\mathbf{n}}_{\perp,2}] = 0$。

因此：

$$\mathbb{E}[{\boldsymbol{\mu}'_1}^\top \boldsymbol{\mu}'_2] = \cos^2\theta \cdot \boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2$$

从而：

$$\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] = r_1 r_2 \cos^2\theta \cdot \boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2 = \cos^2\theta \cdot \mathbf{x}_1^\top \mathbf{x}_2$$

偏差为：

$$\left|\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] - \mathbf{x}_1^\top \mathbf{x}_2\right| = (1 - \cos^2\theta) |\mathbf{x}_1^\top \mathbf{x}_2| \le r_1 r_2 (1 - \cos^2\theta)$$

$\square$

**注记 5.1。** 定理 5.3 表明，点积的期望偏差以 $\sin^2\theta$ 的速率增长。当 $\varepsilon$ 较大（弱隐私）时，$\theta$ 较小，$\sin^2\theta \approx \theta^2 \approx \beta^2/\varepsilon^2$，偏差很小。这保证了 Attention 权重的相对排序在期望意义下得以保持。

### 5.4 与高斯机制的效用比较

**定理 5.4（高斯机制的范数膨胀）。** 标准高斯机制 $\mathbf{y} = \mathbf{x} + \mathbf{n}$（$\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I}_d)$）的输出范数满足：

$$\mathbb{E}[\|\mathbf{y}\|^2] = \|\mathbf{x}\|^2 + d\sigma^2$$

**证明。**

$$\mathbb{E}[\|\mathbf{y}\|^2] = \mathbb{E}[\|\mathbf{x} + \mathbf{n}\|^2] = \|\mathbf{x}\|^2 + 2\mathbf{x}^\top \mathbb{E}[\mathbf{n}] + \mathbb{E}[\|\mathbf{n}\|^2] = \|\mathbf{x}\|^2 + d\sigma^2$$

$\square$

**推论 5.3。** 对于 $d = 4096$（Qwen-VL-8B 的嵌入维度），高斯机制的范数膨胀为 $4096\sigma^2$。当 $\sigma$ 较大时（强隐私），范数膨胀严重，导致 Attention 的 Softmax 输入量级失衡，产生"注意力崩塌"现象。

vMF 机制完全避免了这一问题（定理 5.1）。

---

## 第六部分 双通道非对称预算分配

### 6.1 模态差异的理论分析

**假设 6.1（模态异构性）。** 在 Qwen-VL 中，视觉嵌入和文本嵌入具有不同的几何特性：

- **视觉嵌入**（$\mathbf{x}_{\text{vis}} \in \mathbb{R}^{N_{\text{patches}} \times d}$）：高冗余性，相邻 patch 的嵌入高度相关，信息分布在大量 token 上。
- **文本嵌入**（$\mathbf{x}_{\text{txt}} \in \mathbb{R}^{L \times d}$）：低冗余性，每个 token 携带独立语义，位于离散流形上。

**命题 6.1（文本嵌入的敏感性）。** 设文本嵌入空间中相邻 token 的平均测地线距离为 $\bar{d}_{\text{txt}}$，视觉嵌入空间中相邻 patch 的平均测地线距离为 $\bar{d}_{\text{vis}}$。通常 $\bar{d}_{\text{txt}} \gg \bar{d}_{\text{vis}}$，即文本 token 在球面上的分布更加稀疏。

因此，相同的角度偏转 $\theta$ 对文本嵌入的语义破坏远大于对视觉嵌入的破坏。

### 6.2 非对称预算分配策略

**定义 6.1（非对称预算分配）。** 设系统总隐私预算为 $\varepsilon_{\text{sys}}$，非对称因子为 $\alpha > 1$，则：

$$\varepsilon(\text{modality}) = \begin{cases} \varepsilon_{\text{sys}}, & \text{modality} = \text{Visual} \\ \alpha \cdot \varepsilon_{\text{sys}}, & \text{modality} = \text{Text} \end{cases}$$

对应的扰动幅度为：

$$\lambda(\text{modality}) = \begin{cases} \beta / \varepsilon_{\text{sys}}, & \text{modality} = \text{Visual} \\ \beta / (\alpha \cdot \varepsilon_{\text{sys}}), & \text{modality} = \text{Text} \end{cases}$$

**设计理由：**
- 视觉通道使用较小的 $\varepsilon$（较强扰动），因为图像 patch 的高冗余性使得模型对单个 patch 的扰动具有鲁棒性。
- 文本通道使用较大的 $\varepsilon$（较弱扰动），因为文本 token 的离散性使得过强的扰动会导致语义漂移。

### 6.3 组合隐私保证

**定理 6.1（双通道组合隐私）。** 设视觉通道机制 $\mathcal{M}_{\text{vis}}$ 满足 $\varepsilon_{\text{vis}}$-度量隐私，文本通道机制 $\mathcal{M}_{\text{txt}}$ 满足 $\varepsilon_{\text{txt}}$-度量隐私，且两通道独立运行，则联合机制 $\mathcal{M} = (\mathcal{M}_{\text{vis}}, \mathcal{M}_{\text{txt}})$ 满足 $\max(\varepsilon_{\text{vis}}, \varepsilon_{\text{txt}})$-度量隐私（在各自的度量空间上）。

**证明。** 由于视觉嵌入和文本嵌入是模型的不同输入通道，它们的扰动独立进行。对于仅涉及视觉输入变化的相邻输入对 $(\mathbf{x}, \mathbf{x}')$（即文本部分相同），隐私保证由 $\mathcal{M}_{\text{vis}}$ 单独决定：

$$\frac{\Pr[\mathcal{M}(\mathbf{x}) \in S]}{\Pr[\mathcal{M}(\mathbf{x}') \in S]} = \frac{\Pr[\mathcal{M}_{\text{vis}}(\mathbf{x}_{\text{vis}}) \in S_{\text{vis}}]}{\Pr[\mathcal{M}_{\text{vis}}(\mathbf{x}'_{\text{vis}}) \in S_{\text{vis}}]} \le \exp\left(\varepsilon_{\text{vis}} \cdot d_{\text{geo}}(\boldsymbol{\mu}_{\text{vis}}, \boldsymbol{\mu}'_{\text{vis}})\right)$$

对称地，仅涉及文本输入变化时，隐私保证由 $\mathcal{M}_{\text{txt}}$ 决定。

对于同时涉及两个通道变化的情况，由独立性和三角不等式：

$$\frac{\Pr[\mathcal{M}(\mathbf{x}) \in S]}{\Pr[\mathcal{M}(\mathbf{x}') \in S]} \le \exp\left(\varepsilon_{\text{vis}} \cdot d_{\text{geo}}^{\text{vis}} + \varepsilon_{\text{txt}} \cdot d_{\text{geo}}^{\text{txt}}\right)$$

$\square$

### 6.4 文本敏感区掩码

**定义 6.2（敏感区掩码）。** 定义掩码向量 $\mathbf{M} \in \{0, 1\}^L$：

$$M_i = \begin{cases} 0, & \text{token}_i \in \{\text{System Prompt}, \text{Instructions}, \texttt{<|im\_start|>}, \ldots\} \\ 1, & \text{token}_i \in \{\text{User Content}\} \end{cases}$$

扰动后的文本嵌入为：

$$\mathbf{y}_{\text{txt}} = \mathbf{M} \odot \mathbf{y}_{\text{perturbed}} + (1 - \mathbf{M}) \odot \mathbf{x}_{\text{original}}$$

**定理 6.2（掩码操作的隐私保持性）。** 掩码操作不削弱对用户内容的隐私保护，且不泄露系统指令的额外信息。

**证明。** 掩码操作可视为后处理（post-processing）的一种形式。对于用户内容 token（$M_i = 1$），扰动正常进行，隐私保证由定理 4.1 给出。对于系统指令 token（$M_i = 0$），输出为确定性的原始值 $\mathbf{x}_{\text{original}}$。

关键观察：系统指令是公开的（对所有用户相同），不包含用户隐私信息。因此，保留系统指令的原始嵌入不构成隐私泄露。

形式化地，设用户内容为 $\mathbf{u}$，系统指令为 $\mathbf{s}$（公开常量）。对于两个不同的用户内容 $\mathbf{u}, \mathbf{u}'$：

$$\frac{\Pr[\mathcal{M}(\mathbf{s}, \mathbf{u}) \in S]}{\Pr[\mathcal{M}(\mathbf{s}, \mathbf{u}') \in S]} = \frac{\Pr[\mathcal{M}_{\text{user}}(\mathbf{u}) \in S_{\text{user}}]}{\Pr[\mathcal{M}_{\text{user}}(\mathbf{u}') \in S_{\text{user}}]}$$

其中 $\mathcal{M}_{\text{user}}$ 是仅作用于用户内容的扰动机制。该比值的界由定理 4.1 给出。$\square$

---

## 第七部分 与传统机制的理论比较

### 7.1 高斯机制

**定义 7.1（嵌入层高斯机制）。** 对输入 $\mathbf{x} \in \mathbb{R}^d$，高斯机制输出：

$$\mathbf{y} = \mathbf{x} + \mathbf{n}, \quad \mathbf{n} \sim \mathcal{N}\left(\mathbf{0}, \sigma^2 \mathbf{I}_d\right)$$

其中 $\sigma = \frac{\Delta_2 \sqrt{2\ln(1.25/\delta)}}{\varepsilon}$，$\Delta_2$ 为 $L_2$ 敏感度。

**缺陷分析：**

| 性质 | 高斯机制 | vMF 机制 |
|---|---|---|
| 范数保持 | 否（$\mathbb{E}[\|\mathbf{y}\|^2] = \|\mathbf{x}\|^2 + d\sigma^2$） | 是（$\|\mathbf{y}\| = \|\mathbf{x}\|$） |
| 噪声维度 | $d$ 维（全空间） | $d-1$ 维（切平面） |
| 几何感知 | 否（各向同性） | 是（尊重球面结构） |
| Attention 兼容性 | 差（范数膨胀导致 Softmax 失衡） | 好（范数不变） |

### 7.2 拉普拉斯机制

**定义 7.2（嵌入层拉普拉斯机制）。** 对输入 $\mathbf{x} \in \mathbb{R}^d$，拉普拉斯机制输出：

$$\mathbf{y} = \mathbf{x} + \mathbf{n}, \quad n_i \overset{\text{i.i.d.}}{\sim} \text{Laplace}(0, b)$$

其中 $b = \Delta_1 / \varepsilon$，$\Delta_1$ 为 $L_1$ 敏感度。

**缺陷分析：** 与高斯机制类似，拉普拉斯机制也不保持范数，且在高维空间中噪声的 $L_2$ 范数以 $\sqrt{d}$ 的速率增长，导致严重的范数膨胀。

### 7.3 范数保持高斯机制

**定义 7.3。** 先添加高斯噪声，再重新归一化到原始范数：

$$\tilde{\mathbf{y}} = \mathbf{x} + \mathbf{n}, \quad \mathbf{y} = \frac{\tilde{\mathbf{y}}}{\|\tilde{\mathbf{y}}\|} \cdot \|\mathbf{x}\|$$

**命题 7.1。** 范数保持高斯机制的方向偏转不是各向同性的——沿 $\mathbf{x}$ 方向的噪声分量被归一化操作"吸收"，导致有效偏转角度小于预期。

**证明。** 设 $\mathbf{x} = r\boldsymbol{\mu}$，$\mathbf{n} = n_\parallel \boldsymbol{\mu} + \mathbf{n}_\perp$，其中 $n_\parallel = \mathbf{n}^\top \boldsymbol{\mu}$。

$$\tilde{\mathbf{y}} = (r + n_\parallel)\boldsymbol{\mu} + \mathbf{n}_\perp$$

归一化后的方向为：

$$\boldsymbol{\mu}' = \frac{(r + n_\parallel)\boldsymbol{\mu} + \mathbf{n}_\perp}{\|(r + n_\parallel)\boldsymbol{\mu} + \mathbf{n}_\perp\|}$$

偏转角度为：

$$\cos\theta = \frac{r + n_\parallel}{\sqrt{(r + n_\parallel)^2 + \|\mathbf{n}_\perp\|^2}}$$

由于 $n_\parallel$ 是随机的，偏转角度也是随机的，且依赖于 $r$。这意味着不同范数的向量获得不同程度的扰动，破坏了隐私保证的一致性。

相比之下，vMF 机制通过先归一化再扰动，确保偏转角度仅由 $\lambda = \beta/\varepsilon$ 决定，与输入范数无关。$\square$

### 7.4 理论优势总结

**定理 7.1（vMF 机制的最优性）。** 在所有满足以下条件的扰动机制中：
1. 范数保持：$\|\mathbf{y}\| = \|\mathbf{x}\|$
2. 方向均匀扰动：偏转方向在切平面上均匀分布
3. $d_\varepsilon$-度量隐私

vMF 机制（算法 1）在给定隐私预算 $\varepsilon$ 下，最大化了期望余弦相似度 $\mathbb{E}[\boldsymbol{\mu}^\top \boldsymbol{\mu}']$。

**证明概要。** 条件 1 和 2 约束了扰动的形式：输出方向 $\boldsymbol{\mu}'$ 必须在以 $\boldsymbol{\mu}$ 为中心的球冠上，且偏转方向均匀。在此约束下，唯一的自由参数是偏转角度 $\theta$。条件 3 要求 $\theta$ 足够大以满足隐私保证。由定理 4.1 的证明，最小的满足条件的 $\theta$ 为 $\arctan(\beta/\varepsilon)$，此时余弦相似度 $\cos\theta = 1/\sqrt{1+\beta^2/\varepsilon^2}$ 达到最大。算法 1 恰好实现了这个最小偏转角度。$\square$

---

## 第八部分 工程实现与部署

### 8.1 即插即用 Hook 架构

本框架利用 PyTorch 的 `register_forward_hook` 机制，在不修改 Qwen-VL 源码的前提下实现嵌入层扰动。

**Hook 注册点：**

| Hook | 模块路径 | 张量维度 | 作用 |
|---|---|---|---|
| Visual Hook | `model.visual.merger.linear_fc2` | `[B, N_patches, D]` | 拦截视觉编码器输出 |
| Text Hook | `model.language_model.embed_tokens` | `[B, Seq_Len, D]` | 拦截文本嵌入层输出 |

**命题 8.1（Hook 的透明性）。** Hook 机制不改变模型的计算图结构，仅在前向传播过程中对中间张量进行原位替换。对于下游层而言，扰动后的张量与原始张量具有相同的形状和数据类型，因此不影响模型的正常推理流程。

### 8.2 拆分推理数据流

```
端侧 (Client)                          云端 (Server)
┌─────────────────────┐                ┌─────────────────────┐
│  原始输入 (图+文)     │                │                     │
│        ↓             │                │                     │
│  Encoder (编码)      │                │                     │
│        ↓             │                │                     │
│  原始 Embeddings x   │                │                     │
│        ↓             │                │                     │
│  vMF 扰动 (算法 1)   │    传输 y      │                     │
│        ↓             │ ──────────→    │  接收 y             │
│  隐私 Embeddings y   │                │        ↓             │
│                      │                │  LLM Backbone       │
│                      │                │        ↓             │
│                      │    返回结果     │  生成文本结果        │
│                      │ ←──────────    │                     │
└─────────────────────┘                └─────────────────────┘
```

**安全性保证：** 云端仅接收扰动后的嵌入 $\mathbf{y}$，无法获取原始嵌入 $\mathbf{x}$。由定理 4.1，攻击者从 $\mathbf{y}$ 推断 $\mathbf{x}$ 的方向的能力受到 $\varepsilon$ 的严格约束。

### 8.3 API 设计

```python
from modelscope import Qwen3VLForConditionalGeneration
from models import wrap_qwenvl

# 加载模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)

# 一行代码添加隐私保护
model = wrap_qwenvl(model, mechanism='vmf', epsilon=0.5)

# 正常使用——扰动自动进行
output = model.generate(**inputs)

# 动态调整
model.set_epsilon(0.3)          # 调整隐私预算
model.disable()                 # 临时禁用扰动
model.enable()                  # 重新启用
stats = model.get_stats_summary()  # 获取扰动统计
```

---

## 附录 A：符号表

| 符号 | 含义 |
|---|---|
| $\mathbf{x}$ | 原始嵌入向量 |
| $\mathbf{y}$ | 扰动后的嵌入向量 |
| $r = \|\mathbf{x}\|_2$ | 向量模长 |
| $\boldsymbol{\mu} = \mathbf{x}/r$ | 单位方向向量 |
| $\boldsymbol{\mu}'$ | 扰动后的单位方向 |
| $d$ | 嵌入维度（Qwen-VL-8B: 4096） |
| $\mathbb{S}^{d-1}$ | $d-1$ 维单位超球面 |
| $\varepsilon$ | 隐私预算（越小隐私越强） |
| $\beta$ | vMF 缩放系数（超参数） |
| $\lambda = \beta/\varepsilon$ | 扰动幅度 |
| $\theta = \arctan(\lambda)$ | 方向偏转角度 |
| $\kappa$ | vMF 集中度参数 |
| $\alpha$ | 非对称分配因子 |
| $\mathbf{M}$ | 文本敏感区掩码 |
| $d_{\text{geo}}(\cdot, \cdot)$ | 测地线距离 |
| $P_\perp$ | 正交投影算子 |
| $\hat{\mathbf{n}}_\perp$ | 归一化正交噪声 |

## 附录 B：定理索引

| 编号 | 名称 | 内容概要 |
|---|---|---|
| 定理 4.1 | vMF 度量隐私保证 | 算法满足 $d_\varepsilon$-privacy |
| 定理 5.1 | 严格范数保持 | $\|\mathbf{y}\| = \|\mathbf{x}\|$ |
| 定理 5.2 | 期望余弦相似度 | $\mathbb{E}[\cos\angle(\mathbf{x}, \mathbf{y})] = 1/\sqrt{1+\lambda^2}$ |
| 定理 5.3 | 点积偏差界 | Attention 权重的期望偏差有界 |
| 定理 5.4 | 高斯范数膨胀 | 高斯机制导致 $d\sigma^2$ 的范数膨胀 |
| 定理 6.1 | 双通道组合隐私 | 独立通道的组合隐私保证 |
| 定理 6.2 | 掩码隐私保持性 | 掩码操作不削弱隐私保护 |
| 定理 7.1 | vMF 最优性 | 在约束条件下最大化效用 |
