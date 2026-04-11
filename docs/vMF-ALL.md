# 面向多模态大模型嵌入层的语义感知度量隐私保护机制

## 目录

- [第一部分 预备知识](#第一部分-预备知识)
- [第二部分 语义流形与度量空间](#第二部分-语义流形与度量空间)
- [第三部分 vMF 语义感知扰动算法](#第三部分-vmf-语义感知扰动算法)
- [第四部分 隐私性证明](#第四部分-隐私性证明)
- [第五部分 效用性分析](#第五部分-效用性分析)
- [第六部分 双通道非对称预算分配](#第六部分-双通道非对称预算分配)
- [第七部分 与传统机制的比较](#第七部分-与传统机制的比较)
- [第八部分 工程实现](#第八部分-工程实现)
- [附录 A 符号表](#附录-a-符号表)
- [附录 B 定理索引](#附录-b-定理索引)

---

## 第一部分 预备知识

### 1.1 差分隐私

**定义 1.1（$(\varepsilon, \delta)$-差分隐私）。**
随机机制 $\mathcal{M}: \mathcal{X} \to \mathcal{Y}$ 满足 $(\varepsilon, \delta)$-差分隐私，若对所有相邻数据集 $D, D'$（至多相差一条记录）及所有可测集 $S \subseteq \mathcal{Y}$：

$$\Pr[\mathcal{M}(D) \in S] \le e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

### 1.2 度量隐私

**定义 1.2（$d_\mathcal{X}$-隐私）。**
设 $(\mathcal{X}, d_\mathcal{X})$ 为度量空间。随机机制 $\mathcal{M}: \mathcal{X} \to \mathcal{Y}$ 满足 $d_\mathcal{X}$-隐私，若对所有 $x, x' \in \mathcal{X}$ 及所有可测集 $S \subseteq \mathcal{Y}$：

$$\Pr[\mathcal{M}(x) \in S] \le e^{d_\mathcal{X}(x, x')} \cdot \Pr[\mathcal{M}(x') \in S]$$

当 $d_\mathcal{X}(x, x') = \varepsilon \|x - x'\|$ 时退化为标准局部差分隐私。

**注记 1.1。** 度量隐私的核心优势：隐私保护强度与数据点间的语义距离成正比——语义相近的输入获得相似的输出分布，语义相远的输入获得差异更大的输出分布。这天然适合嵌入空间中的隐私保护。

### 1.3 von Mises-Fisher 分布

**定义 1.3（vMF 分布）。**
$d$ 维单位超球面 $\mathbb{S}^{d-1}$ 上的 vMF 分布 $\text{vMF}(\boldsymbol{\mu}, \kappa)$ 的密度函数为：

$$f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}, \kappa) = C_d(\kappa) \exp\left(\kappa \boldsymbol{\mu}^\top \mathbf{z}\right)$$

其中 $\boldsymbol{\mu} \in \mathbb{S}^{d-1}$ 为均值方向，$\kappa \ge 0$ 为集中度，$C_d(\kappa) = \kappa^{d/2-1} / \left[(2\pi)^{d/2} I_{d/2-1}(\kappa)\right]$ 为归一化常数，$I_\nu$ 为第一类修正 Bessel 函数。

**性质 1.1。** $\kappa = 0$ 时退化为球面均匀分布；$\kappa \to \infty$ 时集中于 $\boldsymbol{\mu}$ 处的点质量。

**性质 1.2。** $\mathbf{z} \sim \text{vMF}(\boldsymbol{\mu}, \kappa)$ 时，$\mathbb{E}[\mathbf{z}] = A_d(\kappa) \boldsymbol{\mu}$，其中 $A_d(\kappa) = I_{d/2}(\kappa) / I_{d/2-1}(\kappa)$。

### 1.4 测地线距离

**定义 1.4。** 对 $\mathbf{u}, \mathbf{v} \in \mathbb{S}^{d-1}$，测地线距离为：

$$d_{\text{geo}}(\mathbf{u}, \mathbf{v}) = \arccos(\mathbf{u}^\top \mathbf{v})$$

即 $\mathbb{S}^{d-1}$ 上的大圆弧长。满足度量公理（非负性、同一性、对称性、三角不等式）。

---

## 第二部分 语义流形与度量空间

### 2.1 超球面嵌入假设

**假设 2.1。** Qwen-VL 模型中，经 LayerNorm 标准化后的特征向量的有效语义信息主要编码在其方向上（$d = 4096$，Qwen-VL-8B）。本框架通过显式的范数-方向分解（§3.1 步骤 1），将方向投影到 $\mathbb{S}^{d-1}$ 上进行扰动，再恢复原始范数。因此超球面假设是算法构造的一部分，而非对原始嵌入分布的约束。

**理论依据：**

**(a) LayerNorm 的归一化效应。** LayerNorm 将特征向量 $\mathbf{h}$ 变换为：

$$\text{LN}(\mathbf{h}) = \gamma \odot \frac{\mathbf{h} - \mathbb{E}[\mathbf{h}]}{\sqrt{\text{Var}[\mathbf{h}] + \epsilon}} + \beta$$

输出向量的统计分布被约束在有界区域内，方向成为主要信息载体。

**(b) Attention 的角度依赖性。** 自注意力核心计算：

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^\top}{\sqrt{d_k}}\right)\mathbf{V}$$

其中 $\mathbf{q}_i^\top \mathbf{k}_j = \|\mathbf{q}_i\| \|\mathbf{k}_j\| \cos\theta_{ij}$，语义相关性主要由夹角 $\theta_{ij}$ 决定。

### 2.2 向量分解

对任意 $\mathbf{x} \in \mathbb{R}^d \setminus \{\mathbf{0}\}$，分解为模长和方向：

$$r = \|\mathbf{x}\|_2, \quad \boldsymbol{\mu} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2} \in \mathbb{S}^{d-1}$$

- **模长** $r$：信号强度/显著性
- **方向** $\boldsymbol{\mu}$：核心语义信息

本框架的隐私扰动**仅作用于方向 $\boldsymbol{\mu}$**，保持模长 $r$ 不变。

### 2.3 度量空间

**定义 2.1（嵌入空间度量）。** 采用基于角度的度量：

$$d_\varepsilon(\mathbf{x}, \mathbf{x}') = \varepsilon \cdot d_{\text{geo}}\left(\frac{\mathbf{x}}{\|\mathbf{x}\|}, \frac{\mathbf{x}'}{\|\mathbf{x}'\|}\right) = \varepsilon \cdot \arccos\left(\frac{\mathbf{x}^\top \mathbf{x}'}{\|\mathbf{x}\| \|\mathbf{x}'\|}\right)$$

**命题 2.1。** $d_\varepsilon$ 是 $\mathbb{R}^d \setminus \{\mathbf{0}\}$ 上的伪度量。

**证明。** 非负性、对称性继承自 $d_{\text{geo}}$，三角不等式由 $\varepsilon > 0$ 保证。$d_\varepsilon(\mathbf{x}, \mathbf{x}') = 0$ 当且仅当方向相同（模长可不同），故为伪度量。$\square$

---

## 第三部分 vMF 语义感知扰动算法

### 3.1 算法描述

**算法 1：正交切平面投影扰动（Orthogonal Tangent Plane Projection Perturbation）**

**输入：** 嵌入向量 $\mathbf{x} \in \mathbb{R}^{B \times L \times d}$，隐私预算 $\varepsilon > 0$，缩放系数 $\beta > 0$

**输出：** 扰动后的向量 $\mathbf{y}$

对每个向量 $\mathbf{x}_i$ 独立处理：

**步骤 1（分解）：**

$$r_i = \|\mathbf{x}_i\|_2, \quad \boldsymbol{\mu}_i = \mathbf{x}_i / r_i$$

**步骤 2（正交噪声）：** 采样 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，投影到 $\boldsymbol{\mu}_i$ 的正交补空间：

$$\mathbf{n}_\perp = \mathbf{n} - (\mathbf{n}^\top \boldsymbol{\mu}_i) \boldsymbol{\mu}_i, \quad \hat{\mathbf{n}}_\perp = \mathbf{n}_\perp / \|\mathbf{n}_\perp\|_2$$

**步骤 3（缩放）：**

$$\lambda = \beta / \varepsilon$$

**步骤 4（偏转）：**

$$\mathbf{z}_i = \boldsymbol{\mu}_i + \lambda \cdot \hat{\mathbf{n}}_\perp$$

**步骤 5（重投影）：**

$$\boldsymbol{\mu}'_i = \mathbf{z}_i / \|\mathbf{z}_i\|_2, \quad \mathbf{y}_i = r_i \cdot \boldsymbol{\mu}'_i$$

### 3.2 几何直觉

算法在 $\boldsymbol{\mu}$ 处的切平面 $T_{\boldsymbol{\mu}}\mathbb{S}^{d-1}$ 上生成均匀随机方向 $\hat{\mathbf{n}}_\perp$，沿该方向移动 $\lambda$ 距离，再通过径向投影映射回球面。等价于球面上的一次随机游走。

**命题 3.1（偏转角度）。**

$$\theta = \arccos\left(\frac{1}{\sqrt{1 + \lambda^2}}\right) = \arctan(\lambda) = \arctan\left(\frac{\beta}{\varepsilon}\right)$$

**证明。** $\hat{\mathbf{n}}_\perp \perp \boldsymbol{\mu}$，故 $\|\mathbf{z}\|^2 = 1 + \lambda^2$，从而 $\cos\theta = \boldsymbol{\mu}^\top \mathbf{z} / \|\mathbf{z}\| = 1/\sqrt{1 + \lambda^2}$。$\square$

### 3.3 正交投影性质

**引理 3.1（切平面投影）。** 投影算子 $P_\perp = \mathbf{I}_d - \boldsymbol{\mu}\boldsymbol{\mu}^\top$ 将 $\mathbb{R}^d$ 投影到 $\boldsymbol{\mu}^\perp = \{\mathbf{v} : \mathbf{v}^\top \boldsymbol{\mu} = 0\}$。

**证明。** (i) 幂等性：$P_\perp^2 = P_\perp$（利用 $\boldsymbol{\mu}^\top \boldsymbol{\mu} = 1$）。(ii) 正交性：$\boldsymbol{\mu}^\top P_\perp \mathbf{v} = 0$。(iii) 像空间维度为 $d-1$。$\square$

**引理 3.2（方向均匀性）。** 若 $\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}_d)$，则 $\hat{\mathbf{n}}_\perp = P_\perp \mathbf{n} / \|P_\perp \mathbf{n}\|$ 在 $\mathbb{S}^{d-2} \subset \boldsymbol{\mu}^\perp$ 上均匀分布。

**证明。** 高斯分布的旋转不变性保证 $P_\perp \mathbf{n}$ 在 $\boldsymbol{\mu}^\perp$ 上各向同性。选取标准正交基 $\{\mathbf{e}_1, \ldots, \mathbf{e}_{d-1}\}$，投影坐标 $n_i = \mathbf{e}_i^\top \mathbf{n} \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0,1)$。由球面对称性，归一化后在 $\mathbb{S}^{d-2}$ 上均匀分布。$\square$

---

## 第四部分 隐私性证明

### 4.1 主定理

**定理 4.1（近似度量隐私）。** 设 $d$ 为嵌入维度，$\delta_{\text{TV}}(d)$ 为切平面分布与匹配 vMF 分布间的全变差距离。算法 1 的扰动机制 $\mathcal{M}$ 满足 $(\kappa^*, \delta_{\text{TV}})$-近似度量隐私：对所有 $\mathbf{x}, \mathbf{x}' \in \mathbb{R}^d \setminus \{\mathbf{0}\}$ 和可测集 $S$，

$$\Pr[\mathcal{M}(\mathbf{x}) \in S] \le e^{\kappa^* \cdot d_{\text{geo}}(\boldsymbol{\mu}, \boldsymbol{\mu}')} \cdot \Pr[\mathcal{M}(\mathbf{x}') \in S] + 2\delta_{\text{TV}}(d)$$

其中 $\boldsymbol{\mu} = \mathbf{x}/\|\mathbf{x}\|$，$\boldsymbol{\mu}' = \mathbf{x}'/\|\mathbf{x}'\|$，$\kappa^*$ 由 §4.3 步骤 B 确定。在 $d = 4096$ 下 $\delta_{\text{TV}}$ 可忽略（见注记 4.2）。

### 4.2 辅助引理

**引理 4.1（vMF 密度比界）。** 对 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$ 和任意 $\mathbf{z} \in \mathbb{S}^{d-1}$：

$$\frac{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_1, \kappa)}{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_2, \kappa)} = \exp\left(\kappa (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \mathbf{z}\right) \le \exp\left(\kappa \|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|\right)$$

**证明。** 归一化常数 $C_d(\kappa)$ 消去后，密度比为 $\exp(\kappa(\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2)^\top \mathbf{z})$。由 Cauchy-Schwarz 不等式及 $\|\mathbf{z}\| = 1$ 得上界。$\square$

**引理 4.2（弦距离-测地线距离）。** 对 $\boldsymbol{\mu}_1, \boldsymbol{\mu}_2 \in \mathbb{S}^{d-1}$：

$$\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\| = 2\sin\left(\frac{d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)}{2}\right) \le d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)$$

**证明。** 设 $\theta = \arccos(\boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2)$，则 $\|\boldsymbol{\mu}_1 - \boldsymbol{\mu}_2\|^2 = 2(1-\cos\theta) = 4\sin^2(\theta/2)$。由 $\sin(\theta/2) \le \theta/2$ 得结论。$\square$

**引理 4.3（TV 距离的隐私传递）。** 若机制 $\mathcal{M}_1$ 满足 $\varepsilon$-度量隐私，且对所有输入 $\mathbf{x}$ 有 $d_{\text{TV}}(\mathcal{M}_1(\mathbf{x}), \mathcal{M}_2(\mathbf{x})) \le \delta$，则 $\mathcal{M}_2$ 满足 $(\varepsilon, 2\delta)$-近似度量隐私。

**证明。** 对任意可测集 $S$ 和输入 $\mathbf{x}, \mathbf{x}'$：

$$\Pr[\mathcal{M}_2(\mathbf{x}) \in S] \le \Pr[\mathcal{M}_1(\mathbf{x}) \in S] + \delta \le e^{\varepsilon \cdot d(\mathbf{x}, \mathbf{x}')} \Pr[\mathcal{M}_1(\mathbf{x}') \in S] + \delta$$

$$\le e^{\varepsilon \cdot d(\mathbf{x}, \mathbf{x}')} (\Pr[\mathcal{M}_2(\mathbf{x}') \in S] + \delta) + \delta \le e^{\varepsilon \cdot d(\mathbf{x}, \mathbf{x}')} \Pr[\mathcal{M}_2(\mathbf{x}') \in S] + 2\delta$$

$\square$

### 4.3 主定理证明

**定理 4.1 的证明。**

算法 1 分解为方向扰动 $\boldsymbol{\mu} \mapsto \boldsymbol{\mu}'$ 和模长恢复 $\boldsymbol{\mu}' \mapsto r\boldsymbol{\mu}'$。模长恢复是确定性后处理，不影响隐私保证。

**步骤 A：输出分布刻画。**

扰动方向 $\boldsymbol{\mu}' = (\boldsymbol{\mu} + \lambda\hat{\mathbf{n}}_\perp) / \|\boldsymbol{\mu} + \lambda\hat{\mathbf{n}}_\perp\|$，其中 $\hat{\mathbf{n}}_\perp$ 在 $\boldsymbol{\mu}^\perp$ 的单位球面上均匀分布（引理 3.2）。偏转角度为确定值 $\theta = \arctan(\lambda)$（命题 3.1），故 $\boldsymbol{\mu}'$ 均匀分布在球环 $\mathcal{C}(\boldsymbol{\mu}, \theta)$ 上。记该分布为 $P_{\text{TP}}(\cdot|\boldsymbol{\mu})$。

**步骤 B：构造 vMF 参考机制。**

定义参考机制 $\mathcal{M}_{\text{vMF}}$：从 $\text{vMF}(\boldsymbol{\mu}, \kappa^*)$ 采样，其中 $\kappa^*$ 通过数值求解确定：

$$A_d(\kappa^*) = \frac{I_{d/2}(\kappa^*)}{I_{d/2-1}(\kappa^*)} = \cos\theta = \frac{1}{\sqrt{1 + \beta^2/\varepsilon^2}}$$

**注记 4.1。** $\kappa^*$ 与 $\varepsilon$ 的关系是隐式的，需数值求解。例如 $d=4096$，$\varepsilon=0.5$，$\beta=1$ 时 $\kappa^* \approx 2290$。

由引理 4.1 和 4.2，$\mathcal{M}_{\text{vMF}}$ 满足严格 $\kappa^*$-度量隐私：

$$\frac{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_1, \kappa^*)}{f_{\text{vMF}}(\mathbf{z}; \boldsymbol{\mu}_2, \kappa^*)} \le \exp\left(\kappa^* \cdot d_{\text{geo}}(\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)\right)$$

**步骤 C：高维浓度与隐私传递。**

切平面分布 $P_{\text{TP}}$ 与 vMF 分布 $P_{\text{vMF}}$ 的全变差距离为 $\delta_{\text{TV}}(d) = d_{\text{TV}}(P_{\text{TP}}(\cdot|\boldsymbol{\mu}), P_{\text{vMF}}(\cdot|\boldsymbol{\mu}))$。

由高维浓度现象，vMF 角度方差随维度衰减：$\text{Var}_{\text{vMF}}[\cos\angle(\boldsymbol{\mu}, \boldsymbol{\mu}')] = O(1/d)$。当 $d \to \infty$ 时 vMF 集中于角度 $\theta$ 处，与切平面的 delta 分布趋于一致，$\delta_{\text{TV}}(d) \to 0$。

由引理 4.3，切平面机制满足 $(\kappa^*, 2\delta_{\text{TV}})$-近似度量隐私。$\square$

**注记 4.2（实验验证）。** $d = 4096$ 下的数值实验（`scripts/validate_vmf_approximation.py`，详见 `docs/vMF-vs-切平面.md`）：

- 角度分布 Wasserstein 距离 $0.58°$（均值一致：$63.43°$）
- vMF 角度标准差仅 $0.73°$
- KS 检验 $p = 0.735$，无法拒绝分布等价
- Rayleigh 检验 $p = 0.918$，各向同性成立

$\delta_{\text{TV}}(4096)$ 在统计意义上可忽略。

### 4.4 隐私保证的解释

**推论 4.1。** 方向夹角为 $\alpha$ 的两个输入，有效隐私保证为 $\varepsilon_{\text{eff}} = \kappa^* \alpha$（忽略 $\delta_{\text{TV}}$）。语义相近（$\alpha$ 小）获强保护，语义相远（$\alpha$ 大）获弱保护——度量隐私的核心优势。

### 4.5 与标准 DP 的关系

**推论 4.2。** 算法 1 满足 $(\kappa^*\pi, 2\delta_{\text{TV}})$-近似 LDP。

**证明。** 球面上最大测地线距离为 $\pi$，代入定理 4.1 即得。$\square$

---

## 第五部分 效用性分析

### 5.1 范数保持

**定理 5.1（严格范数保持）。** $\|\mathbf{y}\| = \|\mathbf{x}\|$。

**证明。** $\|\mathbf{y}\| = \|r\boldsymbol{\mu}'\| = r\|\boldsymbol{\mu}'\| = r \cdot 1 = \|\mathbf{x}\|$。$\square$

**推论 5.1（Attention 信号强度保持）。** $\|\mathbf{y}_i\| \cdot \|\mathbf{y}_j\| = \|\mathbf{x}_i\| \cdot \|\mathbf{x}_j\|$，Softmax 输入量级不变，避免注意力权重的系统性偏移。

### 5.2 余弦相似度

**定理 5.2（期望余弦相似度）。**

$$\mathbb{E}\left[\frac{\mathbf{x}^\top \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|}\right] = \cos\left(\arctan\frac{\beta}{\varepsilon}\right) = \frac{1}{\sqrt{1 + \beta^2/\varepsilon^2}}$$

**证明。** 由范数保持性，余弦相似度 $= \boldsymbol{\mu}^\top \boldsymbol{\mu}'$。偏转角度 $\theta = \arctan(\beta/\varepsilon)$ 是确定性的，故 $\mathbb{E}[\boldsymbol{\mu}^\top \boldsymbol{\mu}'] = \cos\theta$。$\square$

**推论 5.2（隐私预算-余弦对照表）。** $\beta = 1$ 时：

| $\varepsilon$ | $\lambda$ | $\theta$ | $\cos\theta$ |
|---|---|---|---|
| 0.1 | 10.0 | 84.3° | 0.100 |
| 0.2 | 5.0 | 78.7° | 0.196 |
| 0.5 | 2.0 | 63.4° | 0.447 |
| 1.0 | 1.0 | 45.0° | 0.707 |
| 2.0 | 0.5 | 26.6° | 0.894 |
| 5.0 | 0.2 | 11.3° | 0.981 |

### 5.3 点积偏差

**定理 5.3（点积偏差界）。** 对独立扰动的 $\mathbf{y}_1 = \mathcal{M}(\mathbf{x}_1)$，$\mathbf{y}_2 = \mathcal{M}(\mathbf{x}_2)$：

$$\left|\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] - \mathbf{x}_1^\top \mathbf{x}_2\right| \le \|\mathbf{x}_1\| \|\mathbf{x}_2\| (1 - \cos^2\theta)$$

**证明。** 将 $\boldsymbol{\mu}'_k = \cos\theta \cdot \boldsymbol{\mu}_k + \sin\theta \cdot \hat{\mathbf{n}}_{\perp,k}$ 展开点积期望。由引理 3.2，$\mathbb{E}[\hat{\mathbf{n}}_{\perp,k}] = \mathbf{0}$，独立性保证交叉项为零：

$$\mathbb{E}[{\boldsymbol{\mu}'_1}^\top \boldsymbol{\mu}'_2] = \cos^2\theta \cdot \boldsymbol{\mu}_1^\top \boldsymbol{\mu}_2$$

故 $\mathbb{E}[\mathbf{y}_1^\top \mathbf{y}_2] = \cos^2\theta \cdot \mathbf{x}_1^\top \mathbf{x}_2$，偏差 $= (1-\cos^2\theta)|\mathbf{x}_1^\top \mathbf{x}_2| \le r_1 r_2 \sin^2\theta$。$\square$

**注记 5.1。** $\varepsilon$ 较大时 $\sin^2\theta \approx \beta^2/\varepsilon^2$，偏差很小，Attention 权重的相对排序在期望下保持。

### 5.4 与高斯机制的效用比较

**定理 5.4（高斯范数膨胀）。** 高斯机制 $\mathbf{y} = \mathbf{x} + \mathbf{n}$（$\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_d)$）满足 $\mathbb{E}[\|\mathbf{y}\|^2] = \|\mathbf{x}\|^2 + d\sigma^2$。

**证明。** $\mathbb{E}[\|\mathbf{x}+\mathbf{n}\|^2] = \|\mathbf{x}\|^2 + 2\mathbf{x}^\top\mathbb{E}[\mathbf{n}] + \mathbb{E}[\|\mathbf{n}\|^2] = \|\mathbf{x}\|^2 + d\sigma^2$。$\square$

**推论 5.3。** $d = 4096$ 时范数膨胀为 $4096\sigma^2$，强隐私下导致 Softmax 失衡（"注意力崩塌"）。vMF 机制完全避免此问题（定理 5.1）。

---

## 第六部分 双通道非对称预算分配

### 6.1 模态差异

**假设 6.1（模态异构性）。** Qwen-VL 中视觉与文本嵌入几何性质不同：

- **视觉嵌入**（$\mathbf{x}_{\text{vis}} \in \mathbb{R}^{N_{\text{patches}} \times d}$）：高冗余，相邻 patch 高度相关
- **文本嵌入**（$\mathbf{x}_{\text{txt}} \in \mathbb{R}^{L \times d}$）：低冗余，每个 token 携带独立语义

**命题 6.1。** 文本 token 在球面上分布更稀疏（$\bar{d}_{\text{txt}} \gg \bar{d}_{\text{vis}}$），相同偏转 $\theta$ 对文本的语义破坏远大于视觉。

### 6.2 非对称预算

**定义 6.1。** 系统预算 $\varepsilon_{\text{sys}}$，非对称因子 $\alpha > 1$：

$$\varepsilon(\text{modality}) = \begin{cases} \varepsilon_{\text{sys}}, & \text{Visual} \\ \alpha \cdot \varepsilon_{\text{sys}}, & \text{Text} \end{cases}, \quad \lambda(\text{modality}) = \begin{cases} \beta/\varepsilon_{\text{sys}}, & \text{Visual} \\ \beta/(\alpha\varepsilon_{\text{sys}}), & \text{Text} \end{cases}$$

视觉通道用较小 $\varepsilon$（强扰动，图像冗余可承受），文本通道用较大 $\varepsilon$（弱扰动，防语义漂移）。

### 6.3 组合隐私

**定理 6.1（双通道组合隐私）。** 视觉机制 $\mathcal{M}_{\text{vis}}$ 满足 $\varepsilon_{\text{vis}}$-度量隐私，文本机制 $\mathcal{M}_{\text{txt}}$ 满足 $\varepsilon_{\text{txt}}$-度量隐私，两通道独立，则：

**(a) 单模态威胁。** 对手仅观察单一模态输出时，隐私保证由该模态单独决定。

**(b) 联合威胁。** 对手同时观察两模态输出时：

$$\frac{\Pr[\mathcal{M}(\mathbf{x}) \in S]}{\Pr[\mathcal{M}(\mathbf{x}') \in S]} \le \exp\left(\varepsilon_{\text{vis}} \cdot d_{\text{geo}}^{\text{vis}} + \varepsilon_{\text{txt}} \cdot d_{\text{geo}}^{\text{txt}}\right)$$

**证明。** 仅视觉变化时，文本输出不变，密度比由 $\mathcal{M}_{\text{vis}}$ 决定；对称地处理文本。两通道同时变化时，由独立性得加法组合。$\square$

### 6.4 敏感区掩码

**定义 6.2。** 掩码 $\mathbf{M} \in \{0,1\}^L$：系统指令 $M_i = 0$，用户内容 $M_i = 1$。

$$\mathbf{y}_{\text{txt}} = \mathbf{M} \odot \mathbf{y}_{\text{perturbed}} + (1-\mathbf{M}) \odot \mathbf{x}_{\text{original}}$$

**定理 6.2（掩码隐私保持）。** 掩码不削弱用户内容的隐私保护。

**证明。** 系统指令 $\mathbf{s}$ 为公开常量，不含用户隐私。对不同用户内容 $\mathbf{u}, \mathbf{u}'$：

$$\frac{\Pr[\mathcal{M}(\mathbf{s}, \mathbf{u}) \in S]}{\Pr[\mathcal{M}(\mathbf{s}, \mathbf{u}') \in S]} = \frac{\Pr[\mathcal{M}_{\text{user}}(\mathbf{u}) \in S_{\text{user}}]}{\Pr[\mathcal{M}_{\text{user}}(\mathbf{u}') \in S_{\text{user}}]}$$

界由定理 4.1 给出。$\square$

---

## 第七部分 与传统机制的比较

### 7.1 高斯机制

**定义 7.1。** $\mathbf{y} = \mathbf{x} + \mathbf{n}$，$\mathbf{n} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I}_d)$，$\sigma = \Delta_2\sqrt{2\ln(1.25/\delta)}/\varepsilon$。

| 性质 | 高斯机制 | vMF 机制 |
|---|---|---|
| 范数保持 | 否（膨胀 $d\sigma^2$） | 是（严格） |
| 噪声维度 | $d$ 维全空间 | $d{-}1$ 维切平面 |
| 几何感知 | 否 | 是 |
| Attention 兼容 | 差 | 好 |

### 7.2 拉普拉斯机制

**定义 7.2。** $\mathbf{y} = \mathbf{x} + \mathbf{n}$，$n_i \overset{\text{i.i.d.}}{\sim} \text{Laplace}(0, \Delta_1/\varepsilon)$。不保持范数，高维下 $L_2$ 范数以 $\sqrt{d}$ 增长。

### 7.3 范数保持高斯

**定义 7.3。** $\tilde{\mathbf{y}} = \mathbf{x} + \mathbf{n}$，$\mathbf{y} = \|\mathbf{x}\| \cdot \tilde{\mathbf{y}}/\|\tilde{\mathbf{y}}\|$。

**命题 7.2。** 偏转角度依赖输入范数 $r$：

$$\cos\theta = \frac{r + n_\parallel}{\sqrt{(r+n_\parallel)^2 + \|\mathbf{n}_\perp\|^2}}$$

归一化引入对 $r$ 的非线性依赖，密度比分析无法获得与输入无关的统一界。vMF 通过先归一化再扰动避免此问题。$\square$

### 7.4 理论优势

**命题 7.3（效用最优性）。** 在范数保持、方向均匀、$(\varepsilon,\delta)$-近似度量隐私的约束下，vMF 机制最大化期望余弦相似度。

**论证。** 约束 1、2 限定输出在球冠上均匀分布，自由参数仅为偏转角 $\theta$。约束 3 要求 $\theta$ 足够大。算法 1 取最小 $\theta = \arctan(\beta/\varepsilon)$，余弦 $\cos\theta = 1/\sqrt{1+\beta^2/\varepsilon^2}$ 达最大。严格最优性证明留作未来工作。$\square$

---

## 第八部分 工程实现

### 8.1 Hook 架构

利用 PyTorch `register_forward_hook`，不修改 Qwen-VL 源码：

| Hook | 模块路径 | 张量维度 | 作用 |
|---|---|---|---|
| Visual | `model.visual.merger.linear_fc2` | `[B, N_patches, D]` | 拦截视觉编码器输出 |
| Text | `model.language_model.embed_tokens` | `[B, Seq_Len, D]` | 拦截文本嵌入层输出 |

**命题 8.1（透明性）。** Hook 仅替换中间张量，不改变计算图结构，下游层无感知。

### 8.2 拆分推理

```
端侧 (Client)                          云端 (Server)
┌─────────────────────┐                ┌─────────────────────┐
│  原始输入 (图+文)     │                │                     │
│        ↓             │                │                     │
│  Encoder             │                │                     │
│        ↓             │                │                     │
│  vMF 扰动            │    传输 y      │                     │
│        ↓             │ ──────────→    │  LLM Backbone       │
│  隐私 Embeddings y   │                │        ↓             │
│                      │    返回结果     │  生成结果            │
│                      │ ←──────────    │                     │
└─────────────────────┘                └─────────────────────┘
```

云端仅接收 $\mathbf{y}$，由定理 4.1 约束攻击者推断 $\mathbf{x}$ 方向的能力。

### 8.3 API

```python
from modelscope import Qwen3VLForConditionalGeneration
from models import wrap_qwenvl

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
)
model = wrap_qwenvl(model, mechanism='vmf', epsilon=0.5)

output = model.generate(**inputs)

model.set_epsilon(0.3)
model.disable()
model.enable()
stats = model.get_stats_summary()
```

---

## 附录 A 符号表

| 符号 | 含义 |
|---|---|
| $\mathbf{x}$, $\mathbf{y}$ | 原始/扰动后嵌入 |
| $r = \|\mathbf{x}\|_2$ | 模长 |
| $\boldsymbol{\mu}$, $\boldsymbol{\mu}'$ | 原始/扰动后单位方向 |
| $d$ | 嵌入维度（Qwen-VL-8B: 4096） |
| $\mathbb{S}^{d-1}$ | 单位超球面 |
| $\varepsilon$ | 隐私预算（越小越强） |
| $\beta$ | vMF 缩放系数 |
| $\lambda = \beta/\varepsilon$ | 扰动幅度 |
| $\theta = \arctan(\lambda)$ | 偏转角度 |
| $\kappa$, $\kappa^*$ | vMF 集中度 / 匹配切平面的精确值 |
| $\delta_{\text{TV}}$ | 切平面与 vMF 的全变差距离 |
| $\alpha$ | 非对称分配因子 |
| $\mathbf{M}$ | 敏感区掩码 |
| $d_{\text{geo}}$ | 测地线距离 |
| $P_\perp$ | 正交投影算子 |
| $\hat{\mathbf{n}}_\perp$ | 归一化正交噪声 |

## 附录 B 定理索引

| 编号 | 名称 | 概要 |
|---|---|---|
| 定理 4.1 | 近似度量隐私 | $(\kappa^*, 2\delta_{\text{TV}})$-近似度量隐私 |
| 引理 4.3 | TV 隐私传递 | TV 距离有界时的隐私传递 |
| 定理 5.1 | 范数保持 | $\|\mathbf{y}\| = \|\mathbf{x}\|$ |
| 定理 5.2 | 期望余弦 | $\cos\theta = 1/\sqrt{1+\lambda^2}$ |
| 定理 5.3 | 点积偏差界 | Attention 权重期望偏差有界 |
| 定理 5.4 | 高斯范数膨胀 | $d\sigma^2$ 膨胀 |
| 定理 6.1 | 双通道组合隐私 | 加法组合 |
| 定理 6.2 | 掩码隐私保持 | 掩码不削弱保护 |
| 命题 7.2 | 范数保持高斯缺陷 | 偏转依赖输入范数 |
| 命题 7.3 | vMF 效用最优性 | 约束下最大化余弦（论证） |
