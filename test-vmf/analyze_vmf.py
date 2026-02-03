"""
分析 vMF 扰动参数与角度偏差的关系
"""

import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 参数
beta = 1.0
epsilons = np.linspace(0.05, 2.0, 100)

# 计算扰动幅度
lambdas = beta / epsilons

# 理论角度偏差（近似）
# 在高维空间中，角度偏差约为 arctan(λ)
theoretical_angles = np.arctan(lambdas) * 180 / np.pi

# 实际测量数据（从你的运行结果）
measured_data = {
    0.1: 84.0,   # 估计
    0.2: 78.74,
    0.3: 72.0,   # 估计
    0.4: 68.0,   # 估计
    0.5: 63.56,
    1.0: 45.0,   # 估计
}

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 图1: epsilon vs 扰动幅度
ax1 = axes[0, 0]
ax1.plot(epsilons, lambdas, 'b-', linewidth=2)
ax1.set_xlabel('Privacy Budget (ε)', fontsize=12)
ax1.set_ylabel('Perturbation Scale (λ = β/ε)', fontsize=12)
ax1.set_title('Perturbation Scale vs Privacy Budget', fontsize=14)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=1, color='r', linestyle='--', alpha=0.5, label='λ=1')
ax1.legend()

# 标注关键点
for eps in [0.1, 0.2, 0.3, 0.5, 1.0]:
    lam = beta / eps
    ax1.scatter([eps], [lam], color='red', s=50, zorder=5)
    ax1.annotate(f'ε={eps}\nλ={lam:.1f}', (eps, lam),
                textcoords="offset points", xytext=(10, 5), fontsize=9)

# 图2: epsilon vs 理论角度偏差
ax2 = axes[0, 1]
ax2.plot(epsilons, theoretical_angles, 'g-', linewidth=2, label='Theoretical')
ax2.scatter(list(measured_data.keys()), list(measured_data.values()),
           color='red', s=100, zorder=5, label='Measured')
ax2.set_xlabel('Privacy Budget (ε)', fontsize=12)
ax2.set_ylabel('Angular Deviation (degrees)', fontsize=12)
ax2.set_title('Angular Deviation vs Privacy Budget', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim(0, 90)

# 图3: 扰动幅度变化率
ax3 = axes[1, 0]
eps_range = np.linspace(0.1, 1.0, 50)
lambda_change_rate = -beta / (eps_range ** 2)  # dλ/dε 的导数
ax3.plot(eps_range, np.abs(lambda_change_rate), 'purple', linewidth=2)
ax3.set_xlabel('Privacy Budget (ε)', fontsize=12)
ax3.set_ylabel('|dλ/dε| (Rate of Change)', fontsize=12)
ax3.set_title('Sensitivity: How Fast λ Changes with ε', fontsize=14)
ax3.grid(True, alpha=0.3)

# 标注敏感区域
ax3.axvspan(0.1, 0.3, alpha=0.3, color='red', label='High Sensitivity Zone')
ax3.axvspan(0.3, 0.5, alpha=0.3, color='yellow', label='Medium Sensitivity')
ax3.axvspan(0.5, 1.0, alpha=0.3, color='green', label='Low Sensitivity')
ax3.legend()

# 图4: 不同 epsilon 区间的 λ 变化
ax4 = axes[1, 1]
intervals = [
    ('0.1→0.2', 0.1, 0.2),
    ('0.2→0.3', 0.2, 0.3),
    ('0.3→0.4', 0.3, 0.4),
    ('0.4→0.5', 0.4, 0.5),
    ('0.5→1.0', 0.5, 1.0),
]

labels = []
changes = []
for label, e1, e2 in intervals:
    l1, l2 = beta/e1, beta/e2
    change = l1 - l2
    labels.append(label)
    changes.append(change)

colors = ['red' if c > 2 else 'orange' if c > 1 else 'green' for c in changes]
bars = ax4.bar(labels, changes, color=colors)
ax4.set_xlabel('Epsilon Interval', fontsize=12)
ax4.set_ylabel('Δλ (Change in Perturbation Scale)', fontsize=12)
ax4.set_title('Perturbation Scale Change per Interval', fontsize=14)
ax4.grid(True, alpha=0.3, axis='y')

# 在柱子上标注数值
for bar, change in zip(bars, changes):
    ax4.annotate(f'{change:.2f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('vmf_analysis.png', dpi=150, bbox_inches='tight')
print("图表已保存: vmf_analysis.png")

# 打印分析结果
print("\n" + "=" * 60)
print("vMF 扰动参数分析")
print("=" * 60)

print("\n1. 扰动幅度公式: λ = β / ε")
print(f"   当前 β = {beta}")

print("\n2. 各 epsilon 值对应的扰动幅度:")
for eps in [0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0]:
    lam = beta / eps
    angle = np.arctan(lam) * 180 / np.pi
    print(f"   ε={eps:.2f} → λ={lam:.2f} → 理论角度≈{angle:.1f}°")

print("\n3. 为什么 ε=0.2 和 ε=0.3 差距大？")
print(f"   ε=0.2 → λ={beta/0.2:.2f}")
print(f"   ε=0.3 → λ={beta/0.3:.2f}")
print(f"   Δλ = {beta/0.2 - beta/0.3:.2f}")
print("   这个区间的 λ 变化了 1.67，是较大的变化！")

print("\n4. 建议:")
print("   - 如果需要强隐私保护，使用 ε ∈ [0.1, 0.3]")
print("   - 如果需要平衡隐私和效用，使用 ε ∈ [0.3, 0.5]")
print("   - 如果主要关注效用，使用 ε ∈ [0.5, 1.0]")

plt.show()
