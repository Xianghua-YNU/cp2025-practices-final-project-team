import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 定义参数
P = 1000  # 功率
m = 100   # 质量
k = 50    # 阻力系数

# 计算末态速度
v_final = np.sqrt(P / k)

# 生成时间数组
t = np.linspace(0, 15, 500)

# 根据方程 v = sqrt((P/k)*(1-exp(-2*k*t/m))) 计算对应的v值
v = np.sqrt((P / k) * (1 - np.exp(-2 * k * t / m)))

# 创建图形和坐标轴
fig, ax = plt.subplots(figsize=(10, 6))

# 绘制v-t曲线
ax.plot(t, v, 'b-', linewidth=2.5, label=f'速度-时间曲线 v(t)')

# 绘制末态速度水平线
ax.axhline(y=v_final, color='r', linestyle='--', linewidth=2, label=f'末态速度 v={v_final:.2f}')

# 格式化坐标轴
ax.set_xlabel('时间 t (s)', fontsize=14)
ax.set_ylabel('速度 v (m/s)', fontsize=14)
ax.set_title('阻力与速度成正比时的速度-时间图像', fontsize=16, fontweight='bold')

# 添加网格线
ax.grid(True, linestyle='--', alpha=0.7)

# 添加图例
ax.legend(fontsize=12, loc='best')

# 装饰图形
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# 显示图形
plt.tight_layout()
plt.show()

# 另外绘制一个放大的起始阶段图像，展示初始加速过程
t_initial = t[t < 5]  # 选择t小于5的部分
v_initial = v[:len(t_initial)]

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(t_initial, v_initial, 'b-', linewidth=2.5, label=f'速度-时间曲线 v(t)')
ax.axhline(y=v_final, color='r', linestyle='--', linewidth=2, label=f'末态速度 v={v_final:.2f}')

ax.set_xlabel('时间 t (s)', fontsize=14)
ax.set_ylabel('速度 v (m/s)', fontsize=14)
ax.set_title('初始阶段速度-时间图像放大', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(fontsize=12, loc='best')

for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
plt.show()
