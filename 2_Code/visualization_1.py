import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 定义微分方程
def model(t, v, alpha, P, m, k):
    dvdt = (P/(m*v)) - (k/m)*v**alpha
    return dvdt

# 参数设置
P = 1000.0  # 功率
m = 100     # 质量
k = 50      # 阻力系数
v0 = 1e-15  # 初始速度 (避免除以零)
t_span = (0, 10)  # 时间范围
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 时间点

# 定义末态速度函数
def terminal_velocity(alpha, P, k):
    return (P/k)**(1/(alpha+1))

# 计算α=0时的解
alpha = 0
sol = solve_ivp(model, t_span, [v0], args=(alpha, P, m, k), t_eval=t_eval, method='RK45')
v_term = terminal_velocity(alpha, P, k)

# 创建单独的α=0图形
plt.figure(figsize=(12, 8))

# 绘制v-t曲线
plt.plot(sol.t, sol.y[0], 'b-', linewidth=2, label=f'α={alpha} 速度曲线')

# 绘制末态速度上限线
plt.axhline(y=v_term, color='r', linestyle='--', linewidth=2, alpha=0.7, 
            label=f'末态速度上限: {v_term:.2f} m/s')

# 添加标记和文本
plt.text(5, v_term*1.02, f'末态速度 = {v_term:.2f} m/s', fontsize=12, color='red')

# 设置图形属性
plt.title(f'阻力指数 α={alpha} 时的速度-时间曲线', fontsize=16)
plt.xlabel('时间 (s)', fontsize=14)
plt.ylabel('速度 (m/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.ylim(0, v_term*1.1)  # 设置y轴上限为末态速度的1.1倍

# 添加物理参数信息
param_text = f'功率 P = {P} W\n质量 m = {m} kg\n阻力系数 k = {k}'
plt.figtext(0.15, 0.75, param_text, bbox=dict(facecolor='white', alpha=0.5), fontsize=12)

plt.tight_layout()
plt.savefig('alpha0_curve.png', dpi=300)
plt.show()
