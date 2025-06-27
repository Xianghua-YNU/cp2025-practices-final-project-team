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
k = 50     # 阻力系数
v0 = 1e-15   # 初始速度 (避免除以零)
t_span = (0, 2)  # 时间范围
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 时间点

# 定义末态速度函数
def terminal_velocity(alpha, P, k):
    return (P/k)**(1/(alpha+1))

# 颜色和线型
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
alphas_full = [0, 1, 2, 3, 4, 5]
alphas_partial = [2, 3, 4, 5]

# 图1: α=2,3,4,5的v-t曲线与末态速度
plt.figure(figsize=(12, 8))
for i, alpha in enumerate(alphas_partial):
    # 数值求解微分方程
    sol = solve_ivp(model, t_span, [v0], args=(alpha, P, m, k), t_eval=t_eval, method='RK45')
    
    # 计算末态速度
    v_term = terminal_velocity(alpha, P, k)
    
    # 绘制v-t曲线
    plt.plot(sol.t, sol.y[0], color=colors[i], label=f'α={alpha}')
    
    # 绘制末态速度线
    plt.axhline(y=v_term, color=colors[i], linestyle='--', alpha=0.7)

plt.title('  ')
plt.xlabel('时间 (s)')
plt.ylabel('速度 (m/s)')
plt.grid(True)
plt.legend()
plt.ylim(0, 4)
plt.savefig('figure1.png')
plt.show()

# 图2: α=0-5的v-t曲线对比
plt.figure(figsize=(12, 8))
for i, alpha in enumerate(alphas_full):
    # 数值求解微分方程
    sol = solve_ivp(model, t_span, [v0], args=(alpha, P, m, k), t_eval=t_eval, method='RK45')
    
    # 计算末态速度
    v_term = terminal_velocity(alpha, P, k)
    
    # 绘制v-t曲线
    plt.plot(sol.t, sol.y[0], color=colors[i], label=f'α={alpha}')
    
    # 绘制末态速度线（透明）
    plt.axhline(y=v_term, color=colors[i], linestyle='--', alpha=0.3)

plt.title('Comparison of Velocity Curves for Different α Values')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()
plt.ylim(0,21)
plt.savefig('figure2.png')
plt.show()
