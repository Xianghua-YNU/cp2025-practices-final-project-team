import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 定义物理参数
m = 150e3    # 列车质量，单位：kg
P = 2e7      # 功率，单位：W
k1 = 5.90e-4 # 一次项系数，单位：kg/s
k2 = 5.86    # 二次项系数，单位：kg/m

# 定义微分方程 (25)：du/dt = 2P/m - (2k2/m)u^(3/2) - (2k1/m)u
def dudt(t, u, k1_val=k1):
    # 确保u为非负值
    u = max(u, 0)
    return (2*P/m) - (2*k2/m)*u**(3/2) - (2*k1_val/m)*u

# 定义时间范围
t_span = (0, 3000)  # 0到3000秒
t_eval = np.linspace(t_span[0], t_span[1], 1000)  # 时间点

# 初始条件：u(0) = 0
u0 = [0]

# 使用solve_ivp求解微分方程（包含一次项）
solution_with_k1 = solve_ivp(
    dudt, t_span, u0, method='RK45', t_eval=t_eval, args=(k1,)
)

# 求解忽略一次项的情况
solution_without_k1 = solve_ivp(
    dudt, t_span, u0, method='RK45', t_eval=t_eval, args=(0,)
)

# 计算速度 v = sqrt(u)
v_with_k1 = np.sqrt(solution_with_k1.y[0])
v_without_k1 = np.sqrt(solution_without_k1.y[0])

# 计算理论末态速度
def find_final_velocity(P, k1, k2):
    # 定义方程：k1*v^2 + k2*v^3 - P = 0
    def equation(v):
        return k1 * v**2 + k2 * v**3 - P
    
    # 使用数值方法求解
    from scipy.optimize import fsolve
    return fsolve(equation, 100)[0]

v_final_with_k1 = find_final_velocity(P, k1, k2)
v_final_without_k1 = find_final_velocity(P, 0, k2)

print(f"包含一次项的末态速度: {v_final_with_k1:.2f} m/s")
print(f"忽略一次项的末态速度: {v_final_without_k1:.2f} m/s")

# 绘制v-t曲线
plt.figure(figsize=(12, 8))

# 绘制包含一次项的曲线
plt.plot(solution_with_k1.t, v_with_k1, 'b-', linewidth=2, 
         label=f'包含一次项 (k1={k1:.2e})')

# 绘制忽略一次项的曲线
plt.plot(solution_without_k1.t, v_without_k1, 'r--', linewidth=2, 
         label='忽略一次项 (k1=0)')

# 绘制末态速度水平线
plt.axhline(y=v_final_with_k1, color='b', linestyle=':', 
            label=f'理论末态速度 (含k1): {v_final_with_k1:.2f} m/s')
plt.axhline(y=v_final_without_k1, color='r', linestyle=':', 
            label=f'理论末态速度 (不含k1): {v_final_without_k1:.2f} m/s')

# 设置图表属性
plt.title('高速列车速度-时间曲线', fontsize=16)
plt.xlabel('时间 t (s)', fontsize=14)
plt.ylabel('速度 v (m/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# 调整x轴范围以便更好地观察
plt.xlim(0, 2000)  # 只显示前2000秒的结果

# 添加额外的标记线，显示速度差异
plt.tight_layout()
plt.savefig('train_velocity_curve.png', dpi=300)
plt.show()

# 绘制放大的起始部分
plt.figure(figsize=(12, 6))
plt.plot(solution_with_k1.t, v_with_k1, 'b-', linewidth=2, 
         label=f'包含一次项 (k1={k1:.2e})')
plt.plot(solution_without_k1.t, v_without_k1, 'r--', linewidth=2, 
         label='忽略一次项 (k1=0)')
plt.title('列车启动阶段速度对比', fontsize=16)
plt.xlabel('时间 t (s)', fontsize=14)
plt.ylabel('速度 v (m/s)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)
plt.xlim(0, 100)  # 只显示前100秒的结果
plt.ylim(0, 100)  # 调整y轴范围
plt.tight_layout()
plt.savefig('train_startup_comparison.png', dpi=300)
plt.show()

