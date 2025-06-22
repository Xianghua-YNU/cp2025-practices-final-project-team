import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#----------------------------------------第一张图------------------------------------

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




#------------------------------------第二张图---------------------------------------------------------
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
#--------------------------------求解方程-------------------------------------------
# 定义已知参数
k1 = 5.90e-4  # 一次项系数，单位：kg/s
k2 = 5.86     # 二次项系数，单位：kg/m
P = 2e7       # 功率，单位：W

# 定义平衡时的方程：k1*v^2 + k2*v^3 - P = 0
def equation(v):
    return k1 * v**2 + k2 * v**3 - P

# 提供一个合理的初始猜测值，考虑到高速列车的速度范围
initial_guess = 300  # 初始猜测速度为300 m/s

# 使用fsolve函数求解方程
final_velocity = fsolve(equation, initial_guess)[0]

# 输出结果
print(f"列车的末态速度为: {final_velocity:.2f} m/s")
#---------------------------------------第三张图-------------------------------------------------------
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


for i, alpha in enumerate(alphas_partial):
    # 数值求解微分方程
    sol = solve_ivp(model, t_span, [v0], args=(alpha, P, m, k), t_eval=t_eval, method='RK45')
    
    # 计算末态速度
    v_term = terminal_velocity(alpha, P, k)
#---------------------------------第四张图--------------------------------------
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


#--------------------------第五张图-----------------------------------------------



# Physical parameters setup
m = 150000  # Mass (kg) = 150 tons
P = 2e7     # Power (W)
k1 = 5.90e-4  # Linear resistance coefficient (kg/s)
k2 = 5.86    # Quadratic resistance coefficient (kg/m)
f0_maglev = 0  # Constant resistance for maglev train (N)
f0_train = 4000  # Constant resistance for conventional train (N)

# Define the differential equation
def train_model(v, f0):
    """Train motion differential equation"""
    if v < 1e-6:  # Avoid division by zero
        return (P - f0) / m
    return (P/v - f0 - k1*v - k2*v**2) / m

# 4th-order Runge-Kutta solver
def runge_kutta4(f, v0, t_start, t_end, dt, f0):
    """4th-order Runge-Kutta method for solving differential equations"""
    num_steps = int((t_end - t_start) / dt)
    t = np.zeros(num_steps)
    v = np.zeros(num_steps)
    
    t[0] = t_start
    v[0] = v0
    
    for i in range(num_steps - 1):
        k1_val = dt * f(v[i], f0)
        k2_val = dt * f(v[i] + k1_val/2, f0)
        k3_val = dt * f(v[i] + k2_val/2, f0)
        k4_val = dt * f(v[i] + k3_val, f0)
        
        v[i+1] = v[i] + (k1_val + 2*k2_val + 2*k3_val + k4_val)/6
        t[i+1] = t[i] + dt
        
    return t, v

# Calculate terminal velocity (steady-state solution)
def terminal_velocity(f0, P, k1, k2):
    """Calculate terminal velocity (when drag equals thrust)"""
    def equation(v):
        return f0*v + k1*v**2 + k2*v**3 - P
    
    # Solve the cubic equation numerically
    # Initial guess (approximation ignoring f0 and k1)
    v_guess = (P / k2)**(1/3)
    v_term = fsolve(equation, v_guess)[0]
    return v_term

# Calculate terminal velocities for both cases
v_term_maglev = terminal_velocity(f0_maglev, P, k1, k2)
v_term_train = terminal_velocity(f0_train, P, k1, k2)

print(f"Maglev terminal velocity: {v_term_maglev:.2f} m/s ({v_term_maglev*3.6:.2f} km/h)")
print(f"Conventional train terminal velocity: {v_term_train:.2f} m/s ({v_term_train*3.6:.2f} km/h)")

# Solve the differential equations numerically
t_start = 0      # Start time (s)
t_end = 1000     # End time (s)
dt = 0.1         # Time step (s)
v0 = 1e-19         # Initial velocity (m/s) - avoid division by zero

# Solve for maglev train (f0=0)
t_maglev, v_maglev = runge_kutta4(train_model, v0, t_start, t_end, dt, f0_maglev)

# Solve for conventional train (f0=4000)
t_train, v_train = runge_kutta4(train_model, v0, t_start, t_end, dt, f0_train)
