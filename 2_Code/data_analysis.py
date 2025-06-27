import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

#============================ 第一张图：常数阻力模型 ============================
def calculate_constant_resistance():
    """计算常数阻力模型下的末态速度及运动特性"""
    print("\n=== 常数阻力模型计算 ===")
    
    # 定义微分方程 (阻力为常数，alpha=0)
    def model(t, v, alpha, P, m, k):
        dvdt = (P/(m*v)) - (k/m)*v**alpha
        return dvdt
    
    # 参数设置
    P = 1000.0  # 功率 (W)
    m = 100     # 质量 (kg)
    k = 50      # 常数阻力系数 (N)
    v0 = 1e-15  # 初始速度 (m/s)，避免除以零
    t_span = (0, 10)  # 时间范围 (s)
    t_eval = np.linspace(t_span[0], t_span[1], 100)  # 时间点
    
    # 定义末态速度函数
    def terminal_velocity(alpha, P, k):
        return (P/k)**(1/(alpha+1))
    
    # 计算α=0时的末态速度
    alpha = 0
    v_term = terminal_velocity(alpha, P, k)
    
    # 输出结果
    print(f"■ 模型参数: 功率={P}W, 质量={m}kg, 常数阻力={k}N")
    print(f"■ 末态速度: {v_term:.4f} m/s (当t→∞时，阻力与牵引力平衡)")
    print(f"■ 物理意义: 常数阻力模型下，列车最终速度由功率与阻力的比值决定")
    
    return v_term


#============================ 第二张图：线性阻力模型 ============================
def calculate_linear_resistance():
    """计算线性阻力模型下的末态速度及运动特性"""
    print("\n=== 线性阻力模型计算 ===")
    
    # 模型参数
    P = 1000    # 功率 (W)
    m = 100     # 质量 (kg)
    k = 50      # 线性阻力系数 (kg/s)
    
    # 计算末态速度
    v_final = np.sqrt(P / k)
    
    # 输出结果
    print(f"■ 模型参数: 功率={P}W, 质量={m}kg, 线性阻力系数={k}kg/s")
    print(f"■ 末态速度: {v_final:.4f} m/s (公式推导: v=(P/k)^0.5)")
    print(f"■ 物理意义: 线性阻力与速度成正比，高速时阻力增长较快，限制最终速度")
    
    return v_final


#============================ 第三张图：不同幂次阻力对比 ============================
def calculate_power_law_resistance():
    """计算不同幂次阻力模型下的末态速度对比"""
    print("\n=== 不同幂次阻力模型计算 ===")
    
    # 定义末态速度函数
    def terminal_velocity(alpha, P, k):
        return (P/k)**(1/(alpha+1))
    
    # 公共参数
    P = 1000.0  # 功率 (W)
    m = 100     # 质量 (kg)
    k = 50      # 阻力系数
    
    # 不同幂次值
    alphas = [0, 1, 2, 3, 4, 5]
    results = {}
    
    print(f"■ 公共参数: 功率={P}W, 质量={m}kg, 阻力系数={k}")
    print("■ 不同幂次(α)对应的末态速度:")
    
    for alpha in alphas:
        v_term = terminal_velocity(alpha, P, k)
        results[alpha] = v_term
        print(f"  - α={alpha}: {v_term:.4f} m/s")
    
    print("\n■ 规律分析:")
    print("  - 当α增大时，末态速度计算公式为 v=(P/k)^(1/(α+1))，指数衰减")
    print("  - α=0对应常数阻力，α=1对应线性阻力，α=2对应二次阻力")
    print("  - 高阶幂次阻力对高速运动的抑制作用更强，加速过程更平缓")
    
    return results


#============================ 第四张图：磁悬浮列车阻力模型 ============================
def calculate_maglev_resistance():
    """计算磁悬浮列车阻力模型（含一次项与仅二次项）"""
    print("\n=== 磁悬浮列车阻力模型计算 ===")
    
    # 物理参数
    m = 150e3    # 质量 (kg)
    P = 2e7      # 功率 (W)
    k1 = 5.90e-4 # 一次项系数 (kg/s)
    k2 = 5.86    # 二次项系数 (kg/m)
    
    # 计算末态速度的函数
    def find_final_velocity(P, k1, k2):
        # 平衡方程: k1*v² + k2*v³ = P
        def equation(v):
            return k1 * v**2 + k2 * v**3 - P
        
        v_guess = 100  # 初始猜测速度 (m/s)
        return fsolve(equation, v_guess)[0]
    
    # 计算两种情况的末态速度
    v_final_with_k1 = find_final_velocity(P, k1, k2)
    v_final_without_k1 = find_final_velocity(P, 0, k2)
    
    # 输出结果
    print(f"■ 模型参数: 质量={m/1000}吨, 功率={P/1e6}MW")
    print(f"  一次项系数={k1}kg/s, 二次项系数={k2}kg/m")
    print("\n■ 末态速度计算:")
    print(f"  - 包含一次项: {v_final_with_k1:.4f} m/s ({v_final_with_k1*3.6:.2f} km/h)")
    print(f"  - 忽略一次项: {v_final_without_k1:.4f} m/s ({v_final_without_k1*3.6:.2f} km/h)")
    print("\n■ 物理分析:")
    print("  - 一次项(k1v)代表低速粘滞阻力，二次项(k2v²)代表高速惯性阻力")
    print(f"  - 当v>1e-2 m/s时，k1v << k2v²，一次项可忽略（k1比k2小4个数量级）")
    print("  - 高速磁悬浮列车的阻力模型可简化为二次项主导")
    
    return v_final_with_k1, v_final_without_k1


#============================ 第五张图：传统列车阻力模型对比 ============================
def calculate_conventional_train():
    """计算传统列车与磁悬浮列车阻力模型对比"""
    print("\n=== 传统列车与磁悬浮列车阻力模型计算 ===")
    
    # 物理参数
    m = 150000  # 质量 (kg) = 150 吨
    P = 2e7     # 功率 (W)
    k1 = 5.90e-4  # 一次项系数 (kg/s)
    k2 = 5.86    # 二次项系数 (kg/m)
    f0_maglev = 0  # 磁悬浮列车常数阻力 (N)
    f0_train = 4000  # 传统列车常数阻力 (N)
    
    # 定义末态速度计算函数
    def terminal_velocity(f0, P, k1, k2):
        # 平衡方程: f0*v + k1*v² + k2*v³ = P
        def equation(v):
            return f0*v + k1*v**2 + k2*v**3 - P
        
        v_guess = (P / k2)**(1/3)  # 初始猜测值（忽略f0和k1）
        return fsolve(equation, v_guess)[0]
    
    # 计算两种列车的末态速度
    v_term_maglev = terminal_velocity(f0_maglev, P, k1, k2)
    v_term_train = terminal_velocity(f0_train, P, k1, k2)
    
    # 输出结果
    print(f"■ 公共参数: 质量={m/1000}吨, 功率={P/1e6}MW")
    print(f"  一次项系数={k1}kg/s, 二次项系数={k2}kg/m")
    print("\n■ 不同列车模型参数:")
    print(f"  - 磁悬浮列车: 常数阻力f0={f0_maglev}N（无接触摩擦）")
    print(f"  - 传统列车: 常数阻力f0={f0_train}N（轮轨接触摩擦）")
    print("\n■ 末态速度对比:")
    print(f"  - 磁悬浮列车: {v_term_maglev:.4f} m/s ({v_term_maglev*3.6:.2f} km/h)")
    print(f"  - 传统列车: {v_term_train:.4f} m/s ({v_term_train*3.6:.2f} km/h)")
    print(f"  - 速度差: {v_term_maglev - v_term_train:.2f} m/s "
          f"({(v_term_maglev - v_term_train)*3.6:.2f} km/h)")
    print("\n■ 物理结论:")
    print("  - 常数摩擦项(f0)对传统列车的末态速度有显著影响")
    print("  - 磁悬浮列车因消除接触摩擦，末态速度提升约",
          f"{(v_term_maglev/v_term_train-1)*100:.2f}%")
    print("  - 实际列车设计中，减少接触摩擦是提升速度的关键途径")
    
    return v_term_maglev, v_term_train


#============================ 主函数 ============================
if __name__ == "__main__":
    print("===== 列车恒功率启动动力学模拟计算 =====")
    print("(所有绘图已移除，仅保留核心计算与文字输出)\n")
    
    # 执行各模型计算
    v_constant = calculate_constant_resistance()
    v_linear = calculate_linear_resistance()
    power_law_results = calculate_power_law_resistance()
    v_maglev_with_k1, v_maglev_without_k1 = calculate_maglev_resistance()
    v_maglev, v_conventional = calculate_conventional_train()
    
    print("\n===== 综合结论 =====")
    print("1. 阻力模型对列车末态速度的影响规律:")
    print("   - 常数阻力: v∝P/f0")
    print("   - 线性阻力: v∝√(P/k1)")
    print("   - 二次阻力: v∝(P/k2)^(1/3)")
    print("   - 高阶阻力使速度增长放缓，指数衰减特性明显")
    
    print("\n2. 高速列车阻力特性:")
    print(f"   - 磁悬浮列车(无接触摩擦)末态速度: {v_maglev:.2f} m/s")
    print(f"   - 传统列车(含接触摩擦)末态速度: {v_conventional:.2f} m/s")
    print("   - 空气阻力中二次项(k2v²)在高速时起主导作用，一次项可忽略")
    
    print("\n3. 工程意义:")
    print("   - 提升列车速度的关键: ① 增大功率 ② 减小阻力")
    print("   - 磁悬浮技术通过消除接触摩擦，可显著提升极限速度")
    print("   - 高速列车设计需重点考虑空气动力学外形，减少风阻")
