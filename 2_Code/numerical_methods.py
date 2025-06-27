# numerical_methods.py - 列车动力学模拟数值计算核心模块
# 包含龙格-库塔法、微分方程求解、末态速度计算等核心算法

import numpy as np
from scipy.optimize import fsolve
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)  # 忽略数值计算警告


class ResistanceModels:
    """阻力模型定义类，包含不同类型的阻力函数"""
    
    @staticmethod
    def constant_resistance(v, f0):
        """常数阻力模型: f(v) = f0"""
        return f0
    
    @staticmethod
    def linear_resistance(v, k1):
        """线性阻力模型: f(v) = k1*v"""
        return k1 * v
    
    @staticmethod
    def quadratic_resistance(v, k2):
        """二次阻力模型: f(v) = k2*v^2"""
        return k2 * v**2
    
    @staticmethod
    def power_law_resistance(v, alpha, k):
        """幂次阻力模型: f(v) = k*v^alpha"""
        return k * v**alpha
    
    @staticmethod
    def maglev_resistance(v, k1, k2):
        """磁悬浮列车阻力模型: f(v) = k1*v + k2*v^2"""
        return k1 * v + k2 * v**2
    
    @staticmethod
    def conventional_resistance(v, f0, k1, k2):
        """传统列车阻力模型: f(v) = f0 + k1*v + k2*v^2"""
        return f0 + k1 * v + k2 * v**2


class Solver:
    """数值求解器类，包含龙格-库塔法和微分方程求解"""
    
    @staticmethod
    def runge_kutta_4th(order, f, y0, t_span, dt, *args):
        """
        四阶龙格-库塔法求解常微分方程
        
        参数:
            order: 微分方程阶数
            f: 微分方程函数 dy/dt = f(t, y, *args)
            y0: 初始条件 [y1(0), y2(0), ..., yn(0)]
            t_span: 时间区间 (t_start, t_end)
            dt: 时间步长
            *args: 微分方程额外参数
        
        返回:
            t: 时间数组
            y: 解数组 [y1(t), y2(t), ..., yn(t)]
        """
        t_start, t_end = t_span
        num_steps = int((t_end - t_start) / dt)
        t = np.zeros(num_steps)
        y = np.zeros((order, num_steps))
        
        t[0] = t_start
        y[:, 0] = y0
        
        for i in range(num_steps - 1):
            k1 = dt * f(t[i], y[:, i], *args)
            k2 = dt * f(t[i] + dt/2, y[:, i] + k1/2, *args)
            k3 = dt * f(t[i] + dt/2, y[:, i] + k2/2, *args)
            k4 = dt * f(t[i] + dt, y[:, i] + k3, *args)
            
            y[:, i+1] = y[:, i] + (k1 + 2*k2 + 2*k3 + k4) / 6
            t[i+1] = t[i] + dt
            
        return t, y
    
    @staticmethod
    def solve_ivp(f, t_span, y0, method='RK45', t_eval=None, *args):
        """
        封装scipy的solve_ivp函数求解初值问题
        
        参数:
            f: 微分方程函数 dy/dt = f(t, y, *args)
            t_span: 时间区间 (t_start, t_end)
            y0: 初始条件
            method: 求解方法，默认RK45
            t_eval: 计算解的时间点
            *args: 微分方程额外参数
        
        返回:
            t: 时间数组
            y: 解数组
        """
        from scipy.integrate import solve_ivp
        solution = solve_ivp(f, t_span, y0, method=method, t_eval=t_eval, args=args)
        return solution.t, solution.y


class TerminalVelocity:
    """末态速度计算类，包含不同阻力模型的末态速度求解"""
    
    @staticmethod
    def constant_resistance(P, f0):
        """常数阻力模型末态速度: v = P/f0"""
        return P / f0
    
    @staticmethod
    def linear_resistance(P, k1):
        """线性阻力模型末态速度: v = sqrt(P/k1)"""
        return np.sqrt(P / k1)
    
    @staticmethod
    def power_law_resistance(P, alpha, k):
        """幂次阻力模型末态速度: v = (P/k)^(1/(alpha+1))"""
        return (P / k) **(1 / (alpha + 1))
    
    @staticmethod
    def maglev_resistance(P, k1, k2):
        """磁悬浮列车末态速度: 求解k1*v^2 + k2*v^3 = P"""
        def equation(v):
            return k1 * v**2 + k2 * v**3 - P
        v_guess = (P / k2)**(1/3)  # 初始猜测值
        return fsolve(equation, v_guess)[0]
    
    @staticmethod
    def conventional_resistance(P, f0, k1, k2):
        """传统列车末态速度: 求解f0*v + k1*v^2 + k2*v^3 = P"""
        def equation(v):
            return f0 * v + k1 * v**2 + k2 * v**3 - P
        v_guess = (P / k2)**(1/3)  # 初始猜测值（忽略f0和k1）
        return fsolve(equation, v_guess)[0]


class TrainDynamics:
    """列车动力学模型类，整合阻力模型和求解器"""
    
    def __init__(self, mass, power):
        """
        初始化列车动力学模型
        
        参数:
            mass: 列车质量(kg)
            power: 功率(W)
        """
        self.mass = mass
        self.power = power
        self.resistance = ResistanceModels()
        self.solver = Solver()
        self.terminal_velocity = TerminalVelocity()
    
    def differential_equation(self, t, v, resistance_func, *resistance_args):
        """
        列车运动微分方程: m*v*dv/dt = P - f(v)*v
        
        参数:
            t: 时间
            v: 速度
            resistance_func: 阻力函数
            *resistance_args: 阻力函数参数
            
        返回:
            dv/dt: 加速度
        """
        f_v = resistance_func(v, *resistance_args)
        if v < 1e-10:  # 避免除以零
            return self.power / self.mass
        return (self.power / v - f_v) / self.mass
    
    def solve_constant_resistance(self, f0, t_span, dt=0.1):
        """
        求解常数阻力模型
        
        参数:
            f0: 常数阻力(N)
            t_span: 时间区间(t_start, t_end)
            dt: 时间步长
            
        返回:
            t: 时间数组
            v: 速度数组
            v_term: 末态速度
        """
        # 定义微分方程
        def f(t, v):
            return self.differential_equation(t, v, self.resistance.constant_resistance, f0)
        
        # 初始条件
        v0 = 1e-15  # 避免除以零
        
        # 求解微分方程
        t, v = self.solver.runge_kutta_4th(1, f, [v0], t_span, dt)
        
        # 计算末态速度
        v_term = self.terminal_velocity.constant_resistance(self.power, f0)
        
        return t, v[0], v_term
    
    def solve_linear_resistance(self, k1, t_span, dt=0.1):
        """
        求解线性阻力模型
        
        参数:
            k1: 线性阻力系数(kg/s)
            t_span: 时间区间(t_start, t_end)
            dt: 时间步长
            
        返回:
            t: 时间数组
            v: 速度数组
            v_term: 末态速度
        """
        # 定义微分方程
        def f(t, v):
            return self.differential_equation(t, v, self.resistance.linear_resistance, k1)
        
        # 初始条件
        v0 = 1e-15  # 避免除以零
        
        # 求解微分方程
        t, v = self.solver.runge_kutta_4th(1, f, [v0], t_span, dt)
        
        # 计算末态速度
        v_term = self.terminal_velocity.linear_resistance(self.power, k1)
        
        return t, v[0], v_term
    
    def solve_power_law_resistance(self, alpha, k, t_span, dt=0.1):
        """
        求解幂次阻力模型
        
        参数:
            alpha: 幂次
            k: 阻力系数
            t_span: 时间区间(t_start, t_end)
            dt: 时间步长
            
        返回:
            t: 时间数组
            v: 速度数组
            v_term: 末态速度
        """
        # 定义微分方程
        def f(t, v):
            return self.differential_equation(t, v, self.resistance.power_law_resistance, alpha, k)
        
        # 初始条件
        v0 = 1e-15  # 避免除以零
        
        # 求解微分方程
        t, v = self.solver.runge_kutta_4th(1, f, [v0], t_span, dt)
        
        # 计算末态速度
        v_term = self.terminal_velocity.power_law_resistance(self.power, alpha, k)
        
        return t, v[0], v_term
    
    def solve_maglev_resistance(self, k1, k2, t_span, dt=0.1):
        """
        求解磁悬浮列车阻力模型
        
        参数:
            k1: 一次项系数(kg/s)
            k2: 二次项系数(kg/m)
            t_span: 时间区间(t_start, t_end)
            dt: 时间步长
            
        返回:
            t: 时间数组
            v: 速度数组
            v_term_with_k1: 包含一次项的末态速度
            v_term_without_k1: 忽略一次项的末态速度
        """
        # 定义微分方程（包含一次项）
        def f_with_k1(t, v):
            return self.differential_equation(t, v, self.resistance.maglev_resistance, k1, k2)
        
        # 定义微分方程（忽略一次项）
        def f_without_k1(t, v):
            return self.differential_equation(t, v, self.resistance.quadratic_resistance, k2)
        
        # 初始条件
        v0 = 1e-15  # 避免除以零
        
        # 求解微分方程（包含一次项）
        t, v_with_k1 = self.solver.runge_kutta_4th(1, f_with_k1, [v0], t_span, dt)
        
        # 求解微分方程（忽略一次项）
        _, v_without_k1 = self.solver.runge_kutta_4th(1, f_without_k1, [v0], t_span, dt)
        
        # 计算末态速度
        v_term_with_k1 = self.terminal_velocity.maglev_resistance(self.power, k1, k2)
        v_term_without_k1 = self.terminal_velocity.power_law_resistance(self.power, 2, k2)
        
        return t, v_with_k1[0], v_without_k1[0], v_term_with_k1, v_term_without_k1
    
    def solve_conventional_resistance(self, f0_maglev, f0_train, k1, k2, t_span, dt=0.1):
        """
        求解传统列车与磁悬浮列车阻力模型对比
        
        参数:
            f0_maglev: 磁悬浮列车常数阻力(N)
            f0_train: 传统列车常数阻力(N)
            k1: 一次项系数(kg/s)
            k2: 二次项系数(kg/m)
            t_span: 时间区间(t_start, t_end)
            dt: 时间步长
            
        返回:
            t: 时间数组
            v_maglev: 磁悬浮列车速度数组
            v_train: 传统列车速度数组
            v_term_maglev: 磁悬浮列车末态速度
            v_term_train: 传统列车末态速度
        """
        # 定义磁悬浮列车微分方程(f0=0)
        def f_maglev(t, v):
            return self.differential_equation(t, v, self.resistance.maglev_resistance, k1, k2)
        
        # 定义传统列车微分方程(f0=4000)
        def f_train(t, v):
            return self.differential_equation(t, v, self.resistance.conventional_resistance, 
                                             f0_train, k1, k2)
        
        # 初始条件
        v0 = 1e-15  # 避免除以零
        
        # 求解磁悬浮列车微分方程
        t, v_maglev = self.solver.runge_kutta_4th(1, f_maglev, [v0], t_span, dt)
        
        # 求解传统列车微分方程
        _, v_train = self.solver.runge_kutta_4th(1, f_train, [v0], t_span, dt)
        
        # 计算末态速度
        v_term_maglev = self.terminal_velocity.maglev_resistance(self.power, k1, k2)
        v_term_train = self.terminal_velocity.conventional_resistance(self.power, f0_train, k1, k2)
        
        return t, v_maglev[0], v_train[0], v_term_maglev, v_term_train


# 示例用法
if __name__ == "__main__":
    # 创建列车动力学模型 (质量100kg, 功率1000W)
    train = TrainDynamics(mass=100, power=1000)
    
    # 示例1: 常数阻力模型计算
    print("=== 常数阻力模型计算 ===")
    f0 = 50  # 常数阻力(N)
    t_span = (0, 10)
    t, v, v_term = train.solve_constant_resistance(f0, t_span)
    print(f"■ 模型参数: 功率={train.power}W, 质量={train.mass}kg, 常数阻力={f0}N")
    print(f"■ 末态速度: {v_term:.4f} m/s")
    
    # 示例2: 线性阻力模型计算
    print("\n=== 线性阻力模型计算 ===")
    k1 = 50  # 线性阻力系数(kg/s)
    t_span = (0, 10)
    t, v, v_term = train.solve_linear_resistance(k1, t_span)
    print(f"■ 模型参数: 功率={train.power}W, 质量={train.mass}kg, 线性阻力系数={k1}kg/s")
    print(f"■ 末态速度: {v_term:.4f} m/s")
