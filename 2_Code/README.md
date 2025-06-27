# 列车恒功率启动动力学模拟代码说明

## 项目概述

本项目通过数值模拟方法研究列车在恒定功率启动条件下，不同阻力模型（常数阻力、线性阻力、幂次阻力及幂级数阻力）下的运动学特性。代码实现了4阶龙格-库塔法求解运动微分方程，并通过可视化展示不同阻力模型下的速度-时间曲线及末态速度特性。

## 目录结构

```
2_code/
├── numerical_methods.py       # 数值求解核心算法
├── data_analysis.py           # 数据分析与参数计算
├── visualization_1.py         # 常数阻力模型可视化（Fig.1）
├── visualization_2.py         # 线性阻力模型可视化（Fig.2）
├── visualization_3.py         # 幂次阻力模型对比可视化（Fig.3）
├── visualization_4.py         # 磁悬浮列车阻力模型可视化（Fig.4）
├── visualization_5.py         # 传统列车阻力模型对比可视化（Fig.5）
├── requirements.txt           # 环境依赖
└── README.md                  # 本说明文档
```

## 环境配置

### 依赖项

```text
numpy>=1.21.0       # 数值计算
scipy>=1.7.0        # 科学计算
matplotlib>=3.4.0   # 数据可视化
```

### 安装方法

通过pip安装依赖：
```bash
pip install -r requirements.txt
```

## 代码功能说明

### 1. numerical_methods.py

**功能：** 实现4阶龙格-库塔数值求解器，用于求解非线性运动微分方程。

**核心函数：**
- `runge_kutta_4th(func, t_span, y0, h)`: 四阶龙格-库塔法主函数
  - `func`: 微分方程右端函数
  - `t_span`: 时间区间 [t_start, t_end]
  - `y0`: 初始条件 [v0]
  - `h`: 时间步长
- `resistance_models(v, alpha=0, f0=0, k1=0, k2=0)`: 阻力模型定义
  - `alpha`: 幂次阻力指数（α=0为常数阻力，α=1为线性阻力）
  - `f0`: 常数阻力项
  - `k1`: 线性阻力系数
  - `k2`: 二次阻力系数

**示例用法：**
```python
from numerical_methods import runge_kutta_4th, resistance_models

# 定义微分方程: dv/dt = (P - f(v)*v) / (m*v)
def differential_eq(t, v, P, m, resistance_params):
    f = resistance_models(v, **resistance_params)
    return (P - f * v) / (m * v)

# 求解常数阻力模型
solution = runge_kutta_4th(
    lambda t, v: differential_eq(t, v, P=2e7, m=1.5e5, resistance_params={"f0": 4000}),
    t_span=[0, 1000],
    y0=[0],
    h=0.1
)
```

### 2. data_analysis.py

**功能：** 数据分析与末态速度计算，包含：
- 解析解计算（常数阻力、线性阻力）
- 非线性方程数值求解（幂次阻力、幂级数阻力）
- 特征速度分析（如粘滞阻力与惯性阻力平衡点）

**核心函数：**
- `analytical_solution_constant(P, f0, m, t)`: 常数阻力解析解（公式7）
- `analytical_solution_linear(P, k1, m, t)`: 线性阻力解析解（公式14）
- `solve_terminal_velocity(P, f0, k1, k2, alpha=2)`: 末态速度数值求解
- `critical_velocity(k1, k2)`: 粘滞阻力与惯性阻力平衡点计算

### 3. 可视化脚本

#### visualization_1.py
**功能：** 绘制常数阻力模型下的v-t曲线（Fig.1）
- 包含解析解曲线与末态速度渐近线
- 参数：`P=2e7 W`, `m=1.5e5 kg`, `f0=4000 N`

#### visualization_2.py
**功能：** 绘制线性阻力模型下的v-t曲线（Fig.2）
- 包含解析解曲线与末态速度渐近线
- 参数：`P=2e7 W`, `m=1.5e5 kg`, `k1=5.90e-4 kg/s`

#### visualization_3.py
**功能：** 不同幂次阻力模型对比（Fig.3）
- 绘制α=0,1,2,3,4,5时的v-t曲线
- 末态速度用透明直线标注
- 展示α对加速过程的影响

#### visualization_4.py
**功能：** 磁悬浮列车阻力模型可视化（Fig.4）
- 对比包含一次项与仅二次项的阻力模型
- 参数：`f0=0`, `k1=5.90e-4 kg/s`, `k2=5.86 kg/m`
- 展示低速阶段一次项影响与高速阶段二次项主导

#### visualization_5.py
**功能：** 传统列车阻力模型对比（Fig.5）
- 对比包含常数摩擦项与忽略常数项的模型
- 参数：`f0=4000 N`, `k1=5.90e-4 kg/s`, `k2=5.86 kg/m`
- 展示常数摩擦项对末态速度的影响

## 运行方法

### 1. 运行单个模型可视化

```bash
# 运行常数阻力模型可视化
python visualization_1.py

# 运行磁悬浮列车阻力模型可视化
python visualization_4.py
```

### 2. 批量运行所有可视化（建议分步执行）

```bash
python visualization_1.py
python visualization_2.py
python visualization_3.py
python visualization_4.py
python visualization_5.py
```

### 3. 自定义参数运行

如需修改模型参数，可直接编辑可视化脚本中的参数定义部分，例如在`visualization_4.py`中：

```python
# 磁悬浮列车参数
m = 1.5e5          # 质量(kg)
P = 2.0e7          # 功率(W)
f0 = 0             # 常数阻力(N)
k1 = 5.90e-4       # 一次项系数(kg/s)
k2 = 5.86          # 二次项系数(kg/m)
```

## 结果说明

运行可视化脚本后，将生成5幅图像，分别对应论文中的Fig.1至Fig.5：
- Fig.1: 常数阻力模型下的速度-时间曲线
- Fig.2: 线性阻力模型下的速度-时间曲线
- Fig.3: 不同幂次阻力指数(α)的速度曲线对比
- Fig.4: 磁悬浮列车阻力模型（含一次项与仅二次项）对比
- Fig.5: 传统列车（含常数摩擦项）与理想模型对比

所有图像将保存在当前工作目录，默认格式为PNG，包含清晰的图例、坐标轴标签及物理量单位。

## 理论与代码对应关系

| 论文章节 | 代码文件 | 实现方法 |
|----------|----------|----------|
| 阻力为常数 | visualization_1.py | 解析解（公式7） |
| 阻力与速度成正比 | visualization_2.py | 解析解（公式14） |
| 阻力与速度成其他幂次 | visualization_3.py | 4阶龙格-库塔法 |
| 阻力与速度成幂级数 | visualization_4.py <br> visualization_5.py | 4阶龙格-库塔法 |

## 注意事项

1. 数值求解时时间步长`h`可在`numerical_methods.py`中调整，较小的步长可提高精度但增加计算量
2. 幂级数阻力模型中的末态速度通过`data_analysis.solve_terminal_velocity`函数数值求解
3. 磁悬浮列车与传统列车的参数设置可在对应可视化脚本中修改
4. 若需计算高阶阻力项影响，可在`resistance_models`函数中扩展幂级数项
