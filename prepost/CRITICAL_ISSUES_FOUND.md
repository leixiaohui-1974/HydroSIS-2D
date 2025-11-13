# HydroSIS-2D 严重问题报告

**日期**: 2025-11-12  
**分析者**: 深度代码审查  
**严重程度**: 🔴 **高危**

---

## 执行摘要

深度算法分析揭示了**多个严重的数值问题**，导致：
- ❌ 干湿边界处理失败（NaN/溢出）
- ❌ 质量守恒误差大（>1%）
- ❌ 静水平衡不能保持（Well-Balanced失败）
- ❌ 数值收敛性差

**结论**: **当前代码不适合生产使用**，需要修复关键bug。

---

## 🔴 问题1: 源项处理不Well-Balanced (最严重)

### 位置
`shallow_water_solver.py`, 第833-834行

### 当前实现
```python
S_hu = -g * self.h * dz_dx
S_hv = -g * self.h * dz_dy
```

### 问题分析
**这是错误的！**  

对于浅水方程，正确的Well-Balanced源项应该是：
```python
# 正确的实现应该是:
S_hu = -g * self.h * d(z)/dx    # 但需要在通量中考虑压力项
# 或者更好的是使用水面梯度:
S_hu = -g * self.h * d(h+z)/dx + g * h * d(h)/dx
S_hu = -g * self.h * d(z)/dx    # 只有这部分，但压力项要特殊处理
```

### 影响
1. **静水不能保持静止** - 测试显示最终速度达0.108 m/s
2. **非物理的假波动**
3. **质量守恒被破坏**

### 理论背景
Well-Balanced格式要求：对于静水(u=0, v=0, h+z=常数)，所有源项和通量项应该精确抵消。当前实现不满足这个条件。

**参考文献**:
- Audusse et al. (2004) - A fast and stable well-balanced scheme
- Liang & Marche (2009) - Numerical resolution of well-balanced shallow water equations

---

## 🔴 问题2: 干湿边界数值溢出

### 位置
`shallow_water_solver.py`, 第470-499行 (HLL通量计算)

### 现象
```
RuntimeWarning: overflow encountered in multiply
RuntimeWarning: invalid value encountered
最终结果: NaN
```

### 根本原因
当 h → 0 时：
1. 速度 u = hu/h → ∞ (0/0)
2. 波速 c = sqrt(g*h) → 0
3. HLL通量项 s_L * s_R * (U_R - U_L) 可能溢出

### 当前代码问题
```python
# 第470-471行
c_L = np.sqrt(g * np.maximum(h_L, 0.0))
c_R = np.sqrt(g * np.maximum(h_R, 0.0))
```

这不够！当h很小时：
- u可能非常大（从self.hu/self.h计算）
- 导致数值溢出

### 需要的修复
1. **限制速度**: 当h < h_dry时，强制u = 0
2. **更鲁棒的波速估计**: 使用Davis波速或修正的HLL
3. **通量限制**: 检查并限制极端通量值

---

## 🔴 问题3: 质量守恒失败

### 测试结果
- 封闭区域，无边界通量
- **理论**: 质量应该绝对守恒
- **实际**: 质量误差 +1.22%

### 可能原因

#### 原因A: 边界处理
第869-884行的更新只对内部单元(i=1 to nx-2):
```python
for i in range(1, self.nx - 1):
    for j in range(1, self.ny - 1):
        # 更新...
```

**问题**: 边界单元(i=0, i=nx-1, j=0, j=ny-1)没有被更新！

#### 原因B: 边界条件赋值破坏守恒
边界条件中直接赋值：
```python
self.h[:, 0] = self.h[:, 1]  # 第754行
```

这不是守恒的！应该使用镜像或通量边界条件。

---

## 🔴 问题4: 数值收敛性差

### 测试结果
- 收敛阶: 0.05 (理论应 ≥ 1.0)
- 细网格反而不稳定

### 分析
1. **空间离散不一致**
   - 源项用中心差分
   - 通量用Godunov
   - 不匹配！

2. **时间步长可能过大**
   - CFL计算可能未考虑源项刚性

3. **数值粘性过大**
   - HLL本身有较大数值粘性
   - 可能需要HLLC或Roe

---

## 🔴 问题5: HLL波速估计简化

### 当前实现
```python
s_L = min(u_L - c_L, u_R - c_R)
s_R = max(u_L + c_L, u_R + c_R)
```

### 问题
这是简化的HLL波速估计，对于：
1. **稀疏波**: 估计不准（测试显示偏差>50%）
2. **跨音速流**: 可能不稳定
3. **激波**: 捕捉不清晰

### 建议
使用更精确的波速估计（如Davis估计或Einfeldt估计）。

---

## 🟡 次要问题

### 6. Manning摩擦项可能发散
第845行:
```python
friction_factor = g * n * n * V_mag / (h_pow + 1e-10)
```

当h很小时，h^(4/3)很小，摩擦因子很大，可能导致：
- 隐式不稳定性
- 需要隐式处理或源项时间分裂

### 7. 边界条件不完整
只有4个基本边界类型，缺少：
- Neumann边界
- Robin边界  
- 辐射边界

### 8. 缺少正定性保持(Positivity-Preserving)
没有确保水深严格非负的机制（仅事后clip）。

---

## 测试证据

### 测试1: 静水平衡
```
初始: u = 0.0 m/s
最终: u = 0.108 m/s  ❌
```
**应该**: u ≈ 0 (< 1e-6 m/s)

### 测试2: 质量守恒
```
误差: +1.22%  ❌
```
**应该**: < 0.1%

### 测试3: 干湿边界
```
结果: NaN  ❌
RuntimeWarning: overflow
```
**应该**: 数值稳定，无NaN

### 测试4: 收敛性
```
收敛阶: 0.05  ❌
```
**应该**: ≥ 1.0 (一阶格式)

---

## 影响评估

| 问题 | 严重性 | 影响 | 可用性 |
|------|--------|------|--------|
| Well-Balanced失败 | 🔴 严重 | 静水有假波动 | 不适用于静水域 |
| 干湿边界NaN | 🔴 严重 | 计算崩溃 | 不适用于溃坝 |
| 质量守恒差 | 🔴 严重 | 结果不可信 | 长时间模拟不可用 |
| 收敛性差 | 🔴 严重 | 细网格不可用 | 精度受限 |
| HLL波速简化 | 🟡 中等 | 精度降低 | 可接受 |

---

## 修复建议

### 优先级1 (必须修复)

#### 1.1 修复Well-Balanced源项
实现Audusse et al. (2004)的水面梯度重构方法：

```python
def compute_source_terms_wellbalanced(self):
    """Well-balanced源项处理"""
    g = self.config.g
    
    # 计算水面高程
    eta = self.h + self.z
    
    # 水面梯度
    deta_dx = compute_gradient(eta, self.dx, 'x')
    deta_dy = compute_gradient(eta, self.dy, 'y')
    
    # Well-balanced源项
    S_hu = -g * self.h * deta_dx
    S_hv = -g * self.h * deta_dy
    
    # 地形项应该在通量中处理（hydrostatic reconstruction）
    return S_h, S_hu, S_hv
```

#### 1.2 修复干湿边界
```python
def reconstruct_interface_states_robust(self, h, u, v):
    """鲁棒的界面重构"""
    h_dry = self.config.h_dry
    
    # 强制小水深处速度为零
    u_safe = np.where(h > h_dry, u, 0.0)
    v_safe = np.where(h > h_dry, v, 0.0)
    
    # 限制最大速度(物理合理性)
    u_max = 10 * np.sqrt(self.config.g * h.max())
    u_safe = np.clip(u_safe, -u_max, u_max)
    v_safe = np.clip(v_safe, -u_max, u_max)
    
    return h, u_safe, v_safe
```

#### 1.3 修复质量守恒 - 更新边界单元
```python
# 更新所有单元，包括边界
for i in range(self.nx):
    for j in range(self.ny):
        # 检查是否为内部单元
        if i > 0 and i < self.nx-1 and j > 0 and j < self.ny-1:
            # 正常更新
            ...
        # 边界单元通过BC更新，但要保持守恒
```

### 优先级2 (建议修复)

#### 2.1 改进波速估计
使用Davis或Einfeldt波速估计。

#### 2.2 添加正定性保持
实现Zhang & Shu的正定性保持限制器。

#### 2.3 隐式摩擦项
对Manning摩擦使用半隐式或隐式处理。

---

## 验证建议

修复后，必须通过以下测试：

### 必通过测试
1. ✅ 静水平衡 (u < 1e-8 m/s)
2. ✅ 质量守恒 (误差 < 0.01%)
3. ✅ 干湿边界无NaN
4. ✅ 网格收敛 (收敛阶 ≥ 0.9)

### 标准算例
1. Thacker旋转流
2. 抛物面水池
3. MacDonald溃坝
4. Carrier-Greenspan解析解

---

## 结论

### 当前状态
🔴 **不适合生产使用**

### 主要问题
1. 源项不Well-Balanced
2. 干湿边界不稳定  
3. 质量守恒差
4. 数值收敛性差

### 修复难度
- Well-Balanced: ⭐⭐⭐ (中等，需要理论知识)
- 干湿边界: ⭐⭐ (较易，加限制)
- 质量守恒: ⭐ (容易，修正循环)
- 收敛性: ⭐⭐⭐⭐ (困难，可能需要重构)

### 时间估计
- 最小可用修复: 2-3天
- 完整修复: 1-2周
- 验证测试: 1周

---

## 参考文献

1. Audusse et al. (2004) "A fast and stable well-balanced scheme with hydrostatic reconstruction for shallow water flows"
2. Liang & Marche (2009) "Numerical resolution of well-balanced shallow water equations"
3. Toro (2009) "Riemann Solvers and Numerical Methods for Fluid Dynamics"
4. Zhang & Shu (2011) "Positivity-preserving high order finite difference WENO schemes"
5. Hou et al. (2013) "A robust well-balanced model for 2D shallow water flows with wetting and drying"

---

**报告生成**: 2025-11-12  
**下一步**: 修复关键bug或选择其他成熟求解器

