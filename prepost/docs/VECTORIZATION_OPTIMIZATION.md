# 通量计算向量化优化

**日期**: 2025-10-29
**目标**: 使用NumPy向量化操作替代Python循环，提升求解器性能

## 当前实现问题

当前的HLL通量计算使用嵌套Python循环：

```python
# X方向通量 - O(nx * ny)复杂度，但Python循环开销大
for i in range(self.nx + 1):
    for j in range(self.ny):
        # 获取左右状态
        h_L, u_L, v_L = ...
        h_R, u_R, v_R = ...

        # 计算HLL通量
        ...
```

**性能问题**：
- Python循环每次迭代都有解释器开销
- 对于100×100网格：需要 10,000+ 次循环迭代
- 每次迭代涉及多次标量运算
- 无法利用NumPy的BLAS/LAPACK优化库
- 无法利用CPU的SIMD指令

## 优化策略

### 1. 数组切片获取左右状态

使用NumPy的数组切片一次性获取所有界面的状态：

**当前方式**（标量）：
```python
for i in range(self.nx + 1):
    for j in range(self.ny):
        if i == 0:
            h_L, h_R = self.h[i, j], self.h[i, j]
        else:
            h_L, h_R = self.h[i-1, j], self.h[i, j]
```

**优化方式**（向量化）：
```python
# 左状态：所有内部界面的左侧单元
h_L = self.h[:-1, :]  # shape: (nx, ny)
# 右状态：所有内部界面的右侧单元
h_R = self.h[1:, :]   # shape: (nx, ny)
```

### 2. 向量化计算波速和通量

**当前方式**：
```python
for i, j in all_interfaces:
    c_L = np.sqrt(g * max(h_L, 0.0))  # 标量运算
    c_R = np.sqrt(g * max(h_R, 0.0))
    s_L = min(u_L - c_L, u_R - c_R)
```

**优化方式**：
```python
# 向量化波速计算 - 单次操作处理所有界面
c_L = np.sqrt(g * np.maximum(h_L, 0.0))  # shape: (nx, ny)
c_R = np.sqrt(g * np.maximum(h_R, 0.0))  # shape: (nx, ny)
s_L = np.minimum(u_L - c_L, u_R - c_R)   # shape: (nx, ny)
```

### 3. 条件分支向量化

**当前方式**：
```python
for i, j in all_interfaces:
    if s_L >= 0:
        flux = F_L
    elif s_R <= 0:
        flux = F_R
    else:
        flux = (s_R*F_L - s_L*F_R + s_L*s_R*(U_R - U_L)) / (s_R - s_L)
```

**优化方式**：
```python
# 使用np.where实现向量化条件选择
flux_left = F_L  # 左侧超音速
flux_right = F_R  # 右侧超音速
flux_hll = (s_R*F_L - s_L*F_R + s_L*s_R*(U_R - U_L)) / (s_R - s_L)  # HLL

# 向量化选择
flux = np.where(s_L >= 0, flux_left,
                np.where(s_R <= 0, flux_right, flux_hll))
```

## 实现计划

### 新方法：`compute_fluxes_hll_vectorized()`

```python
def compute_fluxes_hll_vectorized(self):
    """向量化HLL通量计算"""
    g = self.config.g
    h_dry = self.config.h_dry

    # ========== X方向通量 ==========
    # 获取左右状态（内部界面）
    h_L = self.h[:-1, :]
    h_R = self.h[1:, :]
    u_L = self.u[:-1, :]
    u_R = self.u[1:, :]
    v_L = self.v[:-1, :]
    v_R = self.v[1:, :]

    # 干床掩码
    wet_mask = (h_L >= h_dry) | (h_R >= h_dry)

    # 波速
    c_L = np.sqrt(g * np.maximum(h_L, 0.0))
    c_R = np.sqrt(g * np.maximum(h_R, 0.0))
    s_L = np.minimum(u_L - c_L, u_R - c_R)
    s_R = np.maximum(u_L + c_L, u_R + c_R)

    # 物理通量
    F_L_h = h_L * u_L
    F_L_hu = h_L * u_L * u_L + 0.5 * g * h_L * h_L
    F_L_hv = h_L * u_L * v_L

    F_R_h = h_R * u_R
    F_R_hu = h_R * u_R * u_R + 0.5 * g * h_R * h_R
    F_R_hv = h_R * u_R * v_R

    # 守恒变量
    U_L_h = h_L
    U_L_hu = h_L * u_L
    U_L_hv = h_L * v_L

    U_R_h = h_R
    U_R_hu = h_R * u_R
    U_R_hv = h_R * v_R

    # HLL通量（分母避免除零）
    s_diff = s_R - s_L
    s_diff = np.where(np.abs(s_diff) < 1e-10, 1e-10, s_diff)

    hll_h = (s_R * F_L_h - s_L * F_R_h + s_L * s_R * (U_R_h - U_L_h)) / s_diff
    hll_hu = (s_R * F_L_hu - s_L * F_R_hu + s_L * s_R * (U_R_hu - U_L_hu)) / s_diff
    hll_hv = (s_R * F_L_hv - s_L * F_R_hv + s_L * s_R * (U_R_hv - U_L_hv)) / s_diff

    # 向量化条件选择
    flux_h = np.where(s_L >= 0, F_L_h,
                      np.where(s_R <= 0, F_R_h, hll_h))
    flux_hu = np.where(s_L >= 0, F_L_hu,
                       np.where(s_R <= 0, F_R_hu, hll_hu))
    flux_hv = np.where(s_L >= 0, F_L_hv,
                       np.where(s_R <= 0, F_R_hv, hll_hv))

    # 应用干床掩码
    flux_h = np.where(wet_mask, flux_h, 0.0)
    flux_hu = np.where(wet_mask, flux_hu, 0.0)
    flux_hv = np.where(wet_mask, flux_hv, 0.0)

    # 存储到内部界面
    self.flux_x[0, 1:-1, :] = flux_h
    self.flux_x[1, 1:-1, :] = flux_hu
    self.flux_x[2, 1:-1, :] = flux_hv

    # 边界通量（保持零通量）
    self.flux_x[:, 0, :] = 0.0
    self.flux_x[:, -1, :] = 0.0

    # ========== Y方向通量 ==========
    # 类似处理...
```

## 预期性能提升

### 理论分析

对于 N×N 网格：

**当前方法**：
- 时间复杂度：O(N²) × Python解释器开销
- 每个界面：~50-100条Python指令
- 总时间：T_old = N² × t_python

**向量化方法**：
- 时间复杂度：O(N²) × C语言执行
- 批量处理：单次NumPy调用处理N²个界面
- 总时间：T_new = C × N² × t_C + NumPy调用开销

**加速比估算**：
- Python vs C执行：~50-100x
- SIMD向量化：额外2-4x
- **总加速比：10-50x** （取决于网格大小）

### 基准测试计划

测试配置：
- 网格规模：50×50, 100×100, 200×200
- 时间步数：100步
- 测量：通量计算总时间

## 实现注意事项

1. **内存布局**：确保数组连续存储以利用缓存
2. **边界处理**：边界通量需要特殊处理（可保持循环或使用切片）
3. **数值稳定性**：注意除零保护（s_R - s_L接近零）
4. **向后兼容**：保留原方法作为`compute_fluxes_hll_loop()`用于调试

## 测试验证

优化后需要确保：
1. 数值结果与原方法一致（相对误差 < 1e-10）
2. 所有单元测试仍然通过
3. 质量守恒不受影响
4. 物理约束（非负性、有界性）保持

## 参考文献

- NumPy documentation: Array broadcasting and vectorization
- "High Performance Python" by Gorelick & Ozsvald (2020)
- LeVeque, R.J. (2002). Finite Volume Methods for Hyperbolic Problems

---

**文档版本**: 1.0
**状态**: 规划完成，准备实现

🤖 Generated with [Claude Code](https://claude.com/claude-code)
