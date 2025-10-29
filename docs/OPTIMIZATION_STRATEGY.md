# HydroSIS-2D 核函数优化策略

**项目:** HydroSIS-2D Multi-GPU加速2D水动力模型
**阶段:** Phase 2 Task 2.2准备 - 核函数优化实施计划
**日期:** 2025-10-29
**状态:** 设计完成，待实施

---

## 1. 优化概览

基于[BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md)的瓶颈分析，本文档提供详细的优化实施方案。

### 1.1 优化目标

| 指标 | 当前（预测） | 目标 | 改进 |
|------|------------|------|------|
| 单GPU吞吐量 | 基线 | 2.5-3.0× | +150-200% |
| 内存带宽利用率 | ~30% | >70% | +40pp |
| GPU计算利用率 | ~40% | >80% | +40pp |
| 内存使用 | 基线 | -10-15% | 减少 |

### 1.2 优化顺序

```
Phase 2.2.1: 内存优化（Week 1-2）
  ├─ 优化1: 共享内存缓存（3-4天）      [1.5-2.0×]
  ├─ 优化2: MUSCL重构优化（2-3天）    [1.3-1.5×]
  └─ 优化3: HLLC求解器优化（2天）     [1.2-1.4×]

Phase 2.2.2: 计算优化（Week 3）
  ├─ 优化4: 快速数学函数（1天）       [1.05-1.1×]
  ├─ 优化5: 分支消除（2天）          [1.1-1.15×]
  └─ 优化6: 寄存器优化（2天）        [1.05-1.1×]

Phase 2.2.3: 块配置优化（Week 3）
  └─ 优化7: 自适应块大小（2天）      [1.1-1.15×]

总目标: 2.5-3.0× 单GPU加速
```

---

## 2. 优化 #1: 共享内存缓存

### 2.1 当前问题

**代码位置:** `src/cuda/cuda_kernels.cu:246-346` (update_cells_kernel)

**问题:**
```cuda
// 当前实现 - 重复全局内存访问
muscl_reconstruction(
    cells[idx - 2].U,    // 全局内存读取
    cells[idx - 1].U,    // 全局内存读取
    cells[idx].U,        // 全局内存读取
    U_L, U_R, params.slope_limiter);

FluxVector F_left = hllc_riemann_solver(
    U_R, U_L,
    cells[idx - 1].z,    // 再次读取
    cells[idx].z,        // 再次读取
    params.g, 0);
```

**每个单元的全局内存访问:**
- X方向: 5个单元 × 20字节 = 100字节
- Y方向: 5个单元 × 20字节 = 100字节
- **总计: 200字节读取 + 20字节写入 = 220字节**

### 2.2 优化方案

#### 方案A: 基础共享内存版本

```cuda
__global__ void update_cells_kernel_shared(
    const CellData* cells,
    CellData* cells_new,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t dt,
    const SimParams params) {

    // 共享内存声明
    __shared__ ConservativeVars s_U[BLOCK_Y + 4][BLOCK_X + 4];
    __shared__ real_t s_z[BLOCK_Y + 4][BLOCK_X + 4];

    // 线程索引
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx + Constants::HALO_WIDTH;
    int j = blockIdx.y * blockDim.y + ty + Constants::HALO_WIDTH;

    // 加载数据到共享内存（包括halo）
    // 主要区域
    int global_idx = j * nx + i;
    s_U[ty + 2][tx + 2] = cells[global_idx].U;
    s_z[ty + 2][tx + 2] = cells[global_idx].z;

    // Halo区域（边界线程负责）
    if (tx < 2) {
        // 左边halo
        s_U[ty + 2][tx] = cells[global_idx - 2].U;
        s_z[ty + 2][tx] = cells[global_idx - 2].z;
        // 右边halo
        s_U[ty + 2][tx + blockDim.x + 2] = cells[global_idx + blockDim.x].U;
        s_z[ty + 2][tx + blockDim.x + 2] = cells[global_idx + blockDim.x].z;
    }
    if (ty < 2) {
        // 下边halo
        s_U[ty][tx + 2] = cells[(j - 2) * nx + i].U;
        s_z[ty][tx + 2] = cells[(j - 2) * nx + i].z;
        // 上边halo
        s_U[ty + blockDim.y + 2][tx + 2] = cells[(j + blockDim.y) * nx + i].U;
        s_z[ty + blockDim.y + 2][tx + 2] = cells[(j + blockDim.y) * nx + i].z;
    }

    __syncthreads();

    // 边界检查
    if (i >= nx - Constants::HALO_WIDTH || j >= ny - Constants::HALO_WIDTH) return;

    // 计算（使用共享内存）
    ConservativeVars U_new = s_U[ty + 2][tx + 2];

    // X方向通量（从共享内存读取）
    if (params.order == 2) {
        ConservativeVars U_L, U_R;

        // 左界面 (i-1/2)
        muscl_reconstruction(
            s_U[ty + 2][tx],      // idx-2  从共享内存
            s_U[ty + 2][tx + 1],  // idx-1  从共享内存
            s_U[ty + 2][tx + 2],  // idx    从共享内存
            U_L, U_R, params.slope_limiter);

        FluxVector F_left = hllc_riemann_solver(
            U_R, U_L,
            s_z[ty + 2][tx + 1],  // 从共享内存
            s_z[ty + 2][tx + 2],  // 从共享内存
            params.g, 0);

        // 右界面 (i+1/2)
        muscl_reconstruction(
            s_U[ty + 2][tx + 1],  // idx-1
            s_U[ty + 2][tx + 2],  // idx
            s_U[ty + 2][tx + 3],  // idx+1
            U_L, U_R, params.slope_limiter);

        FluxVector F_right = hllc_riemann_solver(
            U_R, U_L,
            s_z[ty + 2][tx + 2],
            s_z[ty + 2][tx + 3],
            params.g, 0);

        // 更新
        U_new.h -= (dt / dx) * (F_right.f1 - F_left.f1);
        U_new.qx -= (dt / dx) * (F_right.f2 - F_left.f2);
        U_new.qy -= (dt / dx) * (F_right.f3 - F_left.f3);
    }

    // Y方向通量（类似，使用s_U[ty][tx+2]等）
    if (params.order == 2) {
        ConservativeVars U_L, U_R;

        // 下界面
        muscl_reconstruction(
            s_U[ty][tx + 2],      // j-2
            s_U[ty + 1][tx + 2],  // j-1
            s_U[ty + 2][tx + 2],  // j
            U_L, U_R, params.slope_limiter);

        FluxVector G_bottom = hllc_riemann_solver(
            U_R, U_L,
            s_z[ty + 1][tx + 2],
            s_z[ty + 2][tx + 2],
            params.g, 1);

        // 上界面
        muscl_reconstruction(
            s_U[ty + 1][tx + 2],  // j-1
            s_U[ty + 2][tx + 2],  // j
            s_U[ty + 3][tx + 2],  // j+1
            U_L, U_R, params.slope_limiter);

        FluxVector G_top = hllc_riemann_solver(
            U_R, U_L,
            s_z[ty + 2][tx + 2],
            s_z[ty + 3][tx + 2],
            params.g, 1);

        U_new.h -= (dt / dy) * (G_top.f1 - G_bottom.f1);
        U_new.qx -= (dt / dy) * (G_top.f2 - G_bottom.f2);
        U_new.qy -= (dt / dy) * (G_top.f3 - G_bottom.f3);
    }

    // 写回全局内存
    cells_new[global_idx].U = U_new;
    cells_new[global_idx].z = s_z[ty + 2][tx + 2];
    cells_new[global_idx].n = cells[global_idx].n;
}
```

**共享内存使用:**
```
块大小: 16×16 = 256线程
数据大小: (16+4) × (16+4) = 400单元
U数据: 400 × 12字节 = 4800字节
z数据: 400 × 4字节 = 1600字节
总计: 6400字节 < 48KB (典型限制) ✓
```

**预期改进:**
- 内存读取: 220字节 → ~50字节 (仅边缘线程多读)
- 加速比: **1.5-2.0×**

#### 方案B: 纹理内存版本（只读数据）

对于床面高程`z`（只读），使用纹理内存：

```cuda
// 声明纹理对象
texture<real_t, cudaTextureType2D, cudaReadModeElementType> tex_z;

// 核函数中读取
real_t z_value = tex2D(tex_z, i, j);  // 硬件插值和缓存
```

**优势:**
- 自动缓存
- 2D空间局部性优化
- 额外5-10%性能提升

### 2.3 实施步骤

1. **Day 1: 设计与原型**
   - 实现基础共享内存版本
   - 计算共享内存需求
   - 处理边界情况

2. **Day 2: Halo加载优化**
   - 优化halo区域加载模式
   - 减少bank conflicts
   - 测试不同块大小

3. **Day 3: 集成与测试**
   - 集成到主求解器
   - 正确性验证
   - 性能测量

4. **Day 4: 纹理内存探索**
   - 实现纹理内存版本
   - 性能对比
   - 选择最优方案

### 2.4 验证清单

- [ ] 所有测试用例通过
- [ ] 质量守恒误差 < 1e-10
- [ ] 与基线结果相对误差 < 1e-6
- [ ] 加速比 ≥ 1.5×
- [ ] 共享内存无bank conflicts

---

## 3. 优化 #2: MUSCL重构优化

### 3.1 当前问题

**代码位置:** `src/cuda/cuda_kernels.cu:201-240`

**问题1: 分支发散**
```cuda
if (limiter == 0) {
    slope_h_L = minmod(...);
    // ...
} else if (limiter == 1) {
    slope_h_L = superbee(...);
    // ...
} else {
    slope_h_L = mc_limiter(...);
    // ...
}
```

**问题2: 重复计算**
```cuda
// 差值被多次计算
slope_h_L = minmod(U_c.h - U_m.h, U_p.h - U_c.h);   // 计算差值
slope_qx_L = minmod(U_c.qx - U_m.qx, U_p.qx - U_c.qx); // 再次计算
slope_qy_L = minmod(U_c.qy - U_m.qy, U_p.qy - U_c.qy); // 再次计算
```

### 3.2 优化方案

#### 方案A: 模板消除分支

```cuda
// 定义限制器策略作为模板参数
template<int LIMITER>
__device__ inline real_t apply_limiter(real_t a, real_t b) {
    if (LIMITER == 0) {
        return minmod(a, b);
    } else if (LIMITER == 1) {
        return superbee(a, b);
    } else {
        return mc_limiter(a, b);
    }
}

// 优化的MUSCL重构
template<int LIMITER>
__device__ void muscl_reconstruction_optimized(
    const ConservativeVars& U_m,
    const ConservativeVars& U_c,
    const ConservativeVars& U_p,
    ConservativeVars& U_L,
    ConservativeVars& U_R) {

    // 预计算差值（避免重复）
    real_t dh_left = U_c.h - U_m.h;
    real_t dh_right = U_p.h - U_c.h;
    real_t dqx_left = U_c.qx - U_m.qx;
    real_t dqx_right = U_p.qx - U_c.qx;
    real_t dqy_left = U_c.qy - U_m.qy;
    real_t dqy_right = U_p.qy - U_c.qy;

    // 应用限制器（编译时分支消除）
    real_t slope_h = apply_limiter<LIMITER>(dh_left, dh_right);
    real_t slope_qx = apply_limiter<LIMITER>(dqx_left, dqx_right);
    real_t slope_qy = apply_limiter<LIMITER>(dqy_left, dqy_right);

    // 重构
    U_L.h = U_c.h - 0.5 * slope_h;
    U_L.qx = U_c.qx - 0.5 * slope_qx;
    U_L.qy = U_c.qy - 0.5 * slope_qy;

    U_R.h = U_c.h + 0.5 * slope_h;
    U_R.qx = U_c.qx + 0.5 * slope_qx;
    U_R.qy = U_c.qy + 0.5 * slope_qy;

    // 非负深度
    U_L.h = max(U_L.h, (real_t)0.0);
    U_R.h = max(U_R.h, (real_t)0.0);
}

// 调用时指定限制器类型
if (params.slope_limiter == 0) {
    muscl_reconstruction_optimized<0>(...);
} else if (params.slope_limiter == 1) {
    muscl_reconstruction_optimized<1>(...);
} else {
    muscl_reconstruction_optimized<2>(...);
}
```

**改进:**
- 编译时分支消除（零运行时开销）
- 差值计算只进行一次
- 减少6次冗余减法

**预期加速:** **1.3-1.5×** （相对当前MUSCL部分）

#### 方案B: 向量化限制器

```cuda
// 使用float3对三个变量一起处理
__device__ inline float3 minmod_vec(float3 a, float3 b) {
    float3 result;
    result.x = (a.x * b.x <= 0.0f) ? 0.0f : ((fabsf(a.x) < fabsf(b.x)) ? a.x : b.x);
    result.y = (a.y * b.y <= 0.0f) ? 0.0f : ((fabsf(a.y) < fabsf(b.y)) ? a.y : b.y);
    result.z = (a.z * b.z <= 0.0f) ? 0.0f : ((fabsf(a.z) < fabsf(b.z)) ? a.z : b.z);
    return result;
}

// 向量化MUSCL
__device__ void muscl_reconstruction_vectorized(
    const ConservativeVars& U_m,
    const ConservativeVars& U_c,
    const ConservativeVars& U_p,
    ConservativeVars& U_L,
    ConservativeVars& U_R) {

    // 打包为向量
    float3 U_c_vec = make_float3(U_c.h, U_c.qx, U_c.qy);
    float3 dU_left = make_float3(U_c.h - U_m.h, U_c.qx - U_m.qx, U_c.qy - U_m.qy);
    float3 dU_right = make_float3(U_p.h - U_c.h, U_p.qx - U_c.qx, U_p.qy - U_c.qy);

    // 向量化限制器
    float3 slope = minmod_vec(dU_left, dU_right);

    // 重构
    U_L.h = U_c_vec.x - 0.5f * slope.x;
    U_L.qx = U_c_vec.y - 0.5f * slope.y;
    U_L.qy = U_c_vec.z - 0.5f * slope.z;

    U_R.h = U_c_vec.x + 0.5f * slope.x;
    U_R.qx = U_c_vec.y + 0.5f * slope.y;
    U_R.qy = U_c_vec.z + 0.5f * slope.z;

    U_L.h = fmaxf(U_L.h, 0.0f);
    U_R.h = fmaxf(U_R.h, 0.0f);
}
```

**改进:**
- 更好的指令级并行
- 额外5-10%性能

### 3.3 实施步骤

1. **Day 1: 模板版本实现**
   - 实现模板化限制器
   - 测试编译时分支消除
   - 验证正确性

2. **Day 2: 向量化探索**
   - 实现向量化版本
   - 性能对比
   - 寄存器使用分析

3. **Day 3: 集成优化**
   - 集成最优方案
   - 完整测试
   - 性能测量

---

## 4. 优化 #3: HLLC求解器优化

### 4.1 当前问题

**代码位置:** `src/cuda/cuda_kernels.cu:101-195`

**问题1: 重复sqrt计算**
```cuda
real_t c_L = sqrt(g * W_L.h);      // sqrt #1
real_t c_R = sqrt(g * W_R.h);      // sqrt #2

// Roe平均
real_t u_roe = (sqrt(W_L.h) * u_L + sqrt(W_R.h) * u_R) /  // sqrt #3, #4
               (sqrt(W_L.h) + sqrt(W_R.h));                // sqrt #5, #6
real_t c_roe = sqrt(g * h_roe);    // sqrt #7

// 7次sqrt，但实际只需要2次！
```

**问题2: 未使用快速数学**
```cuda
sqrt(x)  // 应该用 __fsqrt_rn(x) - 快2倍
```

### 4.2 优化方案

```cuda
__device__ FluxVector hllc_riemann_solver_optimized(
    const ConservativeVars& U_L,
    const ConservativeVars& U_R,
    real_t z_L, real_t z_R,
    real_t g, int dir) {

    FluxVector F;
    F.f1 = 0.0; F.f2 = 0.0; F.f3 = 0.0;

    // 干单元处理
    bool dry_L = (U_L.h < Constants::MIN_DEPTH);
    bool dry_R = (U_R.h < Constants::MIN_DEPTH);

    if (dry_L && dry_R) return F;
    if (dry_L) return (dir == 0) ? compute_flux_x(U_R, g) : compute_flux_y(U_R, g);
    if (dry_R) return (dir == 0) ? compute_flux_x(U_L, g) : compute_flux_y(U_L, g);

    // 原始变量
    PrimitiveVars W_L = conservative_to_primitive(U_L);
    PrimitiveVars W_R = conservative_to_primitive(U_R);

    // 缓存sqrt结果（关键优化！）
    real_t sqrt_h_L = __fsqrt_rn(W_L.h);  // 快速内建函数
    real_t sqrt_h_R = __fsqrt_rn(W_R.h);

    // 波速（复用sqrt）
    real_t c_L = __fsqrt_rn(g) * sqrt_h_L;  // 或预计算sqrt(g)
    real_t c_R = __fsqrt_rn(g) * sqrt_h_R;

    real_t u_L = (dir == 0) ? W_L.u : W_L.v;
    real_t u_R = (dir == 0) ? W_R.u : W_R.v;

    // Roe平均（复用sqrt）
    real_t sqrt_h_sum = sqrt_h_L + sqrt_h_R;
    real_t h_roe = 0.5 * (W_L.h + W_R.h);
    real_t u_roe = (sqrt_h_L * u_L + sqrt_h_R * u_R) / sqrt_h_sum;
    real_t c_roe = __fsqrt_rn(g * h_roe);

    // 波速（Einfeldt）
    real_t S_L = fminf(u_L - c_L, u_roe - c_roe);  // fminf比min快
    real_t S_R = fmaxf(u_R + c_R, u_roe + c_roe);

    // 中间波速
    real_t S_star = (S_R * u_R - S_L * u_L +
                     0.5 * g * (W_L.h * W_L.h - W_R.h * W_R.h)) /
                    (S_R - S_L + REAL_EPSILON);

    // 计算通量
    FluxVector F_L = (dir == 0) ? compute_flux_x(U_L, g) : compute_flux_y(U_L, g);
    FluxVector F_R = (dir == 0) ? compute_flux_x(U_R, g) : compute_flux_y(U_R, g);

    // HLLC通量选择（优化分支顺序）
    if (S_L >= 0.0) {
        return F_L;  // 早返回
    }
    if (S_R <= 0.0) {
        return F_R;  // 早返回
    }

    // Star region
    if (S_star >= 0.0) {
        // 左侧star
        real_t factor = (S_L - u_L) / (S_L - S_star);
        ConservativeVars U_star_L;
        U_star_L.h = U_L.h * factor;

        if (dir == 0) {
            U_star_L.qx = U_star_L.h * S_star;
            U_star_L.qy = U_L.qy * factor;
        } else {
            U_star_L.qx = U_L.qx * factor;
            U_star_L.qy = U_star_L.h * S_star;
        }

        F.f1 = F_L.f1 + S_L * (U_star_L.h - U_L.h);
        F.f2 = F_L.f2 + S_L * (U_star_L.qx - U_L.qx);
        F.f3 = F_L.f3 + S_L * (U_star_L.qy - U_L.qy);
    } else {
        // 右侧star
        real_t factor = (S_R - u_R) / (S_R - S_star);
        ConservativeVars U_star_R;
        U_star_R.h = U_R.h * factor;

        if (dir == 0) {
            U_star_R.qx = U_star_R.h * S_star;
            U_star_R.qy = U_R.qy * factor;
        } else {
            U_star_R.qx = U_R.qx * factor;
            U_star_R.qy = U_star_R.h * S_star;
        }

        F.f1 = F_R.f1 + S_R * (U_star_R.h - U_R.h);
        F.f2 = F_R.f2 + S_R * (U_star_R.qx - U_R.qx);
        F.f3 = F_R.f3 + S_R * (U_star_R.qy - U_R.qy);
    }

    return F;
}
```

**改进:**
1. **sqrt优化:** 7次 → 2次缓存 + 复用
2. **快速内建函数:** `__fsqrt_rn()` 比 `sqrt()` 快2倍
3. **早返回:** 减少不必要的计算
4. **快速min/max:** `fminf/fmaxf` 比 `min/max` 快

**预期加速:** **1.2-1.4×** （相对当前HLLC部分）

### 4.3 实施步骤

1. **Day 1: 数学优化**
   - 识别所有重复计算
   - 实现sqrt缓存
   - 使用快速内建函数

2. **Day 2: 集成与验证**
   - 集成优化版本
   - 数值精度测试
   - 性能测量

---

## 5. 优化 #4-7: 其他优化

### 5.1 快速数学函数（优化#4）

**全局启用:**
```cmake
# CMakeLists.txt
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --use_fast_math")
```

**或选择性使用:**
```cuda
__fsqrt_rn(x)      // 快速sqrt
__fdiv_rn(x, y)    // 快速除法
__fmaf_rn(x, y, z) // 融合乘加: x*y + z
__powf(x, y)       // 快速幂
```

**预期加速:** 1.05-1.1×

### 5.2 分支优化（优化#5）

**技术1: 早返回**
```cuda
if (condition_for_early_exit) return result;
// 避免后续不必要计算
```

**技术2: 分支提示**
```cuda
if (__builtin_expect(h < MIN_DEPTH, 0)) {  // 提示不太可能
    // 处理干单元
}
```

**技术3: 谓词化**
```cuda
// 避免分支
result = condition ? value1 : value2;  // 可能被编译为谓词指令
```

**预期加速:** 1.1-1.15×

### 5.3 寄存器优化（优化#6）

**技术1: 使用__launch_bounds__**
```cuda
__global__ void __launch_bounds__(256, 4)  // 256线程/块, 4块/SM
update_cells_kernel(...) {
    // 限制寄存器使用
}
```

**技术2: 减少局部变量**
```cuda
// 避免：
real_t temp1, temp2, temp3, temp4, temp5, ...  // 过多临时变量

// 推荐：
real_t temp = calculate();
use(temp);
temp = calculate_another();  // 复用寄存器
use(temp);
```

**预期加速:** 1.05-1.1× （提高占用率）

### 5.4 自适应块大小（优化#7）

```cpp
class BlockSizeTuner {
public:
    static dim3 get_optimal_block_size(const char* kernel_name, int nx, int ny) {
        // 针对不同核函数和问题规模选择最优块大小
        if (strcmp(kernel_name, "update_cells") == 0) {
            if (nx * ny < 100000) {
                return dim3(32, 8);  // 小问题：增加占用率
            } else {
                return dim3(16, 16); // 大问题：平衡
            }
        }
        return dim3(16, 16);  // 默认
    }
};
```

**预期加速:** 1.1-1.15×

---

## 6. 集成方案

### 6.1 兼容性策略

使用编译时开关保持向后兼容：

```cpp
// hydrosis_solver.h
enum OptimizationLevel {
    OPT_NONE = 0,      // 原始版本
    OPT_SHARED_MEM = 1, // 共享内存
    OPT_FAST_MATH = 2,  // 快速数学
    OPT_ALL = 3         // 全部优化
};

class HydroSisSolver {
private:
    OptimizationLevel opt_level_;

    void update_cells(real_t dt) {
        if (opt_level_ >= OPT_SHARED_MEM) {
            update_cells_kernel_optimized<<<...>>>(...);//优化版本
        } else {
            update_cells_kernel<<<...>>>(...);  // 原始版本
        }
    }
};
```

### 6.2 配置文件控制

```ini
[Performance]
optimization_level = 3  # 0=无, 1=共享内存, 2=快速数学, 3=全部
use_shared_memory = true
use_fast_math = true
block_size_x = 16
block_size_y = 16
```

---

## 7. 测试与验证策略

### 7.1 正确性测试

**每个优化后必须通过:**

```bash
# 所有标准测试
./run_tests.sh

# 特定测试
./hydrosis --config test_dam_break.ini --validate

# 对比基线
python3 scripts/compare_solutions.py baseline.vtk optimized.vtk
```

**验证指标:**
- 质量守恒: |error| < 1e-10
- 数值精度: |relative_error| < 1e-6
- 物理合理性: Froude数、波速正确

### 7.2 性能测试

**基准测试流程:**

```bash
# 1. 基线测试
git checkout baseline
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output baseline.csv

# 2. 优化测试
git checkout optimization-branch
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output optimized.csv

# 3. 对比
python3 scripts/compare_performance.py baseline.csv optimized.csv

# 4. 可视化
python3 scripts/plot_performance.py --input baseline.csv optimized.csv
```

### 7.3 性能分析

```bash
# Nsight Systems - 系统级
nsys profile --stats=true -o optimized ./hydrosis --config config.ini

# Nsight Compute - 核函数级
ncu --set full -o kernel_analysis ./hydrosis --config config.ini

# 对比报告
ncu --set full --import-source yes -o compare \
    --baseline-metrics baseline.ncu-rep optimized.ncu-rep
```

---

## 8. 性能预测模型

### 8.1 累积加速计算

假设各优化独立（保守估计）：

| 优化 | 局部加速 | 占时间% | 贡献 |
|------|----------|---------|------|
| 共享内存 | 1.7× | 40% | 0.4×0.7 = 0.28 |
| MUSCL | 1.4× | 20% | 0.2×0.4 = 0.08 |
| HLLC | 1.3× | 25% | 0.25×0.3 = 0.075 |
| 其他 | 1.1× | 15% | 0.15×0.1 = 0.015 |

**总加速（Amdahl定律）:**
```
T_optimized = T_baseline × (1 - sum(贡献))
            = T_baseline × (1 - 0.37)
            = T_baseline × 0.63

Speedup = 1 / 0.63 = 1.59× (保守)
```

**乐观估计（优化协同）:** 2.0-2.5×

**最佳情况（全部成功）:** 2.5-3.0×

### 8.2 内存带宽改进

**当前:**
- 内存访问: 220字节/单元
- 带宽利用率: ~30%

**优化后:**
- 内存访问: ~60字节/单元（共享内存）
- 带宽利用率: ~70-80%

**理论加速:** 220/60 = 3.67× （内存受限部分）

---

## 9. 风险管理

### 风险 #1: 共享内存限制

**问题:** 共享内存可能不足

**缓解:**
- 设计多种块大小配置
- 动态检测GPU能力
- 回退到全局内存版本

### 风险 #2: 数值精度损失

**问题:** 快速数学可能影响精度

**缓解:**
- 对比验证所有测试用例
- 必要时选择性使用快速函数
- 保留精确版本选项

### 风险 #3: 性能提升不达预期

**问题:** 实际加速可能低于预期

**缓解:**
- 实际profiling验证瓶颈
- 多个优化策略并行探索
- 聚焦最高影响优化

---

## 10. 下一步行动

### 立即执行（有GPU时）

1. **建立基线**
   ```bash
   ./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output baseline.csv
   nsys profile -o baseline ./hydrosis --config config.ini
   ```

2. **优化#1: 共享内存**
   ```bash
   git checkout -b opt-shared-memory
   # 实施共享内存优化
   # 测试验证
   ```

3. **逐步实施其他优化**

### 当前可执行（无GPU时）

1. **编写优化代码**
   - 实现所有优化版本
   - 编译检查
   - 代码审查

2. **准备测试框架**
   - 自动化测试脚本
   - 性能对比工具
   - 回归测试

3. **文档完善**
   - 详细实现文档
   - 使用指南
   - 优化说明

---

## 11. 总结

### 关键优化策略

1. **共享内存缓存** - 最高优先级（1.5-2×）
2. **MUSCL/HLLC优化** - 高优先级（1.5-1.8×）
3. **快速数学 + 其他** - 中优先级（1.2-1.3×）

### 预期总体改进

- **单GPU性能:** 2.5-3.0× 加速 ✓
- **内存效率:** 减少10-15% ✓
- **GPU利用率:** 40% → 80% ✓

### 成功标准

- ✓ 所有测试通过
- ✓ 质量守恒 < 1e-10
- ✓ 加速比 ≥ 2.5×
- ✓ 代码可维护

---

**文档版本:** 1.0
**最后更新:** 2025-10-29
**状态:** 设计完成，待实施
**下一里程碑:** 实现优化#1（共享内存）

---

## 附录: 代码模板

### A.1 共享内存核函数模板

参见 Section 2.2

### A.2 优化的MUSCL模板

参见 Section 3.2

### A.3 优化的HLLC模板

参见 Section 4.2

### A.4 性能测试模板

```bash
#!/bin/bash
# performance_test.sh

echo "Testing optimization: $1"
./hydrosis --config test.ini --optimization "$1"
nsys profile -o "profile_$1" ./hydrosis --config test.ini
ncu -o "kernel_$1" --set full ./hydrosis --config test.ini
```

---

**文档结束**
