# 🚀 下一步开发指南

**最后更新:** 2025-10-29
**当前状态:** Phase 2 Task 2.1完成，Task 2.2准备就绪

---

## ⚡ 快速启动（给下一个AI开发者）

### 1️⃣ 首先阅读这些文档（5分钟）

```bash
# 必读文档（按顺序）
cat docs/PROGRESS_REPORT.md        # 完整项目状态和路线图
cat docs/TASK_2.2_PLAN.md          # Task 2.2详细实施计划
cat docs/OPTIMIZATION_STRATEGY.md  # 优化代码模板
```

### 2️⃣ 检查环境（1分钟）

```bash
cd /home/user/HydroSIS-2D

# 检查CUDA是否可用
nvcc --version && nvidia-smi

# 有CUDA → 继续第3步
# 无CUDA → 跳到第5步（代码编写模式）
```

### 3️⃣ 构建和测试（有GPU时）（5分钟）

```bash
# 构建项目
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# 运行测试验证
cd ..
./run_tests.sh

# 应该看到：All tests passed ✓
```

### 4️⃣ 运行基线基准测试（有GPU时）（10分钟）

```bash
# 运行基准测试
./scripts/benchmark.sh \
    --sizes "500 1000" \
    --gpus "1" \
    --cases "dam_break" \
    --output results/baseline.csv

# 性能分析（可选但推荐）
nsys profile -o baseline_profile \
    ./build/hydrosis --config examples/applications/dam_break_valley.ini

ncu --set full -o kernel_analysis \
    --kernel-name update_cells_kernel \
    ./build/hydrosis --config examples/applications/dam_break_valley.ini
```

### 5️⃣ 开始Task 2.2实施

#### 📅 Day 1-4: 共享内存优化（最高优先级）

**目标:** 1.5-2.0× 加速

**任务清单:**
```bash
□ 创建优化代码文件
  touch src/cuda/cuda_kernels_optimized.cu
  touch include/cuda_kernels_optimized.cuh

□ 实现共享内存版本update_cells_kernel
  # 参考: docs/OPTIMIZATION_STRATEGY.md 第2.2节
  # 代码模板已提供

□ 设计共享内存布局
  __shared__ ConservativeVars s_U[BLOCK_Y+4][BLOCK_X+4];
  __shared__ real_t s_z[BLOCK_Y+4][BLOCK_X+4];

□ 实现halo加载逻辑
  - 主区域：所有线程参与
  - Halo区域：边界线程负责

□ 修改核函数从共享内存读取

□ 测试验证
  ./run_tests.sh
  ./scripts/benchmark.sh --sizes "500" --gpus "1" --output day1.csv

□ 对比性能
  python3 scripts/compare_performance.py baseline.csv day1.csv
```

**参考代码模板:** `docs/OPTIMIZATION_STRATEGY.md` 第2.2节

**验证标准:**
- ✅ 所有测试通过
- ✅ 质量守恒 < 1e-10
- ✅ 加速比 ≥ 1.5×

#### 📅 Day 5: MUSCL优化

**目标:** 1.3-1.5× 加速（MUSCL部分）

**核心改进:**
- 模板化限制器消除运行时分支
- 预计算差值避免重复
- 向量化处理（可选）

**参考:** `docs/OPTIMIZATION_STRATEGY.md` 第3节

#### 📅 Day 6-7: HLLC优化

**目标:** 1.2-1.4× 加速（HLLC部分）

**核心改进:**
- sqrt调用：7次 → 2次缓存
- 使用__fsqrt_rn()快速函数
- 优化分支顺序

**参考:** `docs/OPTIMIZATION_STRATEGY.md` 第4节

---

## 📊 当前项目状态概览

### ✅ 已完成

**Phase 1 (100%)**
- ✅ Task 1.1: 边界条件（4种类型）
- ✅ Task 1.2: 配置系统（INI文件）
- ✅ Task 1.3: 地形加载（ASCII Grid）
- ✅ Task 1.4: 应用案例（3个案例）

**Phase 2 (20%)**
- ✅ Task 2.1: 性能分析
  - 识别6大瓶颈
  - 制定优化策略
  - 准备实施计划

### ⏳ 待完成

**立即任务:**
- 🔥 Task 2.2: 核函数优化（10-14天）**← 当前任务**

**后续任务:**
- ⏳ Task 2.3: 核函数融合（7-10天）
- ⏳ Task 2.4: 多GPU优化（7-10天）
- ⏳ Task 2.5: 高级特性（7-10天）

---

## 🎯 性能优化目标

### 识别的瓶颈

| 优先级 | 瓶颈 | 预期加速 |
|--------|------|----------|
| 🔥 高 | update_cells内存访问 | 1.5-2.0× |
| 🔥 高 | MUSCL重构 | 1.3-1.5× |
| 🔥 高 | HLLC求解器 | 1.2-1.4× |
| 🟡 中 | 核函数启动开销 | 1.2-1.3× |
| 🟡 中 | 线程块配置 | 1.1-1.15× |
| 🟢 低 | 数据结构布局 | 1.1-1.2× |

**总体目标:** 2.5-3.0× 单GPU加速

---

## 🔧 关键命令速查

### 编译
```bash
cd build && cmake .. -DCMAKE_BUILD_TYPE=Release && make -j
```

### 测试
```bash
./run_tests.sh
./hydrosis --config examples/applications/dam_break_valley.ini --validate
```

### 基准测试
```bash
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output results.csv
python3 scripts/plot_performance.py --input results.csv
```

### 性能分析
```bash
nsys profile -o profile ./hydrosis --config config.ini
ncu --set full -o kernel ./hydrosis --config config.ini
```

### Git
```bash
git status
git add <files>
git commit -m "Task 2.2 Day X: [description]"
git push -u origin claude/gpu-accelerated-2d-hydrodynamic-model-011CUb7daEYBcxYzv656Dfbe
```

---

## 📚 关键文件位置

### 需要优化的代码
```
src/cuda/cuda_kernels.cu:246    ← update_cells_kernel (最高优先级)
src/cuda/cuda_kernels.cu:201    ← muscl_reconstruction
src/cuda/cuda_kernels.cu:101    ← hllc_riemann_solver
```

### 优化代码放置位置
```
src/cuda/cuda_kernels_optimized.cu     (新建)
include/cuda_kernels_optimized.cuh     (新建)
```

### 必读文档
```
docs/PROGRESS_REPORT.md           ⭐⭐⭐ 完整状态报告
docs/TASK_2.2_PLAN.md             ⭐⭐⭐ 14天实施计划
docs/OPTIMIZATION_STRATEGY.md     ⭐⭐⭐ 代码模板
docs/BASELINE_ANALYSIS.md         ⭐⭐ 瓶颈分析
```

---

## ⚠️ 重要提醒

### 如果没有GPU硬件

当前环境可能没有CUDA，此时：

**可以做:**
✅ 编写所有优化代码
✅ 设计优化方案
✅ 准备测试脚本
✅ 完善文档

**无法做:**
❌ 编译CUDA代码
❌ 运行基准测试
❌ 验证性能效果

**策略:** 先完成代码编写，等待GPU环境后验证

### 开发工作流

```
每次修改后:
1. 编译检查      → make
2. 运行测试      → ./run_tests.sh
3. 基准测试      → ./scripts/benchmark.sh
4. 对比性能      → python3 scripts/compare_performance.py
5. Git提交       → git commit
```

---

## 💡 成功标准

### Task 2.2必须达成

- ✅ 所有测试用例通过
- ✅ 质量守恒误差 < 1e-10
- ✅ 总加速比 ≥ 2.0×（最低）/ 2.5×（目标）
- ✅ 无数值精度损失（相对误差 < 1e-6）

### 可选目标

- ⭐ 总加速比 ≥ 3.0×
- ⭐ GPU利用率 > 80%
- ⭐ 内存减少 ≥ 10%

---

## 📞 需要帮助？

### 遇到问题时查看

1. **代码问题:** `docs/OPTIMIZATION_STRATEGY.md` 有完整代码模板
2. **性能问题:** `docs/BASELINE_ANALYSIS.md` 有瓶颈分析
3. **实施问题:** `docs/TASK_2.2_PLAN.md` 有详细步骤
4. **理解问题:** `docs/PROGRESS_REPORT.md` 有完整上下文

### 调试策略

- 使用Nsight工具定位瓶颈
- 对比原始版本验证正确性
- 检查共享内存大小限制（< 48KB）
- 分析bank conflicts

---

## 🎉 准备就绪！

**Task 2.2等待启动！**

**预计时间:** 10-14天
**预期结果:** 2.5-3.0× 加速

**立即行动:**
```bash
# 1. 阅读文档
cat docs/TASK_2.2_PLAN.md

# 2. 如果有GPU，运行基线测试
./scripts/benchmark.sh --sizes "500" --gpus "1" --output baseline.csv

# 3. 开始Day 1实施
# 参考 docs/OPTIMIZATION_STRATEGY.md 第2节
```

**祝开发顺利！** 🚀

---

**创建日期:** 2025-10-29
**用途:** 快速启动指南
**维护:** 每个Task完成后更新
