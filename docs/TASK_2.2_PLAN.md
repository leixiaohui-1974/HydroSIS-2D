# Task 2.2 实施计划：核函数优化

**项目:** HydroSIS-2D
**阶段:** Phase 2 - GPU优化与扩展
**任务:** Task 2.2 - Kernel Optimization
**持续时间:** 10-14天
**状态:** 📋 计划完成，待实施

---

## 任务概览

### 目标

完成HydroSIS-2D求解器的核函数优化，实现：
- **性能目标:** 2.5-3.0× 单GPU加速
- **内存优化:** 减少10-15%内存使用
- **GPU利用率:** 从~40%提升到>80%
- **正确性:** 保持所有验证测试通过

### 前置条件

- ✅ Phase 1 完成（边界条件、配置系统、地形加载）
- ✅ Task 2.1 分析完成（瓶颈识别、优化策略）
- ⏳ GPU硬件可用（待确认）

### 可交付成果

1. 优化的CUDA核函数代码
2. 性能对比数据和图表
3. 优化报告
4. 更新的文档

---

## 详细时间表

### Week 1: 内存访问优化（Day 1-7）

#### Day 1-2: 共享内存优化 - 设计与原型

**目标:** 设计并实现update_cells_kernel的共享内存版本

**任务:**
- [ ] 分析当前内存访问模式
- [ ] 计算共享内存需求
- [ ] 设计halo加载策略
- [ ] 实现基础共享内存版本
- [ ] 编译测试

**代码文件:**
- 创建: `src/cuda/cuda_kernels_optimized.cu`
- 修改: `include/cuda_kernels.cuh`

**输出:**
- 共享内存版本原型代码
- 内存布局设计文档

**验证标准:**
- 编译无错误
- 共享内存大小 < 48KB
- 无bank conflicts（理论分析）

#### Day 3-4: 共享内存优化 - 实现与调优

**目标:** 完善实现并优化性能

**任务:**
- [ ] 实现2D halo加载逻辑
- [ ] 优化边界线程处理
- [ ] 测试不同块大小（8×8, 16×16, 32×8等）
- [ ] 消除bank conflicts
- [ ] 集成到主求解器

**代码文件:**
- 修改: `src/cuda/cuda_kernels_optimized.cu`
- 修改: `src/solver/hydrosis_solver.cpp`

**输出:**
- 完整共享内存实现
- 块大小性能数据

**验证标准:**
- 所有测试用例通过
- 质量守恒误差 < 1e-10
- 加速比 ≥ 1.5×

#### Day 5: MUSCL重构优化

**目标:** 优化MUSCL重构消除分支和重复计算

**任务:**
- [ ] 实现模板化限制器
- [ ] 预计算差值避免重复
- [ ] 测试向量化版本
- [ ] 性能对比选择最优方案

**代码文件:**
- 修改: `src/cuda/cuda_kernels_optimized.cu`
- 新增: `include/muscl_optimized.cuh`

**输出:**
- 优化的MUSCL实现
- 性能对比数据

**验证标准:**
- 数值结果与原版相同
- 加速比 ≥ 1.3×（MUSCL部分）

#### Day 6-7: HLLC求解器优化

**目标:** 优化Riemann求解器减少重复计算和使用快速数学

**任务:**
- [ ] 缓存sqrt结果
- [ ] 使用__fsqrt_rn()等内建函数
- [ ] 优化分支顺序
- [ ] 实现早返回优化
- [ ] 性能测试

**代码文件:**
- 修改: `src/cuda/cuda_kernels_optimized.cu`
- 新增: `include/riemann_optimized.cuh`

**输出:**
- 优化的HLLC实现
- sqrt使用减少分析

**验证标准:**
- Riemann问题精度保持
- 加速比 ≥ 1.2×（HLLC部分）

### Week 2: 计算与配置优化（Day 8-14）

#### Day 8: 快速数学函数集成

**目标:** 使用CUDA快速数学库优化浮点运算

**任务:**
- [ ] 识别所有sqrt、div、pow等调用
- [ ] 替换为__fsqrt_rn、__fdiv_rn等
- [ ] 测试数值精度影响
- [ ] 选择性启用（保留精确版本）

**代码文件:**
- 修改: `src/cuda/cuda_kernels_optimized.cu`
- 修改: `CMakeLists.txt`（可选--use_fast_math）

**输出:**
- 快速数学版本
- 精度测试报告

**验证标准:**
- 质量守恒误差 < 1e-10
- 相对误差 < 1e-6
- 加速比 ≥ 1.05×

#### Day 9-10: 分支与寄存器优化

**目标:** 减少warp发散和寄存器压力

**任务:**
- [ ] 识别关键分支点
- [ ] 实现分支提示和谓词化
- [ ] 添加__launch_bounds__
- [ ] 减少临时变量
- [ ] 分析寄存器使用（ptxas -v）

**代码文件:**
- 修改: `src/cuda/cuda_kernels_optimized.cu`

**输出:**
- 寄存器使用报告
- 占用率分析

**验证标准:**
- 占用率 ≥ 50%
- 寄存器溢出 = 0
- 加速比 ≥ 1.1×

#### Day 11-12: 块大小自适应优化

**目标:** 实现运行时块大小选择

**任务:**
- [ ] 设计BlockSizeTuner类
- [ ] 测试不同核函数的最优块大小
- [ ] 实现问题规模自适应
- [ ] 集成到求解器

**代码文件:**
- 新增: `include/block_tuner.h`
- 新增: `src/utils/block_tuner.cpp`
- 修改: `src/solver/hydrosis_solver.cpp`

**输出:**
- 自适应块大小系统
- 不同配置性能数据

**验证标准:**
- 自动选择最优配置
- 平均加速比 ≥ 1.1×

#### Day 13: 集成测试与性能验证

**目标:** 完整系统测试和性能测量

**任务:**
- [ ] 运行完整测试套件
- [ ] 执行基准测试
- [ ] 生成性能对比图表
- [ ] 验证所有目标达成

**测试命令:**
```bash
# 完整测试
./run_all_tests.sh

# 性能基准
./scripts/benchmark.sh --sizes "500 1000 2000" --gpus "1" \
    --output results/task_2.2_final.csv

# 可视化
python3 scripts/plot_performance.py \
    --input results/baseline.csv results/task_2.2_final.csv \
    --output results/task_2.2_comparison.png
```

**输出:**
- 测试结果报告
- 性能对比图表
- 问题诊断（如有）

**验证标准:**
- 所有测试通过 ✓
- 总加速比 ≥ 2.5× ✓
- 无性能回归 ✓

#### Day 14: 文档与代码清理

**目标:** 完善文档并清理代码

**任务:**
- [ ] 编写Task 2.2完成报告
- [ ] 更新性能分析指南
- [ ] 添加代码注释
- [ ] 清理调试代码
- [ ] 准备Git提交

**文档:**
- 创建: `docs/TASK_2.2_REPORT.md`
- 更新: `docs/DEVELOPMENT_STATUS.md`
- 更新: `docs/PERFORMANCE_ANALYSIS_GUIDE.md`

**输出:**
- 完整任务报告
- 清理的代码库
- Git commit

---

## 实施策略

### 开发流程

1. **创建开发分支**
   ```bash
   git checkout -b task-2.2-kernel-optimization
   ```

2. **增量开发**
   - 每个优化独立实现
   - 逐步集成测试
   - 保持主线可用

3. **持续验证**
   - 每次修改后运行测试
   - 记录性能数据
   - 对比基线

4. **代码审查**
   - 自我审查代码质量
   - 验证注释完整性
   - 检查错误处理

### 代码组织

```
src/cuda/
├── cuda_kernels.cu              # 原始版本（保留）
├── cuda_kernels_optimized.cu   # 优化版本（新增）
└── kernels/
    ├── shared_memory.cuh        # 共享内存工具
    ├── muscl_optimized.cuh      # 优化MUSCL
    └── riemann_optimized.cuh    # 优化Riemann

include/
├── cuda_kernels.cuh             # 核函数接口
├── optimization_config.h         # 优化配置
└── block_tuner.h                # 块大小调优

src/utils/
└── block_tuner.cpp              # 块大小调优实现
```

### 版本控制策略

```bash
# 每个主要优化提交一次
git add src/cuda/cuda_kernels_optimized.cu
git commit -m "Optimize: Add shared memory version of update_cells_kernel (1.7× speedup)"

git add src/cuda/cuda_kernels_optimized.cu
git commit -m "Optimize: Improve MUSCL reconstruction (1.4× speedup)"

git add src/cuda/cuda_kernels_optimized.cu
git commit -m "Optimize: Optimize HLLC Riemann solver (1.3× speedup)"

# 最后完成提交
git commit -m "Task 2.2 Complete: Kernel optimization (2.8× total speedup)"
```

---

## 测试策略

### 正确性测试

**Level 1: 单元测试**
```bash
# 测试单个核函数
./test_kernels --test update_cells_optimized
./test_kernels --test muscl_optimized
./test_kernels --test hllc_optimized
```

**Level 2: 集成测试**
```bash
# 标准测试用例
./run_tests.sh

# 特定验证
./hydrosis --config test_dam_break.ini --validate
./hydrosis --config test_circular_dam.ini --validate
./hydrosis --config test_parabolic_bowl.ini --validate
```

**Level 3: 对比测试**
```bash
# 基线vs优化对比
python3 scripts/compare_solutions.py \
    results/baseline/output_0010.vtk \
    results/optimized/output_0010.vtk
```

### 性能测试

**基准测试套件:**
```bash
# 小规模（快速验证）
./scripts/benchmark.sh --sizes "100" --gpus "1" --cases "dam_break"

# 标准规模（详细分析）
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" \
    --cases "dam_break circular_dam parabolic_bowl"

# 大规模（扩展性）
./scripts/benchmark.sh --sizes "2000" --gpus "1" --cases "dam_break"

# 应用案例
./scripts/benchmark.sh --sizes "500" --gpus "1" \
    --cases "dam_break urban river"
```

**性能分析:**
```bash
# Nsight Systems
nsys profile --stats=true \
    --trace=cuda,nvtx \
    -o task_2.2_profile \
    ./hydrosis --config config.ini

# Nsight Compute（关键核函数）
ncu --set full \
    --kernel-name update_cells_kernel_optimized \
    -o task_2.2_kernel \
    ./hydrosis --config config.ini
```

### 回归测试

**自动化检查:**
```python
# scripts/regression_check.py
def check_performance_regression(baseline, current):
    """检查是否有性能回归"""
    speedup = baseline['time'] / current['time']
    if speedup < 0.95:  # 慢于5%视为回归
        raise RuntimeError(f"Performance regression detected: {speedup:.2f}×")
    return speedup

def check_correctness(baseline, current):
    """检查数值正确性"""
    rel_error = compute_relative_error(baseline, current)
    if rel_error > 1e-6:
        raise RuntimeError(f"Numerical error too large: {rel_error}")
    return rel_error
```

---

## 性能监控

### 关键指标

| 指标 | 测量方法 | 目标 | 实际 |
|------|----------|------|------|
| 加速比 | benchmark.sh | ≥2.5× | TBD |
| 内存带宽利用率 | Nsight Compute | >70% | TBD |
| GPU SM效率 | Nsight Compute | >80% | TBD |
| 占用率 | Nsight Compute | >50% | TBD |
| 质量守恒误差 | 验证测试 | <1e-10 | TBD |

### 进度跟踪

**Daily Progress Tracking:**
```markdown
## Day 1 Progress (2025-10-XX)
- [x] 设计共享内存布局
- [x] 实现基础版本
- [x] 编译测试通过
- [ ] 性能测试（待GPU硬件）
- 问题: None
- 下一步: 实现halo加载

## Day 2 Progress (2025-10-XX)
...
```

---

## 风险管理

### Risk #1: GPU硬件不可用

**影响:** 无法运行实际性能测试

**缓解措施:**
- 完成所有代码实现
- 编译检查无误
- 理论分析性能
- 准备在有硬件时快速验证

**状态:** 🟡 中风险

### Risk #2: 性能目标未达成

**影响:** 加速比 < 2.5×

**缓解措施:**
- 实际profiling识别真正瓶颈
- 多种优化策略并行
- 调整优先级聚焦高影响优化
- 必要时调整目标

**状态:** 🟡 中风险

### Risk #3: 优化破坏正确性

**影响:** 测试失败或精度损失

**缓解措施:**
- 每步小心验证
- 保留原始版本对比
- 详细的单元测试
- 使用Git分支隔离变更

**状态:** 🟢 低风险（已有完善测试）

### Risk #4: 代码复杂度增加

**影响:** 可维护性降低

**缓解措施:**
- 清晰的代码结构
- 详细注释
- 保留简单版本作为参考
- 配置开关控制优化级别

**状态:** 🟢 低风险

---

## 成功标准

### 必须达成 (Must Have)

- ✅ 所有测试用例通过
- ✅ 质量守恒 < 1e-10
- ✅ 加速比 ≥ 2.0× （最低要求）
- ✅ 无数值精度损失（相对误差 < 1e-6）

### 应当达成 (Should Have)

- ✅ 加速比 ≥ 2.5× （目标）
- ✅ GPU利用率 > 70%
- ✅ 内存减少 ≥ 10%
- ✅ 文档完整

### 期望达成 (Nice to Have)

- ✅ 加速比 ≥ 3.0× （最佳情况）
- ✅ GPU利用率 > 80%
- ✅ 自适应优化系统
- ✅ 详细性能分析报告

---

## 资源需求

### 硬件

- **必需:** NVIDIA GPU (Compute Capability ≥ 6.0)
- **推荐:** V100或A100用于性能测试
- **内存:** ≥ 8GB GPU内存

### 软件

- **必需:**
  - CUDA Toolkit ≥ 11.0
  - CMake ≥ 3.18
  - GCC/G++ ≥ 7.0

- **推荐:**
  - NVIDIA Nsight Systems
  - NVIDIA Nsight Compute
  - Python 3 + matplotlib

### 时间

- **最少:** 10天（快速实施）
- **标准:** 12天（正常进度）
- **最多:** 14天（含缓冲时间）

---

## 下一步行动

### 立即执行（有GPU时）

```bash
# 1. 创建开发分支
git checkout -b task-2.2-kernel-optimization

# 2. 建立性能基线
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output baseline.csv
nsys profile -o baseline ./hydrosis --config dam_break.ini

# 3. 开始优化实施
# 按照Day 1-14计划逐步执行

# 4. 持续测试验证
./run_tests.sh  # 每次修改后
```

### 当前可执行（无GPU时）

```bash
# 1. 准备代码框架
mkdir -p src/cuda/kernels
mkdir -p include/optimization

# 2. 编写优化版本
# - 共享内存版本
# - MUSCL优化版本
# - HLLC优化版本

# 3. 编译检查
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j  # 验证编译无误

# 4. 准备测试脚本
# - 性能测试自动化
# - 对比工具
# - 报告生成
```

---

## 参考文档

### 内部文档

- [BASELINE_ANALYSIS.md](BASELINE_ANALYSIS.md) - 瓶颈分析
- [OPTIMIZATION_STRATEGY.md](OPTIMIZATION_STRATEGY.md) - 优化策略
- [PERFORMANCE_ANALYSIS_GUIDE.md](PERFORMANCE_ANALYSIS_GUIDE.md) - 性能分析指南
- [PHASE2_PLAN.md](PHASE2_PLAN.md) - Phase 2总体计划

### 外部资源

**CUDA优化:**
- NVIDIA CUDA C++ Best Practices Guide
- CUDA Programming Guide - Shared Memory
- Mark Harris: "An Efficient Matrix Transpose in CUDA"

**Shallow Water GPU:**
- Brodtkorb et al. (2012): "Efficient GPU Implementation"
- Lacasta et al. (2014): "Optimized GPU Implementation"

**性能分析:**
- NVIDIA Nsight Systems Documentation
- NVIDIA Nsight Compute Documentation
- Williams et al.: "Roofline Model"

---

## 附录

### A. 每日检查清单

**每日开始:**
- [ ] 检查Git状态
- [ ] 拉取最新代码
- [ ] 查看前一天问题

**实施期间:**
- [ ] 增量提交代码
- [ ] 运行测试验证
- [ ] 记录性能数据
- [ ] 更新进度文档

**每日结束:**
- [ ] 提交当天工作
- [ ] 记录问题和进展
- [ ] 计划明天任务
- [ ] 备份重要数据

### B. 代码审查清单

**正确性:**
- [ ] 所有测试通过
- [ ] 数值精度验证
- [ ] 边界条件正确
- [ ] 无内存泄漏

**性能:**
- [ ] 加速比达标
- [ ] GPU利用率高
- [ ] 无性能回归
- [ ] Profiling验证

**代码质量:**
- [ ] 注释完整清晰
- [ ] 命名规范一致
- [ ] 无重复代码
- [ ] 错误处理完善

**可维护性:**
- [ ] 模块化良好
- [ ] 接口清晰
- [ ] 文档完整
- [ ] 示例代码

### C. 问题跟踪模板

```markdown
## Issue #1: [简短描述]

**发现时间:** 2025-10-XX Day X
**严重程度:** 🔴高 / 🟡中 / 🟢低
**状态:** 打开 / 进行中 / 已解决

**描述:**
[详细问题描述]

**重现步骤:**
1. ...
2. ...

**预期行为:**
[...]

**实际行为:**
[...]

**解决方案:**
[如何修复]

**验证:**
[如何验证修复]
```

---

**文档版本:** 1.0
**创建日期:** 2025-10-29
**状态:** 📋 计划完成，待实施
**预计开始:** 待GPU硬件可用
**预计完成:** 开始后10-14天

---

## 快速启动指令

```bash
# 当GPU硬件可用时，执行以下命令开始Task 2.2:

# 1. 切换到项目目录
cd /path/to/HydroSIS-2D

# 2. 创建开发分支
git checkout -b task-2.2-kernel-optimization

# 3. 运行基线测试
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output results/baseline.csv

# 4. 开始Day 1任务
# 参见上文"Day 1-2: 共享内存优化"

# 5. 持续跟踪进度
# 每天更新TASK_2.2_PROGRESS.md
```

---

**准备完成！等待GPU硬件开始实施。** 🚀
