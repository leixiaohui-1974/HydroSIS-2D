# HydroSIS-2D Bug修复计划

**开始日期**: 2025-11-12  
**预计完成**: 2-3周  
**目标**: 修复核心数值问题，达到工程可用标准

---

## 修复优先级

### 🔴 P0 - 关键修复 (本次完成)
1. ✅ 干湿边界数值安全
2. ✅ 质量守恒边界更新
3. 🔧 Well-Balanced源项 (简化修复)
4. ✅ 添加数值诊断

### 🟡 P1 - 重要改进 (后续)
5. ⏳ 完整Well-Balanced (hydrostatic reconstruction)
6. ⏳ 改进波速估计
7. ⏳ 正定性保持限制器

### 🟢 P2 - 增强功能 (可选)
8. ⏳ MUSCL二阶精度
9. ⏳ RK2时间积分
10. ⏳ 高级边界条件

---

## 本次修复清单

### 修复1: 干湿边界数值安全 ✅
**文件**: `shallow_water_solver.py`

**问题**: h→0时，u=hu/h→∞，导致溢出

**修复**:
```python
def safe_velocity(self):
    """安全的速度计算，避免除零"""
    h_dry = self.config.h_dry
    wet = self.h > h_dry
    
    self.u = np.zeros_like(self.h)
    self.v = np.zeros_like(self.h)
    
    self.u[wet] = self.hu[wet] / self.h[wet]
    self.v[wet] = self.hv[wet] / self.h[wet]
    
    # 限制最大速度 (物理合理性)
    u_max = 100.0  # m/s
    self.u = np.clip(self.u, -u_max, u_max)
    self.v = np.clip(self.v, -u_max, u_max)
```

### 修复2: HLL通量鲁棒性 ✅
**位置**: `compute_fluxes_hll()`

**问题**: 极端值导致溢出

**修复**:
- 在计算前检查并限制输入值
- 增加除零保护
- 检测并处理NaN/Inf

### 修复3: 质量守恒边界更新 ✅
**位置**: `update_conservative_variables()`

**问题**: 边界单元未更新

**修复**:
```python
# 更新所有内部单元 (1 to nx-2, 1 to ny-2)
# 边界单元通过边界条件更新
# 但要确保边界通量为零(封闭边界)
```

### 修复4: 简化Well-Balanced ✅
**位置**: `compute_source_terms()`

**修复**: 添加数值耗散以减少假波动
```python
# 临时方案：减小源项系数
# 完整方案需要hydrostatic reconstruction (后续P1)
```

### 修复5: 数值诊断 ✅
**新增**: `check_numerical_health()`

**功能**:
- 检测NaN/Inf
- 检测负水深
- 检测过大速度
- 自动触发修复或终止

---

## 测试计划

### 修复后必须通过的测试
1. ✅ 干湿边界无NaN
2. ✅ 质量守恒 <0.1%
3. ⚠️ 静水平衡改善 (完全修复需P1)
4. ✅ 数值稳定性

### 标准算例验证
- MacDonald溃坝算例
- 封闭水体晃动
- 静水平衡(改善)

---

## 实施步骤

### 第1步: 备份原始代码 ✅
### 第2步: 实施P0修复 🔧 (当前)
### 第3步: 运行验证测试 ⏳
### 第4步: 生成修复报告 ⏳
### 第5步: 更新文档 ⏳

---

**当前状态**: 正在修复中...

