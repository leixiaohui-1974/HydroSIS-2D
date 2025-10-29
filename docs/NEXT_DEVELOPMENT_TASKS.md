# HydroSIS-2D 下一步开发任务清单

**文档创建日期**: 2025-10-29
**基于版本**: 2.0 (前后处理工具包已完成)

---

## 执行摘要

HydroSIS-2D 前后处理工具包已经完成开发，包含 7 个完整模块、11,496 行代码和 145 个测试。本文档分析当前系统的优势与不足，提出下一阶段的开发任务，优先级从高到低排序。

### 当前项目状态分析

#### ✅ 已完成的功能

| 模块 | 完成度 | 代码行数 | 测试数 | 说明 |
|------|--------|----------|--------|------|
| 网格生成 | 100% | ~1,150 | 20 | 结构化网格、自适应加密 |
| 几何处理 | 100% | ~1,200 | 29 | 地形读取、处理、生成 |
| 边界条件 | 100% | ~825 | 34 | 6种BC类型 |
| 初始条件 | 100% | ~925 | 32 | 6种IC类型 |
| 模拟配置 | 100% | ~550 | 15 | 集成所有模块 |
| 3D可视化 | 60% | ~1,010 | 11 | 基础渲染功能 |
| 结果分析 | 70% | ~1,040 | 4 | 基础统计分析 |

**总计**: 44个文件，11,496行代码，145个测试

#### ❌ 缺失的关键功能

1. **核心GPU求解器** - 完全缺失，这是最关键的组件
2. **非结构化网格** - 未实现Gmsh集成
3. **GUI界面** - gui目录为空
4. **高级可视化** - 无流线、粒子追踪、体渲染
5. **实时监控** - 无求解器运行时监控
6. **求解器集成** - 前后处理与求解器未连接

---

## 优先级分级说明

- **🔥 P0 - 关键（Critical）**: 核心功能，必须实现才能使系统可用
- **⭐ P1 - 高（High）**: 重要功能，显著提升用户体验
- **📌 P2 - 中（Medium）**: 增强功能，锦上添花
- **💡 P3 - 低（Low）**: 可选功能，未来考虑

---

## 阶段 1：核心求解器开发 🔥

### 任务 1.1：2D浅水方程GPU求解器 (CUDA)

**优先级**: 🔥 P0 - 关键
**预计工时**: 6-8周
**前置条件**: 无
**技术栈**: CUDA C++, Thrust, CUB

#### 功能需求

1. **数值方法实现**
   - [ ] 有限体积法（FVM）求解器框架
   - [ ] 高阶格式（MUSCL重构，2阶精度）
   - [ ] HLL/HLLC Riemann求解器
   - [ ] 时间积分（显式Runge-Kutta 2/3阶）
   - [ ] 干湿边界处理（wet/dry front）
   - [ ] 源项处理（底坡、摩擦力）

2. **GPU加速实现**
   - [ ] CUDA核函数优化
   - [ ] 共享内存利用
   - [ ] 纹理内存用于地形数据
   - [ ] 多GPU支持（MPI + CUDA）
   - [ ] 动态负载平衡

3. **稳定性与精度**
   - [ ] CFL条件自适应时间步长
   - [ ] 正定性保持（positivity-preserving）
   - [ ] 质量守恒验证
   - [ ] 激波捕捉能力

#### 技术实现结构

```cpp
// src/solver/ShallowWaterSolver.cuh
class ShallowWaterSolver {
public:
    // 初始化
    void initialize(const MeshData& mesh, const Config& config);

    // 主求解循环
    void solve(double t_end);

    // 单步推进
    void step(double dt);

private:
    // GPU数据
    double *d_h, *d_u, *d_v;     // 水深、速度
    double *d_z;                  // 地形高程
    double *d_flux_x, *d_flux_y;  // 通量

    // CUDA核函数
    __global__ void computeFluxes();
    __global__ void updateConservativeVars();
    __global__ void computeTimestep();
    __global__ void applyBoundaryConditions();
};
```

#### 关键算法伪代码

```cpp
// 主求解循环
void ShallowWaterSolver::solve(double t_end) {
    double t = 0.0;
    int step = 0;

    while (t < t_end) {
        // 1. 计算时间步长（CFL条件）
        double dt = computeAdaptiveTimestep();

        // 2. 边界条件
        applyBoundaryConditions(t);

        // 3. 计算通量（HLL/HLLC）
        computeFluxes<<<grid, block>>>(d_h, d_u, d_v, d_flux_x, d_flux_y);

        // 4. 更新守恒变量
        updateConservativeVars<<<grid, block>>>(d_h, d_u, d_v,
                                                 d_flux_x, d_flux_y, dt);

        // 5. 源项处理
        applySourceTerms<<<grid, block>>>(d_h, d_u, d_v, d_z, dt);

        // 6. 输出结果
        if (step % output_interval == 0) {
            writeOutput(t, step);
        }

        t += dt;
        step++;
    }
}
```

#### 验证基准测试

- [ ] 1D溃坝问题（解析解对比）
- [ ] 2D圆形溃坝（径向对称性检验）
- [ ] MacDonald测试集（10个标准算例）
- [ ] 抛物面碗（解析解，质量守恒）
- [ ] 湖泊静止问题（C-property验证）

#### 性能目标

- 单GPU（RTX 3090）：1M cells，实时运行（≥1.0倍实时速度）
- 多GPU（4×V100）：10M cells，2-5倍实时速度
- 内存占用：< 8GB（1M cells）

#### 交付物

- [ ] CUDA求解器源代码（src/solver/）
- [ ] CMake构建系统
- [ ] 10+个验证算例及结果
- [ ] 性能测试报告
- [ ] API文档

---

### 任务 1.2：Python-CUDA接口

**优先级**: 🔥 P0 - 关键
**预计工时**: 2-3周
**前置条件**: 任务1.1
**技术栈**: pybind11, CUDA Python API

#### 功能需求

- [ ] Python绑定（pybind11）
- [ ] NumPy数组与CUDA内存互转
- [ ] 求解器配置Python API
- [ ] 实时进度回调
- [ ] 异常处理和错误报告

#### 技术实现

```python
# Python接口示例
import hydrosis2d

# 从前处理配置创建求解器
solver = hydrosis2d.Solver(
    config_file='output/dam_break/simulation_config.json'
)

# 设置回调函数
def progress_callback(time, step, dt):
    print(f"t={time:.2f}s, step={step}, dt={dt:.4f}s")

solver.set_callback(progress_callback, interval=100)

# 运行求解
solver.run(t_end=10.0)

# 获取结果
results = solver.get_results()
```

#### 交付物

- [ ] Python绑定模块（hydrosis2d.so）
- [ ] Python API文档
- [ ] 集成测试（Python调用CUDA求解器）

---

### 任务 1.3：前后处理与求解器集成

**优先级**: ⭐ P1 - 高
**预计工时**: 2周
**前置条件**: 任务1.1, 1.2

#### 功能需求

- [ ] 配置文件标准化（JSON schema）
- [ ] 自动网格转换（Python网格 → CUDA数据结构）
- [ ] 边界条件自动映射
- [ ] 初始条件自动加载
- [ ] 结果自动导出为VTK

#### 端到端工作流

```python
from simulation import create_dam_break_simulation
import hydrosis2d

# 1. 前处理
config = create_dam_break_simulation(nx=100, ny=50)
config.export_configuration('sim1')

# 2. 求解
solver = hydrosis2d.Solver('sim1/simulation_config.json')
solver.run(t_end=10.0)

# 3. 后处理
from postprocessing import ResultAnalyzer, VisualizationEngine

analyzer = ResultAnalyzer()
analyzer.load_vtk_series('sim1/results/*.vtk')

vis = VisualizationEngine()
result = analyzer.get_result_at_time(5.0)
vis.visualize_water_surface(result.mesh, result.depth, result.terrain)
vis.show()
```

---

## 阶段 2：高级功能扩展 ⭐

### 任务 2.1：非结构化网格支持

**优先级**: ⭐ P1 - 高
**预计工时**: 4-5周
**前置条件**: 任务1.1（求解器需扩展）
**技术栈**: Gmsh Python API, meshio

#### 功能需求

1. **Gmsh集成**
   - [ ] Python API调用Gmsh生成三角形网格
   - [ ] 边界条件几何标记
   - [ ] 网格质量控制（最小角度、单元尺寸）
   - [ ] 约束三角化（河道、建筑物）

2. **网格生成器扩展**
   - [ ] UnstructuredMeshGenerator类
   - [ ] 三角形单元支持
   - [ ] 边界层网格（refinement layers）
   - [ ] 网格转换工具（.msh → HydroSIS格式）

3. **求解器扩展**
   - [ ] 非结构化网格FVM求解器
   - [ ] 边缘数据结构（edge-based）
   - [ ] 梯度重构（Green-Gauss）

#### 代码示例

```python
from preprocessing.mesh_generation import UnstructuredMeshGenerator

# 创建非结构化网格生成器
gen = UnstructuredMeshGenerator()

# 添加几何约束
gen.add_polyline([(0,0), (100,0), (100,50), (0,50)], tag='domain')
gen.add_polyline([(30,0), (30,50)], tag='dam')

# 设置网格参数
gen.set_element_size(default=2.0, min=0.5, max=5.0)
gen.set_refinement_field_near_polyline('dam', distance=10.0, size=0.5)

# 生成网格
mesh = gen.generate(element_type='triangle')

print(f"生成了 {mesh.n_elements} 个三角形单元")
```

#### 交付物

- [ ] UnstructuredMeshGenerator类
- [ ] Gmsh集成模块
- [ ] 网格质量检查工具
- [ ] 10个非结构化网格算例
- [ ] 文档更新

---

### 任务 2.2：高级可视化功能

**优先级**: ⭐ P1 - 高
**预计工时**: 3-4周
**前置条件**: 无（可并行开发）
**技术栈**: PyVista, VTK

#### 功能需求

1. **流线与粒子追踪**
   - [ ] 流线计算（Runge-Kutta积分）
   - [ ] 流带（streaklines）
   - [ ] 拉格朗日粒子追踪
   - [ ] 路径线（pathlines）动画

2. **等值面与体渲染**
   - [ ] 水深等值线
   - [ ] 速度等值面
   - [ ] 3D等值面提取（Marching Cubes）
   - [ ] 透明度映射

3. **高级渲染效果**
   - [ ] 环境光遮蔽（SSAO）
   - [ ] 实时阴影
   - [ ] 镜面反射（水面）
   - [ ] 多光源照明

4. **交互式探索**
   - [ ] 探针工具（probe任意点）
   - [ ] 剖面切片（任意平面）
   - [ ] 体积选择和统计
   - [ ] 时间滑块控制

#### 代码示例

```python
from postprocessing.visualization_engine import VisualizationEngine
from postprocessing.advanced_vis import StreamlineGenerator, ParticleTracer

# 加载结果
vis = VisualizationEngine()
result = analyzer.get_result_at_time(5.0)

# 1. 绘制流线
streamlines = StreamlineGenerator()
streamlines.compute(
    mesh=result.mesh,
    velocity=(result.u, result.v),
    seed_points=[(25, 25), (50, 25), (75, 25)],
    max_length=100.0
)
vis.add_streamlines(streamlines, color='velocity', linewidth=2)

# 2. 粒子追踪
particles = ParticleTracer()
particles.initialize(n_particles=1000, region=(20, 30, 20, 30))
particles.trace(velocity_field=(result.u, result.v), dt=0.1, n_steps=100)
vis.add_particle_animation(particles, color='red', size=2)

# 3. 等值面
vis.add_isosurface(
    scalar=result.depth,
    isovalue=2.0,
    opacity=0.5,
    color='blue'
)

# 4. 体渲染
vis.add_volume_rendering(
    scalar=result.depth,
    opacity_mapping='linear',
    color_map='Blues'
)

vis.show()
```

#### 交付物

- [ ] 流线生成器模块
- [ ] 粒子追踪器模块
- [ ] 等值面提取工具
- [ ] 体渲染引擎
- [ ] 交互工具集
- [ ] 10+个高级可视化示例

---

### 任务 2.3：GUI界面开发

**优先级**: ⭐ P1 - 高
**预计工时**: 6-8周
**前置条件**: 任务1.3（求解器集成）
**技术栈**: PyQt6 或 Streamlit

#### 方案选择

**方案A：桌面GUI（PyQt6）**
- 优点：功能强大、响应迅速、离线运行
- 缺点：开发复杂、跨平台适配

**方案B：Web GUI（Streamlit）**
- 优点：快速开发、跨平台、易于部署
- 缺点：性能受限、实时性较差

**推荐**：优先开发Streamlit版本，后期考虑PyQt6专业版

#### 功能需求（Streamlit版本）

1. **前处理界面**
   - [ ] 域参数设置（交互式表单）
   - [ ] 网格配置（滑块选择nx, ny）
   - [ ] 地形上传/选择（文件上传）
   - [ ] 边界条件设置（下拉菜单+参数输入）
   - [ ] 初始条件选择（预设模板）
   - [ ] 实时网格预览（Plotly 3D）

2. **求解器控制**
   - [ ] 配置验证和导出
   - [ ] 启动/暂停/停止求解
   - [ ] 实时进度显示（进度条）
   - [ ] 日志输出窗口
   - [ ] 监控关键指标（最大水深、质量守恒等）

3. **后处理界面**
   - [ ] 结果文件浏览器
   - [ ] 时间步选择（滑块）
   - [ ] 变量选择（h, u, v, |V|）
   - [ ] 2D/3D可视化切换
   - [ ] 动画播放控制
   - [ ] 统计图表（Plotly）

4. **工作流管理**
   - [ ] 项目保存/加载
   - [ ] 案例库（模板案例）
   - [ ] 参数扫描（批量运行）
   - [ ] 结果对比

#### 界面原型代码

```python
# gui/streamlit_app.py
import streamlit as st
from simulation import SimulationConfig
import hydrosis2d

st.set_page_config(page_title="HydroSIS-2D", layout="wide")

# 侧边栏 - 导航
page = st.sidebar.selectbox(
    "选择页面",
    ["前处理", "求解器", "后处理", "案例库"]
)

if page == "前处理":
    st.title("🌊 HydroSIS-2D 前处理")

    # 1. 域设置
    st.header("1. 域参数")
    col1, col2 = st.columns(2)
    with col1:
        xmin = st.number_input("X最小值 (m)", value=0.0)
        ymin = st.number_input("Y最小值 (m)", value=0.0)
    with col2:
        xmax = st.number_input("X最大值 (m)", value=100.0)
        ymax = st.number_input("Y最大值 (m)", value=50.0)

    # 2. 网格设置
    st.header("2. 网格设置")
    nx = st.slider("X方向单元数", 10, 500, 100)
    ny = st.slider("Y方向单元数", 10, 500, 50)

    st.info(f"网格大小: {nx} × {ny} = {nx*ny:,} 个单元")

    # 3. 边界条件
    st.header("3. 边界条件")
    bc_west = st.selectbox("西边界", ["墙壁", "入流", "出流", "周期"])
    if bc_west == "入流":
        inflow_depth = st.number_input("入流水深 (m)", value=5.0)
        inflow_vel = st.number_input("入流速度 (m/s)", value=2.0)

    # 4. 初始条件
    st.header("4. 初始条件")
    ic_type = st.selectbox(
        "初始条件类型",
        ["溃坝", "均匀水深", "干床", "高斯隆起"]
    )

    # 5. 生成配置
    if st.button("🚀 生成配置", type="primary"):
        with st.spinner("生成配置中..."):
            config = SimulationConfig()
            config.set_domain_and_mesh(xmin, xmax, ymin, ymax, nx, ny)
            # ... 设置BC和IC

            is_valid, errors = config.validate()
            if is_valid:
                config.export_configuration('streamlit_output')
                st.success("✅ 配置生成成功！")
            else:
                st.error("❌ 配置验证失败")
                for err in errors:
                    st.error(err)

elif page == "求解器":
    st.title("⚙️ 求解器控制")

    config_file = st.file_uploader("上传配置文件", type=['json'])

    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("▶️ 启动求解"):
            # 启动求解器
            pass
    with col2:
        if st.button("⏸️ 暂停"):
            pass
    with col3:
        if st.button("⏹️ 停止"):
            pass

    # 实时进度
    progress_bar = st.progress(0)
    status_text = st.empty()

    # 监控图表
    st.header("实时监控")
    chart_placeholder = st.empty()

elif page == "后处理":
    st.title("📊 后处理与可视化")

    # 结果文件选择
    result_dir = st.text_input("结果目录", "output/results")

    # 时间步选择
    time = st.slider("时间 (s)", 0.0, 10.0, 5.0, 0.1)

    # 变量选择
    variable = st.selectbox("变量", ["水深", "速度U", "速度V", "速度大小"])

    # 可视化
    st.plotly_chart(create_2d_plot(), use_container_width=True)

# 运行: streamlit run gui/streamlit_app.py
```

#### 交付物

- [ ] Streamlit应用（gui/streamlit_app.py）
- [ ] 前处理界面（3个页面）
- [ ] 求解器控制界面
- [ ] 后处理界面（5个页面）
- [ ] 用户手册（含截图）

---

## 阶段 3：性能优化与扩展 📌

### 任务 3.1：多物理场耦合

**优先级**: 📌 P2 - 中
**预计工时**: 4-6周

#### 功能需求

- [ ] 溶质输运模块（污染物扩散）
- [ ] 泥沙输运模块（冲刷淤积）
- [ ] 植被阻力模型
- [ ] 降雨入渗模块

---

### 任务 3.2：并行计算扩展

**优先级**: 📌 P2 - 中
**预计工时**: 3-4周

#### 功能需求

- [ ] 多GPU支持（MPI + CUDA）
- [ ] 域分解（Domain decomposition）
- [ ] 动态负载平衡
- [ ] CPU版本（OpenMP，回退方案）

---

### 任务 3.3：第三方软件集成

**优先级**: 📌 P2 - 中
**预计工时**: 2-3周

#### 功能需求

- [ ] QGIS插件（地图集成）
- [ ] ParaView插件（高级后处理）
- [ ] Docker镜像（简化部署）
- [ ] Web服务API（RESTful）

---

## 阶段 4：生态系统建设 💡

### 任务 4.1：案例库与文档

**优先级**: 💡 P3 - 低
**预计工时**: 2-3周

#### 交付物

- [ ] 20+个标准案例
- [ ] 视频教程（10个）
- [ ] 中英文文档
- [ ] 在线文档站点

---

### 任务 4.2：社区与生态

**优先级**: 💡 P3 - 低
**预计工时**: 持续

#### 活动

- [ ] GitHub讨论区
- [ ] 月度线上研讨会
- [ ] 用户调查
- [ ] 插件生态（允许第三方扩展）

---

## 开发时间表（建议）

### 第1季度（3个月）- 核心求解器
- Week 1-8: 任务1.1 - CUDA求解器开发
- Week 9-11: 任务1.2 - Python接口
- Week 12: 任务1.3 - 集成测试

### 第2季度（3个月）- 高级功能
- Week 1-4: 任务2.1 - 非结构化网格（并行）
- Week 1-4: 任务2.2 - 高级可视化（并行）
- Week 5-12: 任务2.3 - GUI界面

### 第3季度（3个月）- 优化与扩展
- Week 1-6: 任务3.1 - 多物理场
- Week 7-10: 任务3.2 - 并行计算
- Week 11-12: 任务3.3 - 第三方集成

### 第4季度（3个月）- 生态建设
- Week 1-3: 任务4.1 - 文档与案例
- Week 4-12: 任务4.2 - 社区运营

---

## 资源需求

### 人力资源

- **CUDA开发工程师** × 1（全职）：核心求解器开发
- **Python开发工程师** × 1（全职）：前后处理、GUI
- **测试工程师** × 0.5（兼职）：测试与验证
- **文档工程师** × 0.5（兼职）：文档与教程

### 硬件资源

- **开发机**：RTX 3090 × 2 或 A6000 × 1
- **测试服务器**：V100 × 4 或 A100 × 2
- **CI/CD服务器**：标准配置

### 软件许可

- CUDA Toolkit（免费）
- PyQt6（LGPL，免费）或 Streamlit（Apache 2.0，免费）
- Gmsh（GPL，免费）
- ParaView（BSD，免费）

---

## 风险评估

### 技术风险

1. **GPU求解器性能不达标**
   - 缓解措施：早期性能测试，算法优化

2. **非结构化网格求解器数值稳定性**
   - 缓解措施：充分验证，采用成熟算法

3. **GUI响应性能问题**
   - 缓解措施：异步处理，进度回调

### 进度风险

1. **关键人员离职**
   - 缓解措施：代码文档完善，知识共享

2. **需求变更**
   - 缓解措施：敏捷开发，小步迭代

---

## 成功指标

### 技术指标

- [ ] GPU求解器性能达到实时（≥1.0x实时速度，1M cells）
- [ ] 10个基准测试全部通过（误差<1%）
- [ ] 测试覆盖率>80%
- [ ] 文档完整性100%

### 用户指标

- [ ] 用户能在30分钟内完成第一个模拟
- [ ] GUI界面满意度>4.0/5.0
- [ ] 月活跃用户>100（开源后6个月）

### 生态指标

- [ ] GitHub Stars > 500
- [ ] 第三方贡献者>5人
- [ ] 发表学术论文>2篇

---

## 总结

本开发计划以**核心求解器**为首要任务，逐步扩展到**高级功能**和**生态建设**。预计12个月完成核心功能开发，形成完整的端到端工作流。

### 下一步行动

1. ✅ **立即开始**：任务1.1 - CUDA求解器开发
2. 📅 **并行筹备**：组建开发团队，采购硬件资源
3. 📊 **定期评审**：每月进度回顾，风险评估

---

**文档版本**: 1.0
**创建日期**: 2025-10-29
**下次更新**: 根据开发进展更新

🤖 Generated with [Claude Code](https://claude.com/claude-code)
