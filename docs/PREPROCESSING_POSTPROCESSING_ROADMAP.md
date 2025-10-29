# HydroSIS-2D 前后处理程序开发路线图

## 项目概述

本文档规划了 HydroSIS-2D 前后处理程序的开发任务，旨在构建一个现代化、用户友好的前后处理平台，提升从几何建模到结果分析的完整工作流体验。

**开发目标**：
- 简化前处理流程，减少手动配置工作量
- 提供高级网格生成能力（支持非结构化网格）
- 增强可视化和渲染能力
- 提供GUI界面，降低使用门槛
- 构建一体化工作平台

---

## 任务清单

### 阶段 1：架构设计与技术选型

#### 任务 1.1：系统架构设计
**优先级**：⭐⭐⭐ 高
**预计工时**：1-2周

**目标**：
- 设计前后处理程序的整体架构
- 确定模块划分和接口定义
- 制定技术栈选型方案

**技术选型建议**：
- **编程语言**：Python（主要）+ C++（性能关键部分）
- **网格生成库**：Gmsh（Python API）、PyMesh、meshio
- **可视化库**：PyVista（基于VTK）、Matplotlib、Plotly
- **GUI框架**：Qt（PyQt6/PySide6）或 Streamlit（Web界面）
- **几何处理**：shapely、trimesh、gdal

**交付物**：
- [ ] 系统架构设计文档
- [ ] 技术选型报告
- [ ] 模块接口定义
- [ ] 依赖库清单

---

### 阶段 2：前处理模块开发

#### 任务 2.1：高级网格生成工具
**优先级**：⭐⭐⭐ 高
**预计工时**：3-4周

**功能需求**：

1. **结构化网格生成器增强**
   - [ ] 自适应网格加密（refinement）
   - [ ] 基于地形特征的局部加密
   - [ ] 多块网格支持
   - [ ] 网格质量检查与优化

2. **非结构化网格生成**（基于 Gmsh）
   - [ ] 三角形/四边形混合网格
   - [ ] 边界层网格生成
   - [ ] 基于Delaunay三角化
   - [ ] 约束三角化（考虑河道、建筑物等）

3. **网格转换工具**
   - [ ] 支持多种格式导入：Gmsh (.msh)、CGNS、STL
   - [ ] 网格格式转换器（非结构化 → 结构化）
   - [ ] 与HydroSIS-2D求解器的接口

**技术实现**：
```python
# 示例代码结构
class MeshGenerator:
    def __init__(self, domain_params):
        self.domain = domain_params

    def generate_structured_mesh(self, nx, ny, refinement_zones=None):
        """生成自适应结构化网格"""
        pass

    def generate_unstructured_mesh(self, element_type='triangle',
                                   max_element_size=1.0):
        """使用Gmsh生成非结构化网格"""
        import gmsh
        gmsh.initialize()
        # ... mesh generation logic
        gmsh.finalize()

    def refine_mesh_by_terrain(self, terrain_data, gradient_threshold):
        """基于地形梯度自动加密"""
        pass
```

**交付物**：
- [ ] 网格生成器Python模块
- [ ] 支持5+种网格类型
- [ ] 网格质量报告生成器
- [ ] 单元测试（覆盖率>80%）

---

#### 任务 2.2：几何建模与CAD导入
**优先级**：⭐⭐ 中
**预计工时**：2-3周

**功能需求**：

1. **交互式几何建模**
   - [ ] 2D域绘制（矩形、多边形、圆形）
   - [ ] 布尔运算（并、交、差）
   - [ ] 河道中心线绘制工具
   - [ ] 障碍物/建筑物放置

2. **CAD/GIS数据导入**
   - [ ] Shapefile (.shp) 导入（使用 GDAL）
   - [ ] DXF/DWG 导入
   - [ ] GeoTIFF 地形数据导入
   - [ ] OpenStreetMap 数据提取

3. **地形处理增强**
   - [ ] 多分辨率地形数据融合
   - [ ] 地形平滑/滤波
   - [ ] 河道提取算法
   - [ ] 高程数据插值（Kriging、RBF等）

**技术实现**：
```python
from osgeo import gdal, ogr
import shapely.geometry as geom

class GeometryProcessor:
    def import_shapefile(self, filepath):
        """导入Shapefile边界"""
        driver = ogr.GetDriverByName('ESRI Shapefile')
        dataset = driver.Open(filepath, 0)
        # ... process geometry

    def import_geotiff_terrain(self, filepath):
        """导入GeoTIFF地形"""
        dataset = gdal.Open(filepath)
        band = dataset.GetRasterBand(1)
        elevation = band.ReadAsArray()
        return elevation

    def extract_river_network(self, dem, accumulation_threshold):
        """从DEM提取河网"""
        # 使用D8算法或更先进的方法
        pass
```

**交付物**：
- [ ] 几何处理模块
- [ ] 支持Shapefile、GeoTIFF导入
- [ ] 河道提取工具
- [ ] 示例数据集

---

#### 任务 2.3：边界条件交互式设置
**优先级**：⭐⭐⭐ 高
**预计工时**：2周

**功能需求**：

1. **可视化边界条件编辑器**
   - [ ] 在网格上可视化选择边界
   - [ ] 支持4种边界类型（wall、open、inflow、outflow）
   - [ ] 时间序列边界条件（洪水过程线）
   - [ ] 空间变化边界条件

2. **初始条件设置**
   - [ ] 水深场初始化（均匀、分区、从文件读取）
   - [ ] 流速场初始化
   - [ ] 糙率场空间分布（基于土地利用类型）

3. **参数配置向导**
   - [ ] 分步配置向导（domain → mesh → physics → BC → IC）
   - [ ] 参数验证与合理性检查
   - [ ] 模板库（常见场景：溃坝、河道洪水、城市内涝等）

**技术实现**：
```python
class BoundaryConditionEditor:
    def __init__(self, mesh):
        self.mesh = mesh
        self.bc_zones = []

    def add_bc_zone(self, boundary_edges, bc_type, values):
        """添加边界条件区域"""
        zone = {
            'edges': boundary_edges,
            'type': bc_type,  # 'wall', 'open', 'inflow', 'outflow'
            'values': values  # {'h': 1.0, 'u': 0.0, 'v': 0.0}
        }
        self.bc_zones.append(zone)

    def load_time_series(self, filepath):
        """加载时间序列边界条件（洪水过程线）"""
        # Format: time, discharge
        pass

    def export_to_hydrosis_format(self, output_file):
        """导出为HydroSIS-2D格式"""
        pass
```

**交付物**：
- [ ] 边界条件编辑器模块
- [ ] 配置文件生成器
- [ ] 10+个场景模板
- [ ] 用户文档

---

### 阶段 3：后处理模块开发

#### 任务 3.1：高级3D可视化渲染引擎
**优先级**：⭐⭐⭐ 高
**预计工时**：3-4周

**功能需求**：

1. **3D场景渲染**（基于 PyVista）
   - [ ] 水深3D表面渲染（带高程）
   - [ ] 流速矢量场可视化
   - [ ] 等值线/等值面绘制
   - [ ] 颜色映射方案（科学配色：viridis、plasma等）

2. **高级可视化特性**
   - [ ] 体渲染（volume rendering）
   - [ ] 粒子追踪（Lagrangian tracer）
   - [ ] 流线/流带（streamlines/streaklines）
   - [ ] 剖面切片（任意平面）

3. **渲染质量增强**
   - [ ] 环境光遮蔽（SSAO）
   - [ ] 阴影效果
   - [ ] 透明度混合
   - [ ] 抗锯齿（FXAA/MSAA）

**技术实现**：
```python
import pyvista as pv
import numpy as np

class VisualizationEngine:
    def __init__(self):
        self.plotter = pv.Plotter()

    def render_water_surface_3d(self, mesh, h_field, z_field):
        """渲染水面3D表面"""
        # 创建水面高程 = 底床高程 + 水深
        water_surface = z_field + h_field

        # 创建结构化网格
        grid = pv.StructuredGrid()
        # ... set points based on water_surface

        # 添加到场景
        self.plotter.add_mesh(
            grid,
            scalars=h_field,
            cmap='viridis',
            opacity=0.8,
            lighting=True
        )

    def add_velocity_vectors(self, mesh, u_field, v_field):
        """添加流速矢量"""
        # ... create vector field
        self.plotter.add_arrows(...)

    def render_streamlines(self, mesh, velocity_field, seed_points):
        """渲染流线"""
        streamlines = mesh.streamlines(
            vectors='velocity',
            source_center=seed_points,
            max_time=100.0
        )
        self.plotter.add_mesh(streamlines, ...)

    def export_rendering(self, filepath, resolution=(1920, 1080)):
        """导出高分辨率图像"""
        self.plotter.screenshot(filepath, window_size=resolution)
```

**交付物**：
- [ ] 可视化引擎模块
- [ ] 10+种可视化类型
- [ ] 图像/视频导出功能
- [ ] 交互式3D查看器

---

#### 任务 3.2：实时结果监控与动画生成
**优先级**：⭐⭐ 中
**预计工时**：2-3周

**功能需求**：

1. **实时监控**
   - [ ] 模拟进度实时显示
   - [ ] 关键指标监控（最大水深、质量守恒等）
   - [ ] 探针点时间序列绘制
   - [ ] WebSocket实时数据推送

2. **高质量动画生成**
   - [ ] 多场变量联合动画
   - [ ] 时间戳/图例叠加
   - [ ] 多视角同步动画
   - [ ] 支持格式：MP4、GIF、WebM

3. **对比可视化**
   - [ ] 多个案例并排对比
   - [ ] 时间点对比（t1 vs t2）
   - [ ] 不同参数敏感性分析可视化

**技术实现**：
```python
import matplotlib.animation as animation
from matplotlib import pyplot as plt

class AnimationGenerator:
    def __init__(self, result_files):
        self.results = self.load_results(result_files)

    def create_flood_animation(self, variable='h', fps=20):
        """创建洪水演进动画"""
        fig, ax = plt.subplots(figsize=(12, 8))

        def update(frame):
            ax.clear()
            data = self.results[frame]
            im = ax.imshow(data[variable], cmap='Blues', ...)
            ax.set_title(f'Time: {self.results.time[frame]:.2f} s')
            return im,

        anim = animation.FuncAnimation(
            fig, update, frames=len(self.results),
            interval=1000/fps, blit=True
        )

        return anim

    def save_animation(self, anim, filepath):
        """保存为MP4"""
        writer = animation.FFMpegWriter(fps=20, bitrate=5000)
        anim.save(filepath, writer=writer)
```

**交付物**：
- [ ] 动画生成器模块
- [ ] 实时监控仪表盘
- [ ] 视频编码优化
- [ ] 示例动画库

---

#### 任务 3.3：数据分析与报告生成
**优先级**：⭐⭐ 中
**预计工时**：2周

**功能需求**：

1. **统计分析工具**
   - [ ] 时空统计（最大水深、最大流速、淹没历时）
   - [ ] 剖面提取与对比
   - [ ] 频率分析（淹没频率图）
   - [ ] 误差分析（与观测数据对比）

2. **工程量计算**
   - [ ] 淹没面积统计
   - [ ] 淹没体积计算
   - [ ] 风险区划分（基于水深阈值）
   - [ ] 经济损失评估（结合资产分布）

3. **自动化报告生成**
   - [ ] PDF报告模板（使用 ReportLab 或 LaTeX）
   - [ ] 图表自动嵌入
   - [ ] 关键指标汇总表
   - [ ] 结论与建议生成（基于规则）

**技术实现**：
```python
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, Paragraph

class ResultAnalyzer:
    def __init__(self, results):
        self.results = results

    def compute_inundation_statistics(self):
        """计算淹没统计"""
        stats = {
            'max_depth': np.max(self.results.h),
            'max_velocity': np.max(np.sqrt(self.results.u**2 + self.results.v**2)),
            'inundated_area': np.sum(self.results.h > 0.05) * self.results.dx * self.results.dy,
            'inundated_volume': np.sum(self.results.h) * self.results.dx * self.results.dy
        }
        return stats

    def extract_profile(self, start_point, end_point, num_points=100):
        """提取剖面数据"""
        # 沿线插值
        pass

    def generate_risk_map(self, depth_thresholds=[0.5, 1.0, 2.0]):
        """生成风险区划图"""
        risk_levels = np.zeros_like(self.results.max_h)
        for i, threshold in enumerate(depth_thresholds):
            risk_levels[self.results.max_h > threshold] = i + 1
        return risk_levels

class ReportGenerator:
    def create_pdf_report(self, stats, figures, output_file):
        """生成PDF报告"""
        doc = SimpleDocTemplate(output_file, pagesize=A4)
        story = []

        # 添加标题
        title = Paragraph("HydroSIS-2D Simulation Report", ...)
        story.append(title)

        # 添加统计表格
        data = [['Metric', 'Value'], ...]
        table = Table(data)
        story.append(table)

        # 添加图表
        # ...

        doc.build(story)
```

**交付物**：
- [ ] 结果分析模块
- [ ] 报告生成器
- [ ] 报告模板库
- [ ] 示例报告

---

### 阶段 4：GUI界面开发

#### 任务 4.1：集成工作流GUI
**优先级**：⭐⭐⭐ 高
**预计工时**：4-6周

**功能需求**：

1. **主界面布局**（基于Qt）
   - [ ] 项目管理器（新建/打开/保存项目）
   - [ ] 分模块工作区（前处理、求解、后处理）
   - [ ] 3D可视化窗口
   - [ ] 参数面板
   - [ ] 日志/消息窗口

2. **前处理工作流**
   - [ ] 几何导入/绘制界面
   - [ ] 网格生成向导
   - [ ] 边界条件可视化设置
   - [ ] 参数配置表单
   - [ ] 配置文件预览

3. **求解器集成**
   - [ ] 一键启动求解
   - [ ] 进度条与实时监控
   - [ ] GPU利用率显示
   - [ ] 日志实时输出

4. **后处理工作流**
   - [ ] 结果文件浏览器
   - [ ] 多变量选择与可视化
   - [ ] 动画播放器
   - [ ] 数据导出工具

**技术实现**：
```python
from PyQt6.QtWidgets import QMainWindow, QDockWidget, QTabWidget
from PyQt6.QtCore import QThread, pyqtSignal

class HydroSISGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        # 主窗口布局
        self.setWindowTitle('HydroSIS-2D Pre/Post Processor')

        # 创建菜单栏
        self.create_menus()

        # 创建工具栏
        self.create_toolbars()

        # 创建主工作区（Tab页）
        self.tabs = QTabWidget()
        self.tabs.addTab(self.create_preprocessing_tab(), "Pre-processing")
        self.tabs.addTab(self.create_solver_tab(), "Solver")
        self.tabs.addTab(self.create_postprocessing_tab(), "Post-processing")

        self.setCentralWidget(self.tabs)

        # 创建停靠窗口
        self.create_dock_widgets()

    def create_preprocessing_tab(self):
        """创建前处理标签页"""
        widget = QWidget()
        layout = QVBoxLayout()

        # 添加网格生成器界面
        mesh_group = QGroupBox("Mesh Generation")
        # ... add controls

        layout.addWidget(mesh_group)
        widget.setLayout(layout)
        return widget

class SolverThread(QThread):
    """求解器后台线程"""
    progress_update = pyqtSignal(int)
    log_message = pyqtSignal(str)

    def run(self):
        # 调用HydroSIS-2D求解器
        # 定期发射进度信号
        pass
```

**替代方案：Web界面**（基于Streamlit）
```python
import streamlit as st

def main():
    st.set_page_config(page_title="HydroSIS-2D", layout="wide")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select", ["Pre-processing", "Solver", "Post-processing"])

    if page == "Pre-processing":
        show_preprocessing_page()
    elif page == "Solver":
        show_solver_page()
    else:
        show_postprocessing_page()

def show_preprocessing_page():
    st.header("Mesh Generation")

    col1, col2 = st.columns(2)
    with col1:
        nx = st.number_input("Grid points (X)", min_value=10, value=256)
        ny = st.number_input("Grid points (Y)", min_value=10, value=128)

    if st.button("Generate Mesh"):
        # ... generate mesh
        st.success("Mesh generated successfully!")
```

**交付物**：
- [ ] 完整GUI应用程序（或Web应用）
- [ ] 用户手册（含截图）
- [ ] 安装程序/Docker镜像
- [ ] 教学视频

---

### 阶段 5：集成、优化与测试

#### 任务 5.1：性能优化
**优先级**：⭐⭐ 中
**预计工时**：1-2周

**优化方向**：
- [ ] 大规模网格处理优化（>1M cells）
- [ ] VTK文件读写加速（使用二进制格式）
- [ ] 可视化渲染加速（LOD、culling）
- [ ] 内存使用优化（数据流式处理）
- [ ] 多线程/多进程并行化

#### 任务 5.2：测试与验证
**优先级**：⭐⭐⭐ 高
**预计工时**：2周

**测试内容**：
- [ ] 单元测试（pytest，覆盖率>80%）
- [ ] 集成测试（端到端工作流）
- [ ] 性能测试（大规模案例）
- [ ] 用户验收测试（真实场景）
- [ ] 跨平台测试（Linux/Windows/macOS）

#### 任务 5.3：文档编写
**优先级**：⭐⭐ 中
**预计工时**：1-2周

**文档内容**：
- [ ] API文档（Sphinx）
- [ ] 用户手册（含教程）
- [ ] 开发者指南
- [ ] 最佳实践指南
- [ ] FAQ

---

## 技术栈总结

| 模块 | 推荐技术 | 替代方案 |
|------|---------|---------|
| **编程语言** | Python 3.9+ | C++ (性能关键部分) |
| **网格生成** | Gmsh, PyMesh, meshio | CGAL, Triangle |
| **几何处理** | Shapely, Trimesh | GDAL, Fiona |
| **可视化** | PyVista, Matplotlib | Mayavi, Plotly |
| **GUI框架** | PyQt6/PySide6 | Streamlit, Dash |
| **数据处理** | NumPy, Pandas | Xarray |
| **报告生成** | ReportLab, LaTeX | Jinja2 + WeasyPrint |
| **测试框架** | pytest, unittest | nose2 |

---

## 开发优先级

### Phase 1（必须，3个月）
1. ✅ 系统架构设计
2. ✅ 高级网格生成工具
3. ✅ 边界条件交互式设置
4. ✅ 高级3D可视化渲染

### Phase 2（重要，2个月）
5. ✅ 几何建模与CAD导入
6. ✅ 实时结果监控与动画生成
7. ✅ GUI界面（基础版）

### Phase 3（增强，2个月）
8. ✅ 数据分析与报告生成
9. ✅ 性能优化
10. ✅ 完整文档

---

## 关键依赖库安装

```bash
# 核心依赖
pip install numpy scipy pandas matplotlib
pip install pyvista meshio gmsh-sdk
pip install shapely fiona gdal
pip install PyQt6  # 或 streamlit

# 可视化增强
pip install plotly seaborn colorcet

# 报告生成
pip install reportlab jinja2

# 测试
pip install pytest pytest-cov

# 可选：Web界面
pip install streamlit streamlit-plotly-events
```

---

## 预期成果

完成后，用户将能够：

1. **前处理**：
   - 导入CAD/GIS数据，绘制复杂域几何
   - 生成自适应网格，自动加密关键区域
   - 可视化设置边界条件，无需手动编辑配置文件

2. **后处理**：
   - 生成高质量3D渲染图像和视频
   - 实时监控模拟进度
   - 自动生成专业分析报告

3. **一体化体验**：
   - 通过GUI完成全流程，降低使用门槛
   - 工作流自动化，提高效率
   - 与ParaView等专业工具无缝集成

---

## 参考资源

### 开源项目参考
- [pyHMT2D](https://github.com/psu-efd/pyHMT2D) - 2D水力模拟Python工具
- [PyVista Examples](https://docs.pyvista.org/examples/index.html) - 可视化示例
- [Gmsh Tutorials](https://gmsh.info/doc/texinfo/gmsh.html) - 网格生成教程

### 技术文档
- FHWA: Graphical Visualization Tools for 2D Hydraulic Modeling
- FLOW-3D POST Documentation
- ParaView User Guide

---

**文档版本**：v1.0
**创建日期**：2025-10-29
**作者**：HydroSIS-2D Development Team
