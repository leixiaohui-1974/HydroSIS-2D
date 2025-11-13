# HydroSIS-2D 产品开发路线图 2025-2026

**文档创建日期**: 2025-11-13
**版本**: 3.0
**基于**: 前后处理工具包完成版 + 国际商业软件对标分析

---

## 执行摘要

本路线图基于 HydroSIS-2D 当前开发现状，对标国际知名商业软件（MIKE FLOOD、HEC-RAS、InfoWorks ICM、TUFLOW、RiverFlow2D），提出下一阶段的开发任务。**特别强调全工作流测试**，不仅包括基础单元测试，更要进行端到端的完整工作流验证。

### 国际商业软件对标分析

| 软件 | 核心优势 | HydroSIS-2D 现状 | 差距 |
|------|---------|-----------------|------|
| **RiverFlow2D** | GPU加速（100x+）、非结构化网格、实时模拟 | ❌ 无GPU、✅ 结构化网格 | **关键差距** |
| **TUFLOW** | GPU/CPU双引擎、灵活网格、粒子追踪 | ❌ 仅CPU、❌ 无粒子追踪 | **重要差距** |
| **InfoWorks ICM** | Rain-on-mesh、云计算、15+水文方法 | ❌ 无rain-on-mesh、❌ 无云计算 | **功能差距** |
| **HEC-RAS** | Infra-mesh地形、1D/2D耦合、广泛应用 | ⚠️ 基础地形处理、❌ 无1D模块 | **中等差距** |
| **MIKE FLOOD** | 1D/2D动态耦合、多物理场、成熟生态 | ❌ 无1D、❌ 单物理场 | **生态差距** |
| **Iber+** | 开源、GPU加速、非结构化网格、免费 | ❌ 无GPU、✅ 结构化网格 | **开源竞品** |

### 核心竞争力定位

**目标**: 成为**开源、GPU加速、全功能**的2D水动力模拟平台

**差异化优势**（规划）:
1. 🚀 **性能**: GPU加速（目标100x+ CPU速度）—— 对标 RiverFlow2D
2. 🆓 **开源**: 完全免费、可定制 —— 对标 Iber+，超越商业软件
3. 🎯 **易用性**: Streamlit GUI + Python API —— 降低门槛
4. 🔬 **科研友好**: 完整文档、可扩展架构 —— 学术研究首选
5. ⚡ **现代技术栈**: CUDA + Python + Web技术 —— 新一代工具

---

## 第一阶段：GPU求解器与非结构化网格（6个月）🔥

### 任务 1.1：CUDA GPU求解器开发（P0 - 关键）

**时间**: 10-12周
**优先级**: 🔥🔥🔥 最高
**对标**: RiverFlow2D、TUFLOW GPU、Iber+

#### 1.1.1 核心数值方法

**要求**:
- [ ] **有限体积法（FVM）**：二维Cartesian网格
- [ ] **Riemann求解器**：
  - HLL求解器（基础，鲁棒性好）
  - HLLC求解器（高精度，推荐）
  - Roe求解器（可选，激波捕捉）
- [ ] **空间重构**：
  - 一阶迎风格式（基线）
  - **MUSCL二阶格式**（已有Python版本，需移植）
    - Minmod限制器
    - Van Leer限制器
    - Superbee限制器
    - MC限制器
- [ ] **时间积分**：
  - 显式Euler（一阶）
  - **RK2方法**（二阶，推荐）
  - RK3-TVD方法（三阶，可选）
- [ ] **干湿边界处理**：
  - h_dry阈值（如 1e-6 m）
  - 正定性保持（positivity-preserving）
  - 干湿界面通量修正
- [ ] **源项处理**：
  - 底坡源项：-gh∇z（well-balanced格式）
  - Manning摩擦：τ = ρgn²|V|V/h^(4/3)
  - 风应力（可选）
  - 科氏力（可选）

**对标说明**:
- HEC-RAS使用Infra-mesh技术在网格边界计算截面，HydroSIS应采用高精度重构（MUSCL）来弥补
- RiverFlow2D达到100x加速，HydroSIS目标至少50-100x加速

#### 1.1.2 GPU加速实现

**CUDA优化策略**:

```cpp
// 核心CUDA内核设计
__global__ void computeFluxes_HLLC(
    const double* __restrict__ h,      // 水深 [nx*ny]
    const double* __restrict__ u,      // x速度 [nx*ny]
    const double* __restrict__ v,      // y速度 [nx*ny]
    const double* __restrict__ z,      // 地形 [nx*ny]
    double* __restrict__ flux_x,       // x通量 [3*(nx+1)*ny]
    double* __restrict__ flux_y,       // y通量 [3*nx*(ny+1)]
    int nx, int ny, double dx, double dy
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < nx+1 && j < ny) {
        // MUSCL重构（利用共享内存）
        __shared__ double s_h[BLOCK_Y][BLOCK_X+4];
        __shared__ double s_u[BLOCK_Y][BLOCK_X+4];
        // ... 加载数据到共享内存（包括ghost cells）

        // 左右状态重构
        double hL, hR, uL, uR, vL, vR;
        muscl_reconstruct(&hL, &hR, s_h, tx, ty);

        // HLLC Riemann求解器
        double F_mass, F_mom_x, F_mom_y;
        hllc_solver(hL, uL, vL, hR, uR, vR,
                   &F_mass, &F_mom_x, &F_mom_y);

        // 写入全局内存（合并访问）
        int idx = i + j * (nx+1);
        flux_x[3*idx + 0] = F_mass;
        flux_x[3*idx + 1] = F_mom_x;
        flux_x[3*idx + 2] = F_mom_y;
    }
}
```

**优化技术清单**:
- [ ] **共享内存优化**：每个block加载halo区域，减少全局内存访问
- [ ] **合并访问**：保证内存访问模式coalesced（连续访问）
- [ ] **纹理内存**：地形数据使用texture memory（只读，有缓存）
- [ ] **常量内存**：物理参数（g, Manning系数等）放入constant memory
- [ ] **寄存器优化**：减少寄存器使用，提高occupancy
- [ ] **Stream并发**：多个CUDA stream并行处理（如通量计算+源项更新）
- [ ] **Pinned内存**：使用page-locked memory加速CPU-GPU传输
- [ ] **动态并行**：复杂边界条件使用dynamic parallelism

**性能目标**（对标RiverFlow2D）:

| 网格规模 | CPU时间（当前） | GPU目标 | 加速比 | 对标 |
|---------|----------------|---------|-------|------|
| 50k cells | ~5 min | <10 s | 30x | RiverFlow2D Basic |
| 200k cells | ~30 min | <30 s | 60x | TUFLOW GPU |
| 1M cells | ~3 hr | <2 min | 90x | RiverFlow2D Pro |
| 5M cells | N/A | <10 min | 100x+ | 商业软件顶级 |

**硬件配置**:
- 开发：RTX 3090 (24GB) 或 RTX 4090
- 测试：A100 (40GB/80GB) 或 V100
- 生产：支持消费级GPU（GTX 1660及以上）

#### 1.1.3 自适应时间步长与稳定性

- [ ] **CFL条件**：自动计算时间步长
  ```
  dt = CFL * min(dx, dy) / max(|u| + sqrt(gh), |v| + sqrt(gh))
  ```
  - CFL系数：0.5-0.9（可调）
  - 全局reduction找最小dt（高效GPU实现）

- [ ] **正定性保持**：确保h≥0, h+z单调
  - 限制器修正
  - 激波附近通量限制

- [ ] **质量守恒**：全局质量误差监控
  - 每100步检查：|M(t) - M(0)| / M(0) < 1e-8
  - 超阈值自动报警

#### 1.1.4 边界条件GPU实现

- [ ] **墙壁BC**：反射边界（镜像法）
- [ ] **入流BC**：固定h, u, v或Q
- [ ] **出流BC**：零梯度或临界深度
- [ ] **周期BC**：循环边界
- [ ] **透射BC**：特征线方法
- [ ] **时变BC**：支持时间序列插值

**GPU优化**：边界条件在专用kernel中并行处理

#### 1.1.5 验证基准测试（对标国际标准）

**必须通过的测试**（参考HEC-RAS、TUFLOW验证集）:

1. **1D溃坝（Ritter解析解）**
   - 初始：上游h=10m，下游h=1m
   - 验证：水深剖面与解析解误差<2%
   - 时间：t=0.5s, 1.0s, 2.0s

2. **2D圆形溃坝**
   - 初始：中心r<10m, h=10m；外部h=1m
   - 验证：径向对称性（偏差<1%）
   - 质量守恒：误差<1e-6

3. **MacDonald测试集**（10个标准算例）
   - Test 1: 平面斜坡流
   - Test 2: 下坡流（摩擦平衡）
   - Test 3: 干河床入流
   - Test 4: 阻塞流（转换流态）
   - Test 5: 梯形渠道流
   - ... (参考文献: MacDonald et al. 1997)

4. **Thacker抛物面碗**（解析解）
   - 验证：周期振荡，水位时程误差<5%
   - C-property：静止湖面48小时无假速度

5. **Lake at Rest**（well-balanced验证）
   - 复杂地形上静止水体
   - 验证：max(|u|, |v|) < 1e-10 m/s（机器精度）

6. **UK Environment Agency基准测试**
   - Test 8a: 平面洪水演进
   - Test 8b: 河道洪水
   - （对标：HEC-RAS、TUFLOW、MIKE21通过）

**通过标准**:
- 所有测试误差<5%（优秀）或<10%（及格）
- 质量守恒误差<1e-6
- 无数值不稳定（48小时运行）

#### 1.1.6 性能基准测试

**测试矩阵**:

| 测试案例 | 网格 | 模拟时间 | CPU基线 | GPU目标 | 加速比 |
|---------|------|---------|---------|---------|-------|
| 溃坝小尺度 | 100×50 | 10s | 30s | <1s | 30x+ |
| 溃坝中尺度 | 500×250 | 60s | 30min | <30s | 60x+ |
| 城市洪水 | 1000×1000 | 3600s | 10hr | <6min | 100x+ |
| 河流洪水 | 2000×500 | 7200s | 20hr | <10min | 120x+ |

**对标商业软件**:
- RiverFlow2D: 100-150x加速（RTX 3090）
- TUFLOW GPU: 50-100x加速
- HydroSIS-2D目标：50-150x加速（中等偏上）

#### 1.1.7 代码结构

```
src/solver/
├── cuda/
│   ├── kernels/
│   │   ├── flux_kernels.cu        # 通量计算
│   │   ├── update_kernels.cu      # 守恒变量更新
│   │   ├── source_kernels.cu      # 源项处理
│   │   ├── bc_kernels.cu          # 边界条件
│   │   ├── muscl_kernels.cu       # MUSCL重构
│   │   └── utils_kernels.cu       # 工具函数（CFL等）
│   ├── ShallowWaterSolver.cuh     # 求解器主类
│   ├── RiemannSolver.cuh          # Riemann求解器
│   └── DeviceMemory.cuh           # GPU内存管理
├── python/
│   ├── bindings.cpp               # pybind11绑定
│   └── solver_wrapper.py          # Python包装器
├── CMakeLists.txt
└── README.md
```

---

### 任务 1.2：非结构化网格支持（P0 - 关键）

**时间**: 6-8周
**优先级**: 🔥🔥 高
**对标**: TUFLOW、InfoWorks ICM、RiverFlow2D

#### 1.2.1 Gmsh集成与网格生成

**功能需求**:
- [ ] **Gmsh Python API集成**
  - 自动调用gmsh生成三角形/四边形网格
  - 支持.geo脚本和Python API两种方式

- [ ] **几何定义**
  - 导入GIS数据（Shapefile, GeoJSON）
  - 手动绘制多边形域
  - 河道中心线定义
  - 建筑物/障碍物几何

- [ ] **网格控制**
  - 全局单元尺寸：default_size
  - 局部加密：
    - 河道附近：size = 0.2 * default_size
    - 建筑物周围：size = 0.1 * default_size
    - 地形梯度大区域：自适应加密
  - 边界层网格：refinement layers

- [ ] **网格质量控制**
  - 最小角度：>20°（三角形）
  - 最大单元尺寸比：<3:1
  - 网格平滑：Laplacian smoothing

**代码示例**:

```python
from preprocessing.mesh_generation import UnstructuredMeshGenerator

# 创建生成器
gen = UnstructuredMeshGenerator()

# 方法1：从GIS导入
gen.import_domain_from_shapefile('domain.shp')
gen.import_river_from_shapefile('river_centerline.shp')
gen.import_buildings_from_shapefile('buildings.shp')

# 方法2：手动定义
gen.add_polygon([(0,0), (1000,0), (1000,500), (0,500)], tag='domain')
gen.add_polyline([(300,0), (300,500)], tag='dam', width=2.0)
gen.add_circle(center=(700, 250), radius=50, tag='building_1')

# 网格尺寸控制
gen.set_default_element_size(10.0)  # 默认10m
gen.set_refinement_near_tag('dam', distance=50, size=2.0)  # 坝附近2m
gen.set_refinement_near_tag('building_1', distance=20, size=1.0)

# 地形自适应加密
terrain = TerrainReader.read_ascii_grid('dem.asc')
gen.set_adaptive_refinement_from_terrain(
    terrain,
    slope_threshold=0.1,  # 坡度>10%加密
    size_factor=0.5
)

# 生成网格
mesh = gen.generate(
    element_type='triangle',  # 或 'quad'
    algorithm='delaunay',     # 或 'frontal'
    min_angle=25.0
)

print(f"生成 {mesh.n_elements} 个单元, {mesh.n_nodes} 个节点")
print(f"最小角度: {mesh.min_angle:.1f}°")
print(f"单元尺寸范围: {mesh.min_size:.2f} - {mesh.max_size:.2f} m")

# 导出
mesh.export('mesh.msh')  # Gmsh格式
mesh.export('mesh.vtu')  # VTK格式
mesh.export_to_hydrosis('mesh.h2d')  # HydroSIS格式
```

#### 1.2.2 非结构化FVM求解器

**数据结构**（Edge-based）:

```cpp
struct UnstructuredMesh {
    int n_cells;      // 单元数
    int n_edges;      // 边数
    int n_nodes;      // 节点数

    // 拓扑连接
    int* edge_to_cells;   // [n_edges, 2] 边的左右单元
    int* cell_to_edges;   // [n_cells, max_edges] 单元的边
    int* cell_to_nodes;   // [n_cells, max_nodes] 单元的节点

    // 几何信息
    double* cell_centers;  // [n_cells, 2] 单元中心
    double* cell_volumes;  // [n_cells] 单元面积
    double* edge_normals;  // [n_edges, 2] 边法向量
    double* edge_lengths;  // [n_edges] 边长度

    // 守恒变量
    double* h;  // [n_cells] 水深
    double* u;  // [n_cells] x速度
    double* v;  // [n_cells] y速度
    double* z;  // [n_cells] 地形高程
};
```

**GPU求解器扩展**:

```cpp
// 基于边的通量计算（适合非结构化网格）
__global__ void computeFluxes_EdgeBased(
    const UnstructuredMesh* mesh,
    const double* h,
    const double* u,
    const double* v,
    double* edge_fluxes  // [n_edges, 3]
) {
    int edge_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (edge_id < mesh->n_edges) {
        int cellL = mesh->edge_to_cells[2*edge_id + 0];
        int cellR = mesh->edge_to_cells[2*edge_id + 1];

        // 左右状态
        double hL = h[cellL], uL = u[cellL], vL = v[cellL];
        double hR = (cellR >= 0) ? h[cellR] : hL;  // 边界处理
        // ... MUSCL重构（梯度需预计算）

        // 法向速度
        double nx = mesh->edge_normals[2*edge_id + 0];
        double ny = mesh->edge_normals[2*edge_id + 1];
        double unL = uL*nx + vL*ny;
        // ... Riemann求解器

        // 写入边通量
        edge_fluxes[3*edge_id + 0] = F_mass;
        edge_fluxes[3*edge_id + 1] = F_mom_x;
        edge_fluxes[3*edge_id + 2] = F_mom_y;
    }
}

// 更新守恒变量（累加各边通量）
__global__ void updateConservativeVars_Unstructured(
    const UnstructuredMesh* mesh,
    const double* edge_fluxes,
    double* h, double* u, double* v,
    double dt
) {
    int cell_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (cell_id < mesh->n_cells) {
        double dh = 0.0, dhu = 0.0, dhv = 0.0;

        // 累加单元所有边的通量
        for (int i = 0; i < mesh->cell_n_edges[cell_id]; i++) {
            int edge_id = mesh->cell_to_edges[cell_id * MAX_EDGES + i];
            double sign = mesh->cell_edge_sign[cell_id * MAX_EDGES + i];
            double len = mesh->edge_lengths[edge_id];

            dh  += sign * len * edge_fluxes[3*edge_id + 0];
            dhu += sign * len * edge_fluxes[3*edge_id + 1];
            dhv += sign * len * edge_fluxes[3*edge_id + 2];
        }

        double area = mesh->cell_volumes[cell_id];
        h[cell_id] -= dt / area * dh;
        // ... 更新u, v
    }
}
```

**梯度重构**（MUSCL需要）:
- Green-Gauss方法（推荐）
- 最小二乘法（备选）

#### 1.2.3 网格质量检查与可视化

```python
from preprocessing.mesh_generation import MeshQualityChecker

checker = MeshQualityChecker()
report = checker.analyze(mesh)

print(report.summary())
# 输出:
# 总单元数: 12,458
# 最小角度: 28.3° (良好, >20°)
# 最大角度: 156.7° (可接受, <160°)
# 单元尺寸比: 2.8:1 (优秀, <3:1)
# 畸形单元数: 3 (0.02%)

# 可视化网格质量
vis = MeshVisualizer()
vis.plot_mesh_quality(mesh, metric='min_angle', cmap='RdYlGn')
vis.highlight_bad_elements(mesh, min_angle_threshold=20)
vis.show()
```

**对标说明**:
- TUFLOW支持灵活网格（Flexible Mesh）
- InfoWorks ICM使用三角形网格
- RiverFlow2D支持三角形+四边形混合
- HydroSIS目标：三角形为主，后期支持混合

---

### 任务 1.3：Python-CUDA接口与集成（P0 - 关键）

**时间**: 2-3周
**优先级**: 🔥 高
**对标**: 商业软件易用性

#### 1.3.1 pybind11绑定

```cpp
// src/solver/python/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "../cuda/ShallowWaterSolver.cuh"

namespace py = pybind11;

PYBIND11_MODULE(hydrosis2d_cuda, m) {
    m.doc() = "HydroSIS-2D GPU-accelerated solver";

    py::class_<ShallowWaterSolver>(m, "Solver")
        .def(py::init<>())
        .def("initialize", &ShallowWaterSolver::initialize,
             py::arg("config_file"),
             "Initialize solver from JSON configuration")
        .def("run", &ShallowWaterSolver::run,
             py::arg("t_end"),
             py::arg("output_interval") = 1.0,
             "Run simulation until t_end")
        .def("step", &ShallowWaterSolver::step,
             py::arg("dt"),
             "Single time step")
        .def("get_results", &ShallowWaterSolver::getResults,
             "Get current solution as NumPy arrays")
        .def("set_callback", &ShallowWaterSolver::setCallback,
             py::arg("callback"),
             py::arg("interval") = 100,
             "Set progress callback function")
        .def("export_vtk", &ShallowWaterSolver::exportVTK,
             py::arg("filename"),
             "Export current state to VTK file");

    // GPU设备管理
    m.def("get_gpu_count", &getGPUCount, "Get number of CUDA devices");
    m.def("set_gpu", &setGPU, py::arg("device_id"), "Set active GPU");
    m.def("get_gpu_memory", &getGPUMemory, "Get GPU memory usage");
}
```

#### 1.3.2 Python包装器

```python
# prepost/solver/gpu_solver.py
import hydrosis2d_cuda
import numpy as np
from pathlib import Path

class GPUSolver:
    """HydroSIS-2D GPU求解器Python封装"""

    def __init__(self, config_file: str, gpu_id: int = 0):
        """
        初始化GPU求解器

        参数:
            config_file: 配置文件路径（JSON格式）
            gpu_id: GPU设备ID（默认0）
        """
        # 检查GPU
        n_gpus = hydrosis2d_cuda.get_gpu_count()
        if n_gpus == 0:
            raise RuntimeError("未检测到CUDA设备")

        print(f"检测到 {n_gpus} 个GPU设备")
        hydrosis2d_cuda.set_gpu(gpu_id)
        print(f"使用GPU {gpu_id}")

        # 创建求解器
        self.solver = hydrosis2d_cuda.Solver()
        self.solver.initialize(config_file)

        self.current_time = 0.0
        self.step_count = 0

    def run(self, t_end: float, output_dir: str = 'output',
            output_interval: float = 1.0,
            callback = None, callback_interval: int = 100):
        """
        运行模拟

        参数:
            t_end: 终止时间（秒）
            output_dir: 输出目录
            output_interval: 输出间隔（秒）
            callback: 进度回调函数 callback(time, step, dt)
            callback_interval: 回调间隔（步数）
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # 设置回调
        if callback:
            self.solver.set_callback(callback, callback_interval)
        else:
            # 默认回调：打印进度
            def default_callback(t, step, dt):
                mem = hydrosis2d_cuda.get_gpu_memory()
                print(f"[{step:6d}] t={t:8.2f}s, dt={dt:.4f}s, "
                      f"GPU内存: {mem['used']:.1f}/{mem['total']:.1f} GB")
            self.solver.set_callback(default_callback, callback_interval)

        # 运行
        print(f"开始模拟: t=0 -> {t_end}s")
        import time
        t_start = time.time()

        self.solver.run(t_end, output_interval)

        t_elapsed = time.time() - t_start
        speedup = t_end / t_elapsed
        print(f"\n模拟完成!")
        print(f"  墙钟时间: {t_elapsed:.2f}s")
        print(f"  实时速度: {speedup:.2f}x")

        # 输出最终结果
        self.export_results(f"{output_dir}/final.vtk")

    def get_solution(self):
        """获取当前解（NumPy数组）"""
        results = self.solver.get_results()
        return {
            'time': results['time'],
            'h': results['h'],
            'u': results['u'],
            'v': results['v'],
            'z': results['z'],
            'mesh': results['mesh']
        }

    def export_results(self, filename: str):
        """导出VTK文件"""
        self.solver.export_vtk(filename)
        print(f"结果已导出: {filename}")

# 使用示例
if __name__ == '__main__':
    # 创建求解器
    solver = GPUSolver('config/dam_break.json', gpu_id=0)

    # 自定义回调
    def my_callback(t, step, dt):
        if step % 500 == 0:
            solution = solver.get_solution()
            print(f"  最大水深: {solution['h'].max():.2f} m")

    # 运行
    solver.run(t_end=10.0, output_interval=0.5, callback=my_callback)
```

#### 1.3.3 端到端工作流集成

```python
# 完整工作流示例
from preprocessing import (
    MeshGenerator, TerrainReader, GeometryGenerator,
    BoundaryConditionManager, InitialConditionManager
)
from simulation import SimulationConfig
from solver import GPUSolver
from postprocessing import ResultAnalyzer, VisualizationEngine

# ========== 1. 前处理 ==========
print("=== 前处理 ===")

# 1.1 创建网格
mesh_gen = MeshGenerator()
mesh = mesh_gen.create_uniform_mesh(
    xmin=0, xmax=200, ymin=0, ymax=100,
    nx=400, ny=200  # 80,000单元（GPU可轻松处理）
)

# 1.2 地形
terrain_gen = GeometryGenerator()
terrain = terrain_gen.create_tilted_plane(
    mesh, slope_x=0.01, base_elevation=0.0
)

# 1.3 边界条件
bc_manager = BoundaryConditionManager()
bc_manager.add_wall_bc('north')
bc_manager.add_wall_bc('south')
bc_manager.add_inflow_bc('west', depth=5.0, velocity_x=2.0, velocity_y=0.0)
bc_manager.add_outflow_bc('east', bc_type='zero_gradient')

# 1.4 初始条件
ic_manager = InitialConditionManager()
ic_manager.set_dam_break(
    mesh, dam_position_x=100.0,
    upstream_depth=10.0, downstream_depth=1.0
)

# 1.5 求解器配置
config = SimulationConfig()
config.set_domain_and_mesh(mesh)
config.set_terrain(terrain)
config.set_boundary_conditions(bc_manager)
config.set_initial_conditions(ic_manager)
config.set_solver_params(
    solver_type='cuda_gpu',  # 使用GPU求解器
    riemann_solver='hllc',
    spatial_order=2,  # MUSCL二阶
    time_integrator='rk2',
    cfl=0.8,
    manning_n=0.025,
    dry_threshold=1e-6
)
config.set_output_params(
    output_format='vtk',
    output_interval=0.5,
    variables=['h', 'u', 'v', 'velocity_magnitude']
)

# 验证并导出
is_valid, errors = config.validate()
if not is_valid:
    raise ValueError(f"配置错误: {errors}")

config.export_configuration('output/dam_break_gpu')
print("✓ 前处理完成\n")

# ========== 2. GPU求解 ==========
print("=== GPU求解 ===")

solver = GPUSolver('output/dam_break_gpu/simulation_config.json', gpu_id=0)
solver.run(
    t_end=20.0,
    output_dir='output/dam_break_gpu/results',
    output_interval=0.5
)
print("✓ 求解完成\n")

# ========== 3. 后处理 ==========
print("=== 后处理 ===")

# 3.1 加载结果
analyzer = ResultAnalyzer()
analyzer.load_vtk_series('output/dam_break_gpu/results/*.vtk')

# 3.2 质量守恒检查
mass_error = analyzer.check_mass_conservation()
print(f"质量守恒误差: {mass_error:.2e}")

# 3.3 生成3D可视化
vis = VisualizationEngine()
result_10s = analyzer.get_result_at_time(10.0)
vis.visualize_water_surface(
    result_10s.mesh,
    result_10s.depth,
    result_10s.terrain,
    velocity=(result_10s.u, result_10s.v),
    show_vectors=True
)
vis.save_screenshot('output/visualization_10s.png')
vis.show()

# 3.4 生成动画
from postprocessing.animation import AnimationGenerator
anim = AnimationGenerator()
anim.create_animation(
    vtk_files='output/dam_break_gpu/results/*.vtk',
    output_file='output/dam_break_animation.mp4',
    fps=30,
    variable='depth'
)
print("✓ 后处理完成\n")

print("=== 全工作流完成! ===")
```

---

## 第二阶段：高级可视化与GUI（3个月）⭐

### 任务 2.1：高级可视化功能（P1 - 高）

**时间**: 3-4周
**对标**: TUFLOW、ParaView集成

#### 2.1.1 流线与粒子追踪

- [ ] **流线计算**
  - Runge-Kutta 4阶积分
  - 自适应步长
  - 多起点并行计算

- [ ] **粒子追踪**（对标TUFLOW粒子模块）
  - 拉格朗日粒子
  - 时间积分：RK4
  - 支持1000+粒子实时追踪

- [ ] **路径线动画**
  - 时间演化可视化
  - 轨迹渲染

**代码示例**:

```python
from postprocessing.advanced_vis import StreamlineGenerator, ParticleTracer

# 流线
streamlines = StreamlineGenerator()
streamlines.compute(
    velocity_field=(u, v),
    seed_points=[(25, 25), (50, 25), (75, 25)],
    max_length=200.0,
    integration_method='rk4'
)
vis.add_streamlines(streamlines, color_by='velocity', linewidth=2)

# 粒子追踪
particles = ParticleTracer()
particles.initialize(n_particles=1000, region=(20, 30, 20, 30))
for t in time_series:
    particles.advect(velocity_field, dt=0.1)
    vis.add_particles(particles, color='red', size=3)
```

#### 2.1.2 等值面与体渲染

- [ ] 等值线/等值面提取（Marching Squares/Cubes）
- [ ] 体渲染（Volume Rendering）
- [ ] 透明度映射

#### 2.1.3 交互式探索工具

- [ ] 探针工具（点击查询数值）
- [ ] 剖面切片（任意平面）
- [ ] 时间滑块
- [ ] 体积选择和统计

**对标**:
- TUFLOW: 内置流线追踪
- HEC-RAS: 与ParaView/Tecplot集成
- HydroSIS: 原生支持高级可视化

---

### 任务 2.2：Streamlit Web GUI（P1 - 高）

**时间**: 6-8周
**对标**: 商业软件易用性，降低使用门槛

#### 2.2.1 前处理界面

**页面1：域和网格设置**
- 交互式参数输入（滑块、数字框）
- 实时网格预览（Plotly 3D）
- 网格统计显示

**页面2：地形设置**
- 文件上传（ASCII Grid, GeoTIFF）
- 合成地形生成器
- 3D地形预览

**页面3：边界条件**
- 每条边界独立配置
- 时变BC时间序列编辑器
- BC可视化检查

**页面4：初始条件**
- 预设模板选择
- 自定义参数调整
- 初始状态预览

**页面5：求解器参数**
- 数值方案选择（下拉菜单）
- 物理参数输入
- 输出设置

#### 2.2.2 求解器控制界面

- 启动/暂停/停止按钮
- 实时进度条
- 日志窗口（滚动输出）
- 关键指标监控图表（Plotly实时更新）
  - 最大水深时程
  - 质量守恒误差
  - 时间步长变化
  - GPU利用率

#### 2.2.3 后处理界面

**页面1：结果浏览**
- 文件树（所有VTK文件）
- 时间步选择（滑块+播放按钮）
- 变量选择（多选下拉）

**页面2：2D可视化**
- Plotly交互式2D图（平面色图）
- 颜色映射选择
- 等值线叠加

**页面3：3D可视化**
- PyVista集成（3D水面+地形）
- 相机控制
- 截图导出

**页面4：统计分析**
- 时间序列图表
- 剖面提取
- 质量守恒报告

**页面5：动画生成**
- 参数设置（FPS, 分辨率）
- 进度显示
- 下载链接

#### 2.2.4 案例库

- 预设案例模板（10+个）
- 一键加载
- 参数修改
- 复制/保存为新案例

#### 2.2.5 部署

```bash
# 本地运行
streamlit run gui/streamlit_app.py

# Docker部署
docker build -t hydrosis2d-web .
docker run -p 8501:8501 --gpus all hydrosis2d-web

# 云部署（可选）
# - Streamlit Cloud
# - AWS/Azure/GCP
```

**对标说明**:
- 商业软件GUI复杂但功能强大（HEC-RAS, MIKE）
- HydroSIS采用Web技术，降低安装门槛
- 目标：30分钟完成第一个模拟（新用户）

---

## 第三阶段：全工作流测试体系（贯穿始终）🧪

**重要性**: 🔥🔥🔥 **最高优先级**
**测试哲学**: "测试驱动开发" + "持续集成"

### 3.1 测试分层架构

```
测试金字塔（由下到上）:

    /\
   /  \        E2E测试（5%）
  /----\       ├─ 完整工作流测试（10个案例）
 /      \      └─ 性能回归测试
/--------\
|        |     集成测试（15%)
|        |     ├─ 前处理→求解器集成
|        |     ├─ 求解器→后处理集成
|________|     └─ 端到端数据流
  |    |
  |    |       单元测试（80%）
  |    |       ├─ 数值方法测试
  |    |       ├─ GPU kernel测试
  |____|       └─ 模块功能测试
```

### 3.2 单元测试（已有145个，需扩展）

**新增测试**:

#### 3.2.1 GPU Kernel测试

```python
# prepost/tests/test_cuda_kernels.py
import pytest
import numpy as np
from solver.gpu_solver import GPUSolver

class TestCUDAKernels:
    """GPU核函数单元测试"""

    def test_hllc_flux_accuracy(self):
        """测试HLLC通量计算精度"""
        # 左右状态（Riemann问题）
        hL, uL, vL = 10.0, 0.0, 0.0
        hR, uR, vR = 1.0, 0.0, 0.0

        # GPU计算
        flux_gpu = compute_flux_gpu(hL, uL, vL, hR, uR, vR)

        # CPU参考解
        flux_cpu = compute_flux_cpu_reference(hL, uL, vL, hR, uR, vR)

        # 验证（相对误差<1e-10）
        np.testing.assert_allclose(flux_gpu, flux_cpu, rtol=1e-10)

    def test_muscl_reconstruction(self):
        """测试MUSCL重构"""
        # 简单线性函数：h = 2*x + 5
        h = np.array([5.0, 7.0, 9.0, 11.0, 13.0])

        # 重构中间单元的左右值
        hL, hR = muscl_reconstruct_gpu(h, cell_idx=2, limiter='minmod')

        # 理论值（线性函数应精确重构）
        expected_hL = 9.0 - 1.0  # h_i - dh/2
        expected_hR = 9.0 + 1.0  # h_i + dh/2

        assert abs(hL - expected_hL) < 1e-12
        assert abs(hR - expected_hR) < 1e-12

    def test_cfl_computation(self):
        """测试CFL时间步长计算"""
        h = np.random.rand(100, 50) * 10  # 随机水深0-10m
        u = np.random.rand(100, 50) * 2 - 1  # 随机速度-1到1 m/s
        v = np.random.rand(100, 50) * 2 - 1

        dt_gpu = compute_timestep_gpu(h, u, v, dx=1.0, dy=1.0, cfl=0.8)
        dt_cpu = compute_timestep_cpu(h, u, v, dx=1.0, dy=1.0, cfl=0.8)

        assert abs(dt_gpu - dt_cpu) < 1e-8

    def test_dry_wet_handling(self):
        """测试干湿边界处理"""
        # 左侧干（h=1e-8），右侧湿（h=5.0）
        hL, uL = 1e-8, 0.0
        hR, uR = 5.0, 2.0

        flux = compute_flux_gpu(hL, uL, 0, hR, uR, 0)

        # 验证：干单元不应产生通量
        assert abs(flux[0]) < 1e-10  # 质量通量≈0

    def test_gpu_cpu_consistency(self):
        """测试GPU与CPU结果一致性"""
        # 相同输入
        h_init = create_dam_break_initial_condition(nx=100, ny=50)

        # GPU求解10步
        solver_gpu = GPUSolver(use_gpu=True)
        solver_gpu.initialize(h_init)
        for _ in range(10):
            solver_gpu.step(dt=0.01)
        h_gpu = solver_gpu.get_solution()['h']

        # CPU求解10步
        solver_cpu = CPUSolver()
        solver_cpu.initialize(h_init)
        for _ in range(10):
            solver_cpu.step(dt=0.01)
        h_cpu = solver_cpu.get_solution()['h']

        # 验证（GPU与CPU误差<1e-6，允许浮点舍入）
        np.testing.assert_allclose(h_gpu, h_cpu, rtol=1e-6, atol=1e-8)
```

#### 3.2.2 非结构化网格测试

```python
# prepost/tests/test_unstructured_mesh.py
class TestUnstructuredMesh:
    """非结构化网格测试"""

    def test_gmsh_integration(self):
        """测试Gmsh集成"""
        gen = UnstructuredMeshGenerator()
        gen.add_polygon([(0,0), (10,0), (10,10), (0,10)])
        mesh = gen.generate(element_type='triangle')

        assert mesh.n_elements > 0
        assert mesh.n_nodes > 0
        assert mesh.min_angle > 15  # 最小角度>15°

    def test_mesh_quality(self):
        """测试网格质量"""
        mesh = load_mesh('tests/data/test_mesh.msh')
        checker = MeshQualityChecker()
        report = checker.analyze(mesh)

        assert report.min_angle > 20  # 角度>20°
        assert report.max_aspect_ratio < 5  # 长宽比<5
        assert report.n_bad_elements == 0

    def test_adaptive_refinement(self):
        """测试自适应加密"""
        terrain = create_synthetic_terrain_with_valley()
        gen = UnstructuredMeshGenerator()
        gen.set_adaptive_refinement_from_terrain(terrain, slope_threshold=0.1)
        mesh = gen.generate()

        # 验证：高坡度区域单元更小
        steep_region_size = mesh.get_avg_size_in_region(x=50, y=50, radius=10)
        flat_region_size = mesh.get_avg_size_in_region(x=150, y=150, radius=10)
        assert steep_region_size < 0.5 * flat_region_size
```

### 3.3 集成测试（新增20+个）

**目标**: 测试模块间接口和数据流

```python
# prepost/tests/test_integration.py
class TestIntegration:
    """集成测试套件"""

    def test_preprocessing_to_solver(self):
        """测试前处理→求解器数据传递"""
        # 前处理
        config = create_dam_break_config(nx=50, ny=25)
        config.export_configuration('temp/integration_test')

        # 加载到求解器
        solver = GPUSolver('temp/integration_test/simulation_config.json')

        # 验证：网格尺寸正确
        assert solver.mesh.nx == 50
        assert solver.mesh.ny == 25

        # 验证：初始条件加载正确
        h_init = solver.get_solution()['h']
        assert h_init.max() > 9.0  # 上游水深≈10m
        assert h_init.min() < 2.0  # 下游水深≈1m

    def test_solver_to_postprocessing(self):
        """测试求解器→后处理数据流"""
        # 求解
        solver = GPUSolver('tests/data/simple_case.json')
        solver.run(t_end=1.0, output_dir='temp/solver_output')

        # 后处理加载
        analyzer = ResultAnalyzer()
        analyzer.load_vtk_series('temp/solver_output/*.vtk')

        # 验证：时间序列正确
        times = analyzer.get_time_series()
        assert len(times) > 0
        assert times[-1] >= 1.0

        # 验证：数据完整
        result = analyzer.get_result_at_time(0.5)
        assert result.mesh is not None
        assert result.depth.shape == (50, 25)

    def test_boundary_condition_application(self):
        """测试边界条件应用"""
        # 创建入流BC
        config = SimulationConfig()
        bc_manager = BoundaryConditionManager()
        bc_manager.add_inflow_bc('west', depth=5.0, velocity_x=2.0)
        config.set_boundary_conditions(bc_manager)

        # 求解器初始化
        solver = GPUSolver(config)
        solver.step(dt=0.01)

        # 验证：西边界单元有入流
        u = solver.get_solution()['u']
        assert u[0, :].mean() > 1.5  # x速度≈2 m/s

    def test_mass_conservation_integration(self):
        """测试质量守恒（集成）"""
        solver = GPUSolver('tests/data/closed_domain.json')

        # 记录初始质量
        h0 = solver.get_solution()['h']
        mass0 = h0.sum() * solver.mesh.dx * solver.mesh.dy

        # 运行100步
        for _ in range(100):
            solver.step(dt=0.01)

        # 最终质量
        h1 = solver.get_solution()['h']
        mass1 = h1.sum() * solver.mesh.dx * solver.mesh.dy

        # 验证：误差<1e-8
        error = abs(mass1 - mass0) / mass0
        assert error < 1e-8
```

### 3.4 端到端（E2E）测试 🎯

**最重要的测试层级** - 模拟真实用户工作流

#### 3.4.1 完整工作流测试案例

**测试案例1：溃坝模拟全流程**

```python
# prepost/tests/test_e2e_dam_break.py
import pytest
from pathlib import Path

class TestE2E_DamBreak:
    """端到端测试：溃坝完整工作流"""

    @pytest.fixture
    def output_dir(self, tmp_path):
        """临时输出目录"""
        return tmp_path / "e2e_dam_break"

    def test_complete_dam_break_workflow(self, output_dir):
        """
        完整工作流测试：
        1. 前处理（网格、地形、BC、IC）
        2. GPU求解（10秒模拟）
        3. 后处理（可视化、分析）
        4. 结果验证
        """
        # ========== 阶段1：前处理 ==========
        print("\n[E2E] 阶段1/3：前处理")

        from preprocessing import (
            MeshGenerator, GeometryGenerator,
            BoundaryConditionManager, InitialConditionManager
        )
        from simulation import SimulationConfig

        # 网格
        mesh_gen = MeshGenerator()
        mesh = mesh_gen.create_uniform_mesh(
            xmin=0, xmax=200, ymin=0, ymax=100,
            nx=200, ny=100
        )
        assert mesh.n_cells == 200 * 100

        # 地形
        geom_gen = GeometryGenerator()
        terrain = geom_gen.create_flat_terrain(mesh, elevation=0.0)

        # 边界条件（四周墙壁）
        bc_manager = BoundaryConditionManager()
        bc_manager.add_wall_bc('north')
        bc_manager.add_wall_bc('south')
        bc_manager.add_wall_bc('east')
        bc_manager.add_wall_bc('west')

        # 初始条件（溃坝）
        ic_manager = InitialConditionManager()
        ic_manager.set_dam_break(
            mesh, dam_position_x=100.0,
            upstream_depth=10.0,
            downstream_depth=1.0
        )

        # 配置
        config = SimulationConfig()
        config.set_domain_and_mesh(mesh)
        config.set_terrain(terrain)
        config.set_boundary_conditions(bc_manager)
        config.set_initial_conditions(ic_manager)
        config.set_solver_params(
            solver_type='cuda_gpu',
            riemann_solver='hllc',
            spatial_order=2,
            cfl=0.8,
            manning_n=0.025
        )
        config.set_output_params(
            output_interval=1.0,
            output_format='vtk'
        )

        # 验证并导出
        is_valid, errors = config.validate()
        assert is_valid, f"配置验证失败: {errors}"

        config.export_configuration(str(output_dir / "config"))
        assert (output_dir / "config" / "simulation_config.json").exists()
        print("  ✓ 前处理完成")

        # ========== 阶段2：GPU求解 ==========
        print("[E2E] 阶段2/3：GPU求解")

        from solver import GPUSolver

        solver = GPUSolver(
            str(output_dir / "config" / "simulation_config.json"),
            gpu_id=0
        )

        # 记录初始质量
        h0 = solver.get_solution()['h']
        mass0 = h0.sum() * mesh.dx * mesh.dy

        # 运行10秒模拟
        import time
        t_start = time.time()
        solver.run(
            t_end=10.0,
            output_dir=str(output_dir / "results"),
            output_interval=1.0
        )
        t_elapsed = time.time() - t_start

        print(f"  ✓ 求解完成 (耗时: {t_elapsed:.2f}s, 实时速度: {10/t_elapsed:.2f}x)")

        # 验证：输出文件存在
        vtk_files = list((output_dir / "results").glob("*.vtk"))
        assert len(vtk_files) >= 10  # 至少10个时间步

        # 验证：质量守恒
        h_final = solver.get_solution()['h']
        mass_final = h_final.sum() * mesh.dx * mesh.dy
        mass_error = abs(mass_final - mass0) / mass0
        assert mass_error < 1e-6, f"质量守恒误差过大: {mass_error:.2e}"
        print(f"  ✓ 质量守恒验证通过 (误差: {mass_error:.2e})")

        # ========== 阶段3：后处理 ==========
        print("[E2E] 阶段3/3：后处理")

        from postprocessing import ResultAnalyzer, VisualizationEngine

        # 加载结果
        analyzer = ResultAnalyzer()
        analyzer.load_vtk_series(str(output_dir / "results" / "*.vtk"))

        # 时间序列验证
        times = analyzer.get_time_series()
        assert len(times) >= 10
        assert times[-1] >= 10.0

        # 统计分析
        stats = analyzer.compute_statistics()
        assert stats['max_depth'] > 9.0  # 初始上游10m
        assert stats['max_depth'] < 11.0  # 不应超过初始值太多

        # 3D可视化（保存截图）
        vis = VisualizationEngine()
        result_5s = analyzer.get_result_at_time(5.0)
        vis.visualize_water_surface(
            result_5s.mesh, result_5s.depth, result_5s.terrain
        )
        screenshot_path = output_dir / "visualization_5s.png"
        vis.save_screenshot(str(screenshot_path))
        assert screenshot_path.exists()

        # 动画生成
        from postprocessing.animation import AnimationGenerator
        anim = AnimationGenerator()
        anim_path = output_dir / "animation.mp4"
        anim.create_animation(
            vtk_files=str(output_dir / "results" / "*.vtk"),
            output_file=str(anim_path),
            fps=10,
            variable='depth'
        )
        assert anim_path.exists()

        print("  ✓ 后处理完成")

        # ========== 最终验证 ==========
        print("[E2E] 最终验证")

        # 验证波前位置（理论估算）
        # 溃坝波速约 sqrt(gh) ≈ sqrt(10*10) ≈ 10 m/s
        # 10秒传播距离≈100m（粗略）
        x_wave_front = analyzer.get_wave_front_position(time=10.0)
        assert 80 < x_wave_front < 120, f"波前位置异常: {x_wave_front}"

        print("  ✓ 物理结果合理")
        print("\n[E2E] ========== 全工作流测试通过！==========\n")
```

**测试案例2：非结构化网格全流程**

```python
# prepost/tests/test_e2e_unstructured.py
class TestE2E_UnstructuredMesh:
    """端到端测试：非结构化网格工作流"""

    def test_unstructured_workflow(self, output_dir):
        """
        非结构化网格完整流程:
        1. Gmsh生成三角形网格
        2. 地形插值到网格
        3. GPU求解（非结构化FVM）
        4. 后处理
        """
        # 1. 生成非结构化网格
        gen = UnstructuredMeshGenerator()
        gen.add_polygon([(0,0), (200,0), (200,100), (0,100)])
        gen.add_polyline([(100,0), (100,100)], tag='dam')
        gen.set_refinement_near_tag('dam', distance=20, size=1.0)
        gen.set_default_element_size(5.0)

        mesh = gen.generate(element_type='triangle')
        assert mesh.n_elements > 1000
        assert mesh.min_angle > 20

        # 2. 地形（插值到三角形网格）
        terrain_grid = TerrainReader.read_ascii_grid('tests/data/terrain.asc')
        terrain = terrain_grid.interpolate_to_mesh(mesh)

        # 3. 配置（与结构化网格类似）
        config = SimulationConfig()
        config.set_domain_and_mesh(mesh)
        config.set_terrain(terrain)
        # ... BC, IC, solver params

        config.export_configuration(str(output_dir / "config"))

        # 4. GPU求解（使用非结构化求解器）
        solver = GPUSolver(
            str(output_dir / "config" / "simulation_config.json"),
            mesh_type='unstructured'  # 关键参数
        )
        solver.run(t_end=10.0, output_dir=str(output_dir / "results"))

        # 5. 后处理（支持非结构化网格）
        analyzer = ResultAnalyzer(mesh_type='unstructured')
        analyzer.load_vtk_series(str(output_dir / "results" / "*.vtk"))

        # 验证
        assert analyzer.check_mass_conservation() < 1e-6
        print("✓ 非结构化网格全流程测试通过")
```

**测试案例3：真实地形全流程**

```python
class TestE2E_RealTerrain:
    """端到端测试：真实地形数据工作流"""

    def test_real_terrain_workflow(self, output_dir):
        """
        真实地形流程:
        1. 导入ASCII Grid地形
        2. 地形处理（平滑、插值）
        3. 自适应网格（基于地形坡度）
        4. 降雨入流模拟
        5. 结果分析
        """
        # 1. 导入真实地形
        terrain_reader = TerrainReader()
        terrain_data = terrain_reader.read_ascii_grid('tests/data/real_dem.asc')

        # 验证地形数据
        assert terrain_data.nrows > 0
        assert terrain_data.ncols > 0
        assert not np.isnan(terrain_data.elevation).any()

        # 2. 地形处理
        processor = TerrainProcessor()
        terrain_smooth = processor.smooth(terrain_data, method='gaussian', sigma=2.0)

        # 3. 生成自适应网格
        gen = MeshGenerator()
        mesh = gen.create_adaptive_mesh_from_terrain(
            terrain_smooth,
            base_size=10.0,
            slope_threshold=0.1,
            min_size=2.0
        )

        # 4. 配置降雨入流
        bc_manager = BoundaryConditionManager()
        # 周边出流
        bc_manager.add_outflow_bc('north', bc_type='zero_gradient')
        bc_manager.add_outflow_bc('south', bc_type='zero_gradient')
        bc_manager.add_outflow_bc('east', bc_type='zero_gradient')
        bc_manager.add_outflow_bc('west', bc_type='zero_gradient')

        # Rain-on-mesh（源项）
        source_manager = SourceTermManager()
        source_manager.add_rainfall(
            intensity=50.0,  # mm/hr
            duration=3600.0,  # 1小时
            spatial_distribution='uniform'
        )

        # 5. 配置并求解
        config = SimulationConfig()
        config.set_domain_and_mesh(mesh)
        config.set_terrain(terrain_smooth)
        config.set_boundary_conditions(bc_manager)
        config.set_source_terms(source_manager)
        config.set_initial_conditions_dry_bed()

        config.export_configuration(str(output_dir / "config"))

        # 6. GPU求解
        solver = GPUSolver(str(output_dir / "config" / "simulation_config.json"))
        solver.run(t_end=7200.0, output_dir=str(output_dir / "results"))

        # 7. 后处理分析
        analyzer = ResultAnalyzer()
        analyzer.load_vtk_series(str(output_dir / "results" / "*.vtk"))

        # 淹没范围分析
        flood_extent = analyzer.compute_flood_extent(depth_threshold=0.1)
        print(f"淹没面积: {flood_extent:.2f} m²")

        # 最大水深图
        max_depth_map = analyzer.compute_maximum_depth_envelope()
        analyzer.export_raster(max_depth_map, str(output_dir / "max_depth.tif"))

        print("✓ 真实地形全流程测试通过")
```

**测试案例4-10（简要列出）**:

4. **test_e2e_urban_flood**: 城市洪水（建筑物、街道）
5. **test_e2e_river_flood**: 河道洪水（1D/2D耦合，未来）
6. **test_e2e_coastal_inundation**: 海岸淹没（潮汐BC）
7. **test_e2e_multi_scenario**: 多情景对比（参数扫描）
8. **test_e2e_long_duration**: 长时间模拟（48小时）
9. **test_e2e_large_scale**: 大规模模拟（100万+单元）
10. **test_e2e_gui_workflow**: GUI全流程（Selenium自动化测试）

#### 3.4.2 性能回归测试

```python
# prepost/tests/test_performance_regression.py
class TestPerformanceRegression:
    """性能回归测试：确保优化不降低性能"""

    BENCHMARK_CASES = [
        {'name': 'small', 'nx': 100, 'ny': 50, 't_end': 10.0},
        {'name': 'medium', 'nx': 500, 'ny': 250, 't_end': 60.0},
        {'name': 'large', 'nx': 1000, 'ny': 1000, 't_end': 600.0},
    ]

    def test_performance_baseline(self):
        """建立性能基线"""
        results = {}

        for case in self.BENCHMARK_CASES:
            config = create_benchmark_config(**case)
            solver = GPUSolver(config)

            import time
            t_start = time.time()
            solver.run(t_end=case['t_end'], output_interval=1e6)  # 不输出
            t_elapsed = time.time() - t_start

            speedup = case['t_end'] / t_elapsed
            results[case['name']] = {
                'time': t_elapsed,
                'speedup': speedup
            }

            print(f"{case['name']}: {speedup:.2f}x实时速度")

        # 保存基线
        with open('tests/performance_baseline.json', 'w') as f:
            json.dump(results, f)

    def test_performance_check(self):
        """检查性能回归"""
        # 加载基线
        with open('tests/performance_baseline.json') as f:
            baseline = json.load(f)

        # 运行当前版本
        for case in self.BENCHMARK_CASES:
            config = create_benchmark_config(**case)
            solver = GPUSolver(config)

            t_start = time.time()
            solver.run(t_end=case['t_end'], output_interval=1e6)
            t_elapsed = time.time() - t_start

            speedup_current = case['t_end'] / t_elapsed
            speedup_baseline = baseline[case['name']]['speedup']

            # 允许5%性能波动
            ratio = speedup_current / speedup_baseline
            assert ratio > 0.95, (
                f"性能回归！{case['name']}案例: "
                f"当前{speedup_current:.2f}x < 基线{speedup_baseline:.2f}x"
            )

            if ratio > 1.05:
                print(f"性能提升！{case['name']}: {ratio:.2%}")
```

### 3.5 持续集成（CI）配置

```yaml
# .github/workflows/ci.yml
name: HydroSIS-2D CI/CD

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  # Job 1: 单元测试（CPU，无需GPU）
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: |
          pytest prepost/tests/test_*.py \
            --cov=prepost \
            --cov-report=xml \
            --ignore=prepost/tests/test_cuda*.py

      - name: Upload coverage
        uses: codecov/codecov-action@v3

  # Job 2: GPU测试（需要自托管runner with GPU）
  gpu-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v3

      - name: Check CUDA
        run: nvidia-smi

      - name: Build CUDA solver
        run: |
          mkdir build && cd build
          cmake .. -DCUDA_ARCH=86
          make -j$(nproc)

      - name: Run GPU unit tests
        run: pytest prepost/tests/test_cuda*.py -v

      - name: Run integration tests
        run: pytest prepost/tests/test_integration.py -v

  # Job 3: E2E测试（每日或PR时）
  e2e-tests:
    runs-on: [self-hosted, gpu]
    if: github.event_name == 'push' || github.event.pull_request.draft == false
    steps:
      - uses: actions/checkout@v3

      - name: Run E2E tests
        run: pytest prepost/tests/test_e2e_*.py -v --timeout=3600

      - name: Performance regression check
        run: pytest prepost/tests/test_performance_regression.py -v

      - name: Archive results
        uses: actions/upload-artifact@v3
        with:
          name: e2e-results
          path: |
            temp/e2e_*/
            tests/performance_baseline.json

  # Job 4: 代码质量检查
  code-quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Lint Python code
        run: |
          pip install flake8 black
          flake8 prepost/ --max-line-length=100
          black prepost/ --check

      - name: Lint CUDA code
        run: |
          # 使用clang-format检查CUDA代码风格
          find src/ -name "*.cu" -o -name "*.cuh" | \
            xargs clang-format --dry-run --Werror
```

### 3.6 测试覆盖率目标

| 模块 | 当前覆盖率 | 目标覆盖率 | 优先级 |
|------|-----------|-----------|--------|
| 前处理（网格、地形、BC、IC） | 95% | 95% | 保持 |
| GPU求解器（核心） | 0% → | **90%** | 🔥 最高 |
| 非结构化网格 | 0% → | **85%** | 🔥 高 |
| 后处理（可视化、分析） | 60% | **80%** | ⭐ 中 |
| GUI | 0% → | **70%** | ⭐ 中 |
| **端到端工作流** | 0% → | **100%** | 🔥🔥🔥 **关键** |

**验收标准**:
- ✅ 所有单元测试通过率100%
- ✅ 集成测试通过率100%
- ✅ **E2E测试10个案例100%通过**
- ✅ 代码覆盖率总体>85%
- ✅ 无性能回归（相对基线>95%）
- ✅ 所有验证基准测试误差<5%

---

## 第四阶段：优化与扩展（3个月）📌

### 任务 4.1：多GPU并行（P2 - 中）

**时间**: 3-4周
**对标**: 商业软件大规模计算能力

- [ ] MPI + CUDA混合编程
- [ ] 域分解（Domain Decomposition）
- [ ] Halo通信优化
- [ ] 动态负载平衡

**性能目标**: 4xGPU达到3.5x加速（相对单GPU）

### 任务 4.2：多物理场耦合（P2 - 中）

**时间**: 6-8周
**对标**: MIKE FLOOD多物理场

- [ ] **溶质输运模块**
  - 对流-扩散方程
  - 点源/面源污染
  - 衰减系数

- [ ] **泥沙输运模块**（基础）
  - 冲刷淤积
  - 床面变化

- [ ] **植被阻力**
  - Manning系数空间变化
  - 植被拖曳力

### 任务 4.3：1D/2D耦合（P2 - 中）

**时间**: 8-10周
**对标**: HEC-RAS、MIKE FLOOD

- [ ] 1D圣维南方程求解器
- [ ] 1D/2D界面耦合（堰流公式）
- [ ] 河道-洪泛区双向交互

### 任务 4.4：Rain-on-mesh（P2 - 中）

**时间**: 2-3周
**对标**: InfoWorks ICM核心功能

- [ ] 降雨源项处理
- [ ] 时空分布降雨（雷达数据）
- [ ] 入渗模型（Green-Ampt、Horton）

---

## 第五阶段：生态与推广（持续）💡

### 任务 5.1：案例库建设（P3 - 低）

**内容**:
- 20+标准案例
- 每个案例包含：
  - 完整数据（网格、地形、BC）
  - 参考结果
  - 文档说明
  - Jupyter Notebook教程

**案例分类**:
1. **学术验证**（10个）：MacDonald测试集、解析解对比
2. **工程应用**（5个）：溃坝、城市洪水、河流洪水
3. **高级功能**（5个）：非结构化网格、多GPU、多物理场

### 任务 5.2：文档与教程（P3 - 低）

- [ ] 用户手册（200+页）
- [ ] API参考文档（Sphinx自动生成）
- [ ] 视频教程（10集）
  - 第1集：安装与快速开始（15分钟）
  - 第2集：GUI界面使用（20分钟）
  - 第3集：前处理详解（30分钟）
  - ...
- [ ] 在线文档站点（Read the Docs）

### 任务 5.3：社区运营（P3 - 低）

- [ ] GitHub讨论区（Q&A）
- [ ] 月度线上研讨会
- [ ] 用户调查（收集需求）
- [ ] 学术论文发表（JCP、WRR等）

---

## 开发时间表（12个月）

### Q1（月1-3）：GPU核心 🔥
- **月1**:
  - Week 1-2: CUDA开发环境搭建、数据结构设计
  - Week 3-4: HLL/HLLC Riemann求解器实现
- **月2**:
  - Week 1-2: MUSCL重构移植到CUDA
  - Week 3-4: 时间积分、干湿边界处理
- **月3**:
  - Week 1-2: 边界条件、源项处理
  - Week 3-4: pybind11集成、验证测试

**里程碑1**: GPU求解器通过所有验证测试 ✅

### Q2（月4-6）：非结构化网格 🔥
- **月4**:
  - Week 1-2: Gmsh集成、网格生成器
  - Week 3-4: 自适应加密、质量检查
- **月5**:
  - Week 1-3: 非结构化FVM求解器（CPU版本）
  - Week 4: GPU移植准备
- **月6**:
  - Week 1-2: 非结构化GPU求解器
  - Week 3-4: 验证测试、性能优化

**里程碑2**: 非结构化网格求解器完成 ✅

### Q3（月7-9）：GUI与可视化 ⭐
- **月7**:
  - Week 1-2: 高级可视化（流线、粒子追踪）
  - Week 3-4: 等值面、体渲染
- **月8-9**:
  - Week 1-8: Streamlit GUI开发
    - 前处理界面（3周）
    - 求解器控制（2周）
    - 后处理界面（3周）

**里程碑3**: GUI Alpha版本发布 ✅

### Q4（月10-12）：优化与扩展 📌
- **月10**:
  - Week 1-2: 多GPU支持（MPI+CUDA）
  - Week 3-4: Rain-on-mesh功能
- **月11**:
  - Week 1-4: 溶质输运模块
- **月12**:
  - Week 1-2: 案例库建设（20个案例）
  - Week 3-4: 文档完善、v1.0发布准备

**里程碑4**: HydroSIS-2D v1.0正式发布 🎉

---

## 资源需求

### 人力资源

| 角色 | 人数 | 工时 | 说明 |
|------|-----|------|------|
| CUDA开发工程师 | 1人 | 全职6个月 | GPU求解器、非结构化网格 |
| Python开发工程师 | 1人 | 全职6个月 | GUI、前后处理集成 |
| 测试工程师 | 0.5人 | 兼职12个月 | 编写测试、CI/CD |
| 文档工程师 | 0.5人 | 兼职3个月 | 用户手册、教程 |

**总人月**: 约15人月

### 硬件资源

**开发环境**:
- 工作站：RTX 4090 (24GB) × 2，约 $3,500
- 或租用云GPU：AWS p3.2xlarge (V100)，$3/小时

**测试环境**:
- 服务器：4×A100 (40GB)，约 $40,000
- 或云端：AWS p4d.24xlarge，$32/小时

**预算**（云端方案）:
- 开发测试：100小时/月 × $3/hr × 6月 = $1,800
- 性能测试：50小时 × $32/hr = $1,600
- **总计**: ~$3,500

### 软件许可（全部免费）

- CUDA Toolkit: 免费
- Gmsh: GPL，免费
- PyQt6/Streamlit: 开源，免费
- ParaView: BSD，免费

---

## 风险管理

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|-----|------|---------|
| GPU性能不达标 | 中 | 高 | 早期基准测试，算法优化，咨询专家 |
| 非结构化网格数值不稳定 | 中 | 中 | 采用成熟算法，充分验证 |
| Gmsh集成困难 | 低 | 中 | 有丰富文档和示例，社区活跃 |
| GUI响应性差 | 低 | 低 | 异步处理，进度回调 |

### 进度风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|-----|------|---------|
| 关键人员离职 | 中 | 高 | 代码文档完善，知识共享会议 |
| 需求变更 | 中 | 中 | 敏捷开发，每月评审 |
| 硬件资源不足 | 低 | 中 | 云端备份方案 |

### 质量风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|-----|------|---------|
| 测试覆盖率不足 | 低 | 高 | **强制E2E测试**，CI/CD自动化 |
| 数值精度问题 | 中 | 高 | 多个验证基准，与商业软件对比 |
| 内存泄漏 | 中 | 中 | Valgrind/CUDA-MEMCHECK检查 |

---

## 成功指标

### 技术指标（量化）

- [ ] **GPU加速**: ≥50x CPU速度（1M cells）
- [ ] **数值精度**: 所有验证测试误差<5%
- [ ] **质量守恒**: 误差<1e-6
- [ ] **测试覆盖率**: >85%（总体），E2E 100%
- [ ] **代码质量**: Flake8无错误，文档完整率100%

### 性能基准（对标商业软件）

| 指标 | RiverFlow2D | TUFLOW GPU | HydroSIS目标 | 状态 |
|------|------------|-----------|-------------|------|
| GPU加速比 | 100-150x | 50-100x | 50-150x | ⏳ 开发中 |
| 最大网格规模 | 10M cells | 5M cells | 5M cells | ⏳ 开发中 |
| 非结构化网格 | ✅ | ✅ | ✅ | ⏳ 开发中 |
| Rain-on-mesh | ✅ | ✅ | ✅ | 📅 Q4计划 |
| 1D/2D耦合 | ✅ | ✅ | ✅ | 📅 未来 |
| 开源免费 | ❌ ($$$) | ❌ ($$$) | ✅ | ✅ 已是 |

### 用户指标（定性）

- [ ] **易用性**: 新用户30分钟完成第一个模拟
- [ ] **GUI满意度**: >4.0/5.0（用户调查）
- [ ] **文档完整性**: 所有功能有文档+示例

### 生态指标（6个月后）

- [ ] GitHub Stars: >500
- [ ] 月活跃用户: >100
- [ ] 第三方贡献者: >5人
- [ ] 学术论文: >2篇（已投稿或发表）

---

## 下一步行动（立即开始）

### 本周任务（Week 1）

1. **环境搭建**:
   - [ ] 安装CUDA Toolkit 12.x
   - [ ] 配置CMake构建系统
   - [ ] 搭建Git开发分支（`develop`, `feature/cuda-solver`）

2. **代码框架**:
   - [ ] 创建`src/solver/cuda/`目录结构
   - [ ] 编写CUDA求解器类框架（空函数）
   - [ ] 配置pybind11绑定模板

3. **第一个GPU kernel**:
   - [ ] 实现简单的HLL通量计算kernel
   - [ ] 编写对应单元测试
   - [ ] 验证GPU-CPU一致性

### 本月目标（Month 1）

- [ ] HLL/HLLC Riemann求解器完成
- [ ] 一阶迎风格式GPU求解器可运行
- [ ] 通过1D溃坝验证测试
- [ ] 性能测试：小网格达到10x+加速

### 本季度目标（Q1）

- [ ] **里程碑1**：GPU求解器完成
  - 通过所有10个验证基准测试
  - 性能达到50x+加速（中大网格）
  - Python接口完整可用
  - 文档完整

---

## 附录

### A. 参考文献

**数值方法**:
1. Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*. Springer.
2. LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*. Cambridge.
3. Kurganov, A., & Petrova, G. (2007). *A second-order well-balanced positivity preserving central-upwind scheme for the Saint-Venant system*. CNMM.

**GPU加速**:
4. Lacasta, A., et al. (2014). *GPU-enhanced Finite Volume Shallow Water solver for fast flood simulations*. Environmental Modelling & Software.
5. Smith, L. S., & Liang, Q. (2013). *Towards a generalised GPU/CPU shallow-flow modelling tool*. Computers & Fluids.

**验证基准**:
6. MacDonald, I., et al. (1997). *Analytic benchmark solutions for open-channel flows*. Journal of Hydraulic Engineering.
7. UK Environment Agency (2013). *Benchmarking of 2D Hydraulic Modelling Packages*.

### B. 商业软件对比详表

| 功能 | HEC-RAS | MIKE FLOOD | TUFLOW | InfoWorks ICM | RiverFlow2D | HydroSIS-2D目标 |
|------|---------|-----------|--------|--------------|-------------|----------------|
| GPU加速 | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ |
| 2D SWE | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| 1D/2D耦合 | ✅ | ✅ | ✅ | ✅ | ✅ | 📅 Phase 4 |
| 非结构化网格 | ✅ (8边) | ✅ | ✅ | ✅ (三角形) | ✅ | ✅ Q2 |
| Rain-on-mesh | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ Q4 |
| 粒子追踪 | ❌ | ❌ | ✅ | ❌ | ✅ | ✅ Q3 |
| 泥沙输运 | ✅ (基础) | ✅ (全面) | ✅ | ❌ | ✅ | 📅 Phase 4 |
| GUI | ✅ (桌面) | ✅ (桌面) | ✅ (桌面) | ✅ (桌面) | ✅ (桌面) | ✅ (Web) Q3 |
| Python API | ✅ | ❌ | ✅ | ❌ | ❌ | ✅ |
| 开源 | ✅ (免费) | ❌ | ❌ | ❌ | ❌ | ✅ |
| 价格 | 免费 | $$$ | $$$ | $$$ | $$$ | 免费 |

### C. 词汇表

- **FVM**: Finite Volume Method，有限体积法
- **MUSCL**: Monotone Upstream-centered Scheme for Conservation Laws，单调迎风守恒律格式
- **HLL**: Harten-Lax-van Leer Riemann求解器
- **HLLC**: HLL-Contact（HLL改进版，考虑接触间断）
- **CFL**: Courant-Friedrichs-Lewy条件（时间步长稳定性条件）
- **C-property**: 静止湖面测试（well-balanced验证）
- **E2E**: End-to-End，端到端
- **CI/CD**: Continuous Integration/Continuous Deployment，持续集成/持续部署

---

**文档版本**: 3.0
**创建日期**: 2025-11-13
**下次更新**: 每月评审更新
**维护者**: HydroSIS-2D开发团队

---

## 总结

本路线图基于HydroSIS-2D当前成熟的前后处理工具包，对标国际商业软件（RiverFlow2D、TUFLOW、HEC-RAS等），提出了清晰的12个月开发计划：

**核心战略**:
1. **Q1-Q2（6个月）**: 🔥 **GPU求解器 + 非结构化网格** —— 建立核心竞争力
2. **Q3（3个月）**: ⭐ **GUI + 高级可视化** —— 降低使用门槛
3. **Q4（3个月）**: 📌 **优化扩展 + 生态建设** —— 形成完整产品
4. **贯穿始终**: 🧪 **全工作流测试** —— 确保质量可靠

**差异化优势**:
- ✅ **开源免费**（对标昂贵商业软件）
- ✅ **GPU加速**（对标RiverFlow2D, TUFLOW）
- ✅ **现代技术栈**（Python + CUDA + Web GUI）
- ✅ **科研友好**（完整文档、可扩展）

**成功关键**:
- **测试第一**：E2E测试100%覆盖，确保全工作流可用
- **性能至上**：GPU加速50-150x，对标商业软件
- **用户导向**：30分钟上手，Streamlit GUI简化操作
- **质量保证**：CI/CD自动化，所有验证测试通过

**12个月后**，HydroSIS-2D将成为**开源领域最强大的2D水动力模拟平台**，与商业软件在功能和性能上并驾齐驱，同时完全免费开放。

🚀 **Let's build the future of open-source hydraulic modeling!**
