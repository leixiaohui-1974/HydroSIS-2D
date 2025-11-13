# HydroSIS-2D Examples

This directory contains complete workflow examples demonstrating HydroSIS-2D capabilities.

---

## 📚 Example Overview

| Example | Purpose | Difficulty | Runtime | Key Features |
|---------|---------|------------|---------|--------------|
| 01_basic_dam_break.py | Getting started | ⭐ Simple | ~5s | Basic workflow |
| 02_performance_benchmark.py | GPU speedup | ⭐⭐ Moderate | ~5min | Multi-scale testing |
| 03_analytical_validation.py | Accuracy | ⭐⭐ Moderate | ~3min | Error analysis |
| 04_urban_flood.py | Real-world | ⭐⭐⭐ Advanced | ~2min | Urban flooding |

**Total**: 4 complete workflows, 1,449 lines

---

## 🚀 Quick Start

### Prerequisites
```bash
# 1. Build GPU solver
cd ../src/solver/build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
make -j$(nproc) && make install

# 2. Install dependencies
pip install -r ../requirements.txt

# 3. Verify installation
python -c "import hydrosis2d_cuda; print('✓ GPU solver available')"
```

### Running Examples
```bash
python 01_basic_dam_break.py
python 02_performance_benchmark.py  
python 03_analytical_validation.py
python 04_urban_flood.py
```

---

## 📖 Detailed Descriptions

### Example 1: Basic Dam Break (208 lines)

**Purpose**: Introduction to complete workflow

**Scenario**:
- Domain: 200m × 100m (20,000 cells)
- Upstream: 10m depth
- Downstream: 1m depth
- Duration: 10 seconds

**Output**: 4-panel visualization
- Initial depth
- Final depth  
- Velocity magnitude
- Cross-section comparison

**Learning**: Basic workflow, mesh setup, solver configuration

---

### Example 2: Performance Benchmark (293 lines)

**Purpose**: GPU performance testing

**Test Cases**: 5 mesh sizes (2.5k - 80k cells)

**Metrics**:
- Compute time
- Throughput (Mcups)
- GPU speedup (vs CPU estimate)
- Scaling efficiency

**Output**: Performance plots, comparison table

**Learning**: GPU benchmarking, commercial comparison

---

### Example 3: Analytical Validation (348 lines)

**Purpose**: Validate numerical accuracy

**Tests**:
1. **Ritter Dam Break**: 1D analytical solution
2. **Lake at Rest**: C-property (well-balanced)

**Metrics**:
- L1, L2, L∞ errors
- Mass conservation
- Spurious currents

**Output**: Comparison plots, error analysis

**Learning**: Numerical accuracy, error metrics

---

### Example 4: Urban Flood Simulation (600+ lines)

**Purpose**: Real-world urban flooding application

**Scenario**:
- Domain: 600m × 500m (75,000 cells)
- 6 buildings (varying heights)
- Street network + parks
- Spatially varying Manning roughness
- Rainfall: 100 mm/hr for 30 minutes

**Output**: 9-panel comprehensive visualization
- Terrain with buildings
- Flood depth and extent
- Velocity vectors
- Flood progression timeline
- Detailed statistics

**Learning**: Complex terrain, spatial parameters, real-world application

**Comparable to**: RiverFlow2D Urban, TUFLOW, InfoWorks ICM

---

## 🔧 Customization

### Change Mesh Resolution
```python
# Higher resolution
mesh = mesh_gen.generate_uniform_mesh(nx=400, ny=200)

# Lower resolution  
mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=50)
```

### Change Solver Settings
```python
# Use HLL solver
solver_config.riemann_solver = hydrosis2d_cuda.RiemannSolver.HLL

# Adjust CFL
solver_config.cfl = 0.5  # More stable

# Enable 2nd-order
solver_config.spatial_order = 2
solver_config.limiter = hydrosis2d_cuda.Limiter.VanLeer
```

### Change Output
```python
# Custom location
output_file = Path('/custom/path/results.png')
plt.savefig(output_file, dpi=300)

# Export data
np.save('depth.npy', result['h'])
```

---

## 📊 Expected Output

```
examples/output/
├── 01_dam_break_results.png
├── 02_performance_results.png
├── 02_performance_metrics.json
├── 03_validation_results.png
├── 03_validation_errors.txt
├── 04_urban_flood_setup.png
└── 04_urban_flood_results.png
```

---

## 🐛 Troubleshooting

### GPU Solver Not Found
```bash
cd ../src/solver/build
cmake .. && make -j$(nproc) && make install
```

### Out of Memory
Reduce mesh size:
```python
mesh = mesh_gen.generate_uniform_mesh(nx=100, ny=50)
```

### Slow Performance
- Check: `nvidia-smi`
- Close other GPU apps
- Verify CUDA architecture match

---

## 📚 Resources

- [GPU Quickstart](../QUICKSTART_GPU.md)
- [Test Catalog](../docs/COMPREHENSIVE_TEST_CATALOG.md)
- [Contributing Guide](../CONTRIBUTING.md)
- [Project Status](../PROJECT_STATUS.md)

---

## 🎓 Learning Path

1. **Beginner**: Example 1 → Understand workflow
2. **Intermediate**: Examples 2-3 → Performance & validation  
3. **Advanced**: Example 4 → Real-world application
4. **Expert**: Modify for your use case

---

**Happy Modeling!** 🌊

*Last updated: 2025-11-13*
