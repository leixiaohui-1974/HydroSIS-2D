# HydroSIS-2D GPU Solver Development Guide

**Status**: 🚧 Under Active Development

This directory contains the CUDA GPU-accelerated solver for HydroSIS-2D.

---

## 📋 Quick Start

### Prerequisites

- CUDA Toolkit 11.0+ (12.x recommended)
- CMake 3.18+
- C++17 compatible compiler
- Python 3.10+
- NVIDIA GPU with compute capability 6.0+ (Pascal or newer)

### Build Instructions

```bash
# 1. Navigate to solver directory
cd src/solver

# 2. Create build directory
mkdir build && cd build

# 3. Configure with CMake
cmake .. -DCMAKE_CUDA_ARCHITECTURES=native

# 4. Build
make -j$(nproc)

# 5. Install Python module
make install
```

### Verify Installation

```bash
# Test GPU availability
python -c "import hydrosis2d_cuda; print(f'GPUs: {hydrosis2d_cuda.get_gpu_count()}')"

# Check GPU memory
python -c "import hydrosis2d_cuda; print(hydrosis2d_cuda.get_gpu_memory())"
```

---

## 🏗️ Project Structure

```
src/solver/
├── cuda/
│   ├── ShallowWaterSolver.cuh     # Main solver class
│   ├── RiemannSolver.cuh          # Riemann solvers (HLL, HLLC)
│   ├── kernels/
│   │   ├── flux_kernels.cu        # Flux computation
│   │   ├── update_kernels.cu      # State update (TODO)
│   │   ├── source_kernels.cu      # Source terms (TODO)
│   │   └── bc_kernels.cu          # Boundary conditions (TODO)
│   └── python/
│       └── bindings.cpp           # pybind11 Python interface
├── CMakeLists.txt                 # Build configuration
└── README.md                      # This file
```

---

## 🚀 Development Roadmap

### ✅ Completed
- [x] Project structure
- [x] CUDA solver header (ShallowWaterSolver.cuh)
- [x] Riemann solver interface (HLL, HLLC)
- [x] Python bindings framework (pybind11)
- [x] CMake build system
- [x] First-order flux kernels (skeleton)

### 🚧 In Progress
- [ ] **Complete CUDA kernel implementation**
  - [ ] Flux computation (x and y directions)
  - [ ] Conservative variable update
  - [ ] Source terms (bed slope, friction)
  - [ ] Boundary conditions
  - [ ] CFL time step calculation

- [ ] **MUSCL reconstruction (2nd order)**
  - [ ] Gradient computation
  - [ ] Limiters (Minmod, Van Leer, Superbee, MC)
  - [ ] Left/right state reconstruction

- [ ] **Time integration**
  - [ ] Euler (1st order) - DONE in skeleton
  - [ ] RK2 (2nd order)
  - [ ] RK3-TVD (3rd order)

### 📅 Planned
- [ ] Unstructured mesh support
- [ ] Multi-GPU parallelization (MPI+CUDA)
- [ ] Advanced optimizations (shared memory, streams)
- [ ] Benchmark suite

---

## 🧪 Testing

### Unit Tests

```bash
# CPU unit tests (no GPU required)
cd ../../prepost
pytest tests/test_*.py --ignore=tests/test_gpu*.py -v

# GPU unit tests (requires GPU)
pytest tests/test_gpu*.py -v

# GPU-CPU consistency tests
pytest tests/test_gpu_cpu_consistency.py -v
```

### End-to-End Tests

```bash
# Run complete workflow test
pytest tests/e2e/test_e2e_dam_break.py -v -s

# This validates:
# - Preprocessing (mesh, terrain, BC, IC)
# - Solver configuration
# - (GPU solving - pending implementation)
# - Postprocessing
# - Result validation
```

### Performance Benchmarks

```bash
# Once GPU solver is ready
pytest tests/test_performance_regression.py -v
```

---

## 💡 Usage Example

```python
import numpy as np
import hydrosis2d_cuda

# Create solver configuration
config = hydrosis2d_cuda.SolverConfig()
config.nx = 200
config.ny = 100
config.dx = 1.0
config.dy = 1.0
config.riemann_solver = hydrosis2d_cuda.RiemannSolver.HLLC
config.spatial_order = 2  # MUSCL
config.time_integrator = hydrosis2d_cuda.TimeIntegrator.RK2
config.cfl = 0.8

# Create solver
solver = hydrosis2d_cuda.Solver()
solver.initialize(config)

# Set initial conditions (numpy arrays)
h = np.zeros((config.ny, config.nx))
h[:, :100] = 10.0  # Dam upstream
h[:, 100:] = 1.0   # Dam downstream

u = np.zeros_like(h)
v = np.zeros_like(h)
z = np.zeros_like(h)  # Flat terrain

solver.set_initial_conditions(h, u, v, z)

# Set progress callback
def callback(time, step, dt):
    print(f"t={time:.2f}s, step={step}, dt={dt:.4f}s")

solver.set_callback(callback, interval=100)

# Run simulation
solver.run(t_end=10.0)

# Get results
result = solver.get_solution()
print(f"Final time: {result['time']}")
print(f"Max depth: {result['h'].max():.2f} m")

# Export to VTK
solver.export_vtk('output/final_state.vtk')
```

---

## 🔧 Development Tips

### CUDA Debugging

```bash
# Compile with debug symbols
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_FLAGS="-g -G"

# Check for memory errors
compute-sanitizer --tool memcheck ./your_test

# Profile with Nsight Compute
ncu --set full -o profile ./your_test
```

### Common Issues

**Issue**: `undefined reference to cudaXXX`
**Fix**: Make sure CUDA toolkit is in `CMAKE_PREFIX_PATH`

**Issue**: `ImportError: No module named 'hydrosis2d_cuda'`
**Fix**: Run `make install` or add build directory to `PYTHONPATH`

**Issue**: `CUDA error: out of memory`
**Fix**: Reduce mesh size or use smaller data types

---

## 📚 References

### Numerical Methods
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*

### CUDA Programming
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Optimizing CUDA Applications](https://developer.nvidia.com/blog/tag/cuda/)

### Shallow Water Solvers
- Lacasta et al. (2014). *GPU-enhanced Finite Volume Shallow Water solver*
- Kurganov & Petrova (2007). *Well-balanced positivity preserving central-upwind scheme*

---

## 🤝 Contributing

1. **Create feature branch**: `git checkout -b feature/your-feature`
2. **Write tests first**: Add tests to `prepost/tests/`
3. **Implement feature**: Update CUDA kernels/solver
4. **Run tests**: `pytest -v`
5. **Submit PR**: With clear description and test results

### Code Style
- C++/CUDA: Follow CUDA best practices
- Python: Black formatter, max line length 100
- Comments: Document all GPU kernels with complexity analysis

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/leixiaohui-1974/HydroSIS-2D/issues)
- **Discussions**: [GitHub Discussions](https://github.com/leixiaohui-1974/HydroSIS-2D/discussions)
- **Docs**: `docs/` directory

---

**Last Updated**: 2025-11-13
**Status**: Pre-alpha development
**Target Release**: Q1 2025
