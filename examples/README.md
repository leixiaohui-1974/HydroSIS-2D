# HydroSIS-2D Examples

This directory contains example scripts demonstrating how to use HydroSIS-2D for shallow water flow simulations.

## Prerequisites

1. **Python Environment**:
   ```bash
   python >= 3.10
   numpy
   matplotlib
   ```

2. **GPU Solver** (required for GPU examples):
   ```bash
   cd src/solver
   mkdir build && cd build
   cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
   make -j$(nproc)
   make install
   ```

3. **Verify Installation**:
   ```bash
   python -c "import hydrosis2d_cuda; print(f'GPUs: {hydrosis2d_cuda.get_gpu_count()}')"
   ```

---

## Examples

### 01_basic_dam_break.py

**Basic dam break simulation with visualization**

Demonstrates:
- Creating a dam break scenario
- Configuring GPU solver
- Running simulation
- Visualizing results

Usage:
```bash
python 01_basic_dam_break.py
```

Output:
- Console progress updates
- Visualization with 4 subplots:
  1. Initial water depth
  2. Final water depth
  3. Velocity magnitude
  4. Cross-section comparison
- Saved figure: `output/01_dam_break_results.png`

Expected runtime: ~5-10 seconds (200×100 mesh, 10s simulation)

---

### 02_performance_benchmark.py

**Performance benchmarking against commercial software**

Demonstrates:
- Running tests on multiple mesh sizes
- Computing GPU speedup vs CPU
- Comparing against commercial benchmarks (RiverFlow2D, TUFLOW)
- Throughput analysis

Usage:
```bash
python 02_performance_benchmark.py
```

Output:
- Performance metrics for each mesh size
- Speedup comparison table
- 4 performance plots:
  1. Speedup vs mesh size
  2. Throughput (Mcups)
  3. GPU vs CPU time comparison
  4. Scaling efficiency
- Saved figure: `output/02_performance_results.png`

Expected runtime: 2-5 minutes (depends on mesh sizes)

Benchmark targets:
- Small mesh (10k cells): 20x speedup
- Medium mesh (20k cells): 30x speedup
- Large mesh (40k cells): 50x speedup
- XLarge mesh (80k cells): 60x speedup

---

### 03_analytical_validation.py

**Validation against analytical solutions**

Demonstrates:
- Ritter dam break (1D analytical solution)
- Lake at rest (C-property test)
- Error metrics (L1, L2, L-infinity)
- Well-balanced property verification

Usage:
```bash
python 03_analytical_validation.py
```

Output:
- Error analysis for each test
- Validation status (PASS/FAIL)
- Comparison plots (numerical vs analytical)
- Saved figures in `output/`

Expected runtime: 1-2 minutes

Validation criteria:
- Ritter dam break: L2 error < 0.1
- Lake at rest: Max velocity < 1e-6 m/s

---

## Common Workflow

### 1. Quick Start - Run a Single Example

```bash
# Basic dam break (simplest example)
python 01_basic_dam_break.py

# Expected output:
# ============================================================
# Example 1: Basic Dam Break
# ============================================================
#
# [1/4] Creating dam break configuration...
#   ✓ Mesh created: 200×100 = 20,000 cells
#   ...
#   ✓ Simulation complete!
```

### 2. Performance Testing

```bash
# Run performance benchmarks
python 02_performance_benchmark.py

# Compare your results with commercial tools:
# - RiverFlow2D: 30-100x GPU speedup
# - TUFLOW GPU: 50-100x GPU speedup
```

### 3. Validation Workflow

```bash
# Validate numerical accuracy
python 03_analytical_validation.py

# Check that:
# - Analytical solutions match (Ritter)
# - No spurious currents (Lake at rest)
# - Mass is conserved
```

---

## Output Directory Structure

```
examples/
├── output/
│   ├── 01_dam_break_results.png
│   ├── 02_performance_results.png
│   ├── 03_ritter_validation.png
│   └── 03_lake_at_rest.png
├── 01_basic_dam_break.py
├── 02_performance_benchmark.py
├── 03_analytical_validation.py
└── README.md
```

---

## Customization

### Modify Mesh Resolution

```python
# In any example, change:
config = create_dam_break_simulation(
    nx=400,  # Increase for finer resolution
    ny=200,  # Increase for finer resolution
    ...
)
```

### Change Solver Settings

```python
# Modify solver configuration:
solver_config.cfl = 0.5  # Lower for more stability
solver_config.riemann_solver = hydrosis2d_cuda.RiemannSolver.HLL  # Change solver
solver_config.spatial_order = 2  # Use 2nd-order MUSCL
```

### Add Custom Initial Conditions

```python
# Create custom initial conditions:
h = np.zeros((ny, nx))
# Your custom setup...
h[y > 50] = 5.0  # Example

ic_manager.set_depth_array(h)
```

---

## Troubleshooting

### GPU Solver Not Found

**Error**: `ModuleNotFoundError: No module named 'hydrosis2d_cuda'`

**Solution**:
1. Ensure CUDA toolkit is installed
2. Compile the GPU solver:
   ```bash
   cd src/solver
   mkdir build && cd build
   cmake .. -DCMAKE_CUDA_ARCHITECTURES=native
   make -j$(nproc)
   make install
   ```

### CUDA Out of Memory

**Error**: `CUDA error: out of memory`

**Solution**:
- Reduce mesh size (lower `nx` and `ny`)
- Use smaller data types (if implemented)
- Close other GPU applications

### Slow Performance

**Issue**: GPU solver slower than expected

**Debugging**:
1. Check GPU utilization:
   ```bash
   nvidia-smi -l 1
   ```
2. Verify GPU is not throttling (temperature)
3. Ensure `--use_fast_math` is enabled in CMake
4. Profile with Nsight Compute:
   ```bash
   ncu --set full -o profile python 01_basic_dam_break.py
   ```

---

## Performance Expectations

### RTX 3090 (24 GB)

| Mesh Size | Cells | Simulation Time | Wall Time | Speedup |
|-----------|-------|-----------------|-----------|---------|
| Small | 10k | 10s | ~1s | 30x |
| Medium | 50k | 60s | ~10s | 60x |
| Large | 200k | 60s | ~20s | 90x |
| XLarge | 1M | 3600s | ~80s | 100x |

### A100 (40 GB)

Expected 1.5-2x better performance than RTX 3090 due to higher memory bandwidth.

---

## Advanced Examples

### Create Custom Scenarios

See `prepost/preprocessing/utils.py` for helper functions:

```python
from preprocessing.utils import (
    create_dam_break_simulation,
    create_uniform_flow_simulation,
    create_rainfall_runoff_simulation
)
```

### MacDonald Test Suite

Standard validation cases from literature:

```bash
cd ../prepost
pytest tests/validation/test_macdonald_suite.py -v -s
```

---

## References

### Numerical Methods
- Toro, E. F. (2009). *Riemann Solvers and Numerical Methods for Fluid Dynamics*
- LeVeque, R. J. (2002). *Finite Volume Methods for Hyperbolic Problems*

### Validation Cases
- Ritter, A. (1892). *Die Fortpflanzung der Wasserwellen*
- MacDonald et al. (1997). *Analytic benchmark solutions for open-channel flows*

### Commercial Software Benchmarks
- RiverFlow2D GPU: https://www.hydronia.com/riverflow2d/
- TUFLOW GPU: https://www.tuflow.com/

---

## Contributing

To add a new example:

1. Create `0X_example_name.py`
2. Follow the existing structure:
   - Clear docstring
   - Step-by-step execution
   - Progress messages
   - Visualization
   - Save output
3. Update this README
4. Test on different mesh sizes

---

## Support

- **Issues**: https://github.com/leixiaohui-1974/HydroSIS-2D/issues
- **Documentation**: `docs/` directory
- **Discussions**: GitHub Discussions

---

**Last Updated**: 2025-11-13
**Version**: 0.1.0-dev
