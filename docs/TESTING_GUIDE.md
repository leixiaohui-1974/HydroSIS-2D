# HydroSIS-2D Testing Guide

## Overview

This document describes the comprehensive testing suite for HydroSIS-2D, including 9 standard test cases for validation and performance benchmarking.

## Test Cases

### 1. **1D Dam Break (Ritter's Solution)** [Test ID: 0]
- **Purpose**: Validate shock capturing and analytical comparison
- **Initial Condition**: High water (h=10m) on left, low water (h=1m) on right
- **Physics**: Rarefaction wave and shock formation
- **Validation**: Compare with Ritter's analytical solution
- **Expected Error**: L1 < 0.1, L2 < 0.15

### 2. **2D Circular Dam Break** [Test ID: 1]
- **Purpose**: Test radial symmetry and 2D propagation
- **Initial Condition**: Circular high water region in center
- **Physics**: Radially symmetric wave propagation
- **Validation**: Check radial symmetry preservation
- **Expected**: Symmetric flood pattern

### 3. **Partial Dam Break** [Test ID: 2]
- **Purpose**: Test asymmetric 2D flow patterns
- **Initial Condition**: Dam with partial breach/gap
- **Physics**: Complex 2D flow through opening
- **Validation**: Physical plausibility check
- **Expected**: Flow concentration through gap

### 4. **Thacker's Planar Beach** [Test ID: 3]
- **Purpose**: Validate wetting/drying on sloping beach
- **Initial Condition**: Oscillating flow on parabolic bed
- **Physics**: Analytical wetting/drying test
- **Validation**: Compare with analytical solution
- **Expected Error**: Very low (<1e-3)

### 5. **MacDonald Wetting/Drying** [Test ID: 4]
- **Purpose**: Standard benchmark with friction
- **Initial Condition**: Water flow over hump with friction
- **Physics**: Bed slope + Manning friction + wetting/drying
- **Validation**: Compare with published results
- **Expected**: Similar to MacDonald (2005) results

### 6. **Lake at Rest** [Test ID: 5]
- **Purpose**: Test well-balanced property
- **Initial Condition**: Still water over complex topography
- **Physics**: Exact balance of pressure and bed slope
- **Validation**: Velocity should remain ~0, surface flat
- **Expected**: max|V| < 1e-4 m/s, mass conserved exactly

### 7. **Small Perturbation** [Test ID: 6]
- **Purpose**: Test stability for small amplitude waves
- **Initial Condition**: Gaussian perturbation on flat bed
- **Physics**: Linear wave propagation
- **Validation**: Wave should propagate without spurious oscillations
- **Expected**: Smooth wave propagation

### 8. **Flow Over Bump** [Test ID: 7]
- **Purpose**: Test transcritical flow transitions
- **Initial Condition**: Steady inflow over Gaussian bump
- **Physics**: Subcritical/supercritical flow transition
- **Validation**: Check Froude number behavior
- **Expected**: Fr > 1 near crest

### 9. **Oblique Hydraulic Jump** [Test ID: 8]
- **Purpose**: Test 2D shock capturing
- **Initial Condition**: Diagonal discontinuity
- **Physics**: 2D hydraulic jump formation
- **Validation**: Sharp shock resolution
- **Expected**: No spurious oscillations

## Running Tests

### Quick Sanity Test

```bash
cd HydroSIS-2D
./test/quick_test.sh
```

This runs a fast test (256×256, 1 second) to verify the installation.

### Full Test Suite

```bash
./test/run_all_tests.sh
```

This runs all 9 test cases with:
- Grid: 256×256
- CFL: 0.5
- Time: 2.0 seconds
- Validation enabled
- VTK output enabled

Results are saved to `test_results/` directory.

### Performance Benchmark

```bash
./test/benchmark.sh
```

This tests performance with increasing grid sizes:
- 128×128
- 256×256
- 512×512
- 1024×1024
- 2048×2048

Results saved to `benchmark_results/scaling_single_gpu.csv`.

### Individual Test

```bash
cd build
./hydrosis --test 0 --nx 512 --ny 512 --cfl 0.5 --tend 5.0 --validate --vtk
```

Command line options:
- `--test <id>`: Test case ID (0-8)
- `--nx, --ny`: Grid dimensions
- `--cfl`: CFL number (0.5-0.9 recommended)
- `--tend`: End time in seconds
- `--validate`: Enable validation and error analysis
- `--vtk`: Enable VTK output for ParaView

## Validation Metrics

### Mass Conservation
The solver should conserve mass to machine precision:
```
Mass Error = |M(t) - M(0)| / M(0) < 1e-6
```

### Physical Validity
- No negative depths: h ≥ 0
- Reasonable velocities: |V| < 100 m/s
- No NaN or Inf values

### Numerical Accuracy
For tests with analytical solutions:
- **L1 norm**: Mean absolute error
- **L2 norm**: Root mean square error
- **L∞ norm**: Maximum error

Target accuracy for 2nd-order scheme:
- Error ~ O(Δx²)
- Grid convergence rate ≈ 2.0

## Output Files

### ASCII Output (default)
```
output_XXXX.dat
```
Format: `x y h u v z`

### VTK Output (with --vtk)
```
output_XXXXXX.vtk
```
Can be opened in ParaView for visualization.

Variables:
- `depth`: Water depth (m)
- `velocity`: Velocity vector (m/s)
- `velocity_magnitude`: Speed (m/s)
- `bed_elevation`: Bed elevation (m)
- `surface_elevation`: Water surface elevation (m)
- `froude_number`: Froude number (dimensionless)
- `wet_dry`: Wet (1) or dry (0) flag

## Visualization

### Using ParaView

1. Install ParaView: https://www.paraview.org/
2. Open VTK files: `File -> Open -> output_*.vtk`
3. Apply filter
4. Visualize:
   - Depth: Use "Surface" representation with "depth" coloring
   - Velocity: Use "Glyph" filter with arrows
   - Animation: Load time series

### Example Visualization Workflow

```python
# Python script for ParaView
from paraview.simple import *

# Load data
reader = XMLStructuredGridReader(FileName='output_*.vtk')

# Create depth plot
depthDisplay = Show(reader)
depthDisplay.Representation = 'Surface'
ColorBy(depthDisplay, ('POINTS', 'depth'))

# Create velocity vectors
glyph = Glyph(Input=reader, GlyphType='Arrow')
glyph.Vectors = ['POINTS', 'velocity']
glyph.ScaleFactor = 0.1
glyphDisplay = Show(glyph)

# Render
Render()
```

## Expected Performance

### Single GPU (RTX 3090)
- 256×256: ~0.1-0.2 gigacells/s
- 512×512: ~0.3-0.5 gigacells/s
- 1024×1024: ~0.8-1.2 gigacells/s
- 2048×2048: ~1.5-2.5 gigacells/s

### Multi-GPU (4× V100)
- 2048×2048: ~3-5 gigacells/s
- 4096×4096: ~8-12 gigacells/s
- 8192×8192: ~15-25 gigacells/s

Performance depends on:
- GPU model and memory bandwidth
- Grid size (larger is more efficient)
- CFL number (higher = fewer steps)
- Test case complexity

## Troubleshooting

### Test Failures

1. **Mass not conserved**
   - Check boundary conditions
   - Verify CFL < 1.0
   - Ensure no NaN/Inf values

2. **Simulation unstable**
   - Reduce CFL number
   - Check initial conditions
   - Verify bed elevation smoothness

3. **Wrong results**
   - Check test case ID
   - Verify grid resolution sufficient
   - Compare with published results

### Performance Issues

1. **Slow execution**
   - Check GPU utilization (nvidia-smi)
   - Verify CUDA-Aware MPI working
   - Try larger grid size

2. **Out of memory**
   - Reduce grid size
   - Use single precision (default)
   - Enable multi-GPU mode

## Regression Testing

### Continuous Integration

A CI/CD pipeline should run:
1. Quick test on every commit
2. Full test suite on pull requests
3. Performance benchmarks weekly

### Acceptance Criteria

All tests must:
- ✓ Complete without crashes
- ✓ Conserve mass (error < 1e-6)
- ✓ Pass physical validity checks
- ✓ Match analytical solutions (where available, error < 10%)
- ✓ Achieve target performance (> 0.1 gigacells/s on modern GPU)

## References

1. Ritter, A. (1892). "Die Fortpflanzung der Wasserwellen"
2. Thacker, W. C. (1981). "Some exact solutions to the nonlinear shallow-water wave equations"
3. MacDonald, I., et al. (1997). "Analysis of the Mac Cormack scheme"
4. Toro, E. F. (2001). "Shock-Capturing Methods for Free-Surface Shallow Flows"
5. Liang, Q., & Marche, F. (2009). "Numerical resolution of well-balanced shallow water equations"

---

**Last Updated**: 2025-01-29
**Version**: 1.0
