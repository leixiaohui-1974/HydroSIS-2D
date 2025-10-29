# HydroSIS-2D User Manual

## Table of Contents
1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Configuration](#configuration)
4. [Running Simulations](#running-simulations)
5. [Visualization](#visualization)
6. [Analysis Tools](#analysis-tools)
7. [Advanced Topics](#advanced-topics)
8. [Troubleshooting](#troubleshooting)

---

## 1. Introduction

HydroSIS-2D is a GPU-accelerated 2D hydrodynamic model for simulating shallow water flows, floods, dam breaks, and other free-surface flow phenomena.

### Key Features
- Multi-GPU support with near-linear scaling
- Second-order accurate numerical schemes
- Wetting and drying capability
- Multiple test cases and validation tools
- VTK output for ParaView visualization
- Python post-processing tools

---

## 2. Getting Started

### System Requirements

**Minimum**:
- NVIDIA GPU with CUDA compute capability ≥ 7.0
- 4 GB GPU memory
- 8 GB system RAM
- Ubuntu 20.04+ or similar Linux distribution

**Recommended**:
- NVIDIA GPU with compute capability ≥ 8.0 (Ampere or newer)
- 8+ GB GPU memory
- 16+ GB system RAM
- Multiple GPUs for large simulations

### Installation

```bash
# Clone repository
git clone https://github.com/your-repo/HydroSIS-2D.git
cd HydroSIS-2D

# Create build directory
mkdir build && cd build

# Configure (single GPU)
cmake .. -DENABLE_MPI=OFF

# Configure (multi-GPU with MPI)
cmake .. -DENABLE_MPI=ON

# Build
make -j$(nproc)

# Test installation
./hydrosis --help
```

### Quick Test

```bash
cd ..
./test/quick_test.sh
```

If successful, you should see output indicating the simulation completed.

---

## 3. Configuration

### Command Line Options

```bash
./hydrosis [options]

Options:
  --test <id>       Test case ID (0-8)
  --nx <value>      Grid size in x-direction
  --ny <value>      Grid size in y-direction
  --cfl <value>     CFL number (0.5-0.9)
  --tend <value>    End time in seconds
  --validate        Enable validation
  --vtk             Enable VTK output
  --multi-gpu       Enable multi-GPU mode
  --config <file>   Load configuration from file
  --help            Show help message
```

### Configuration Files

Create a configuration file (e.g., `my_simulation.ini`):

```ini
# Grid Parameters
nx = 512
ny = 512
dx = 1.0
dy = 1.0

# Physical Parameters
cfl = 0.5
h_dry = 1e-4

# Time Parameters
t_start = 0.0
t_end = 10.0
output_interval = 1.0

# Boundary Conditions
bc_left = 0    # 0: wall, 1: open
bc_right = 0
bc_bottom = 0
bc_top = 0

# Solver Options
riemann_solver = 1   # 1: HLLC (recommended)
slope_limiter = 0    # 0: minmod
order = 2            # 2: second-order MUSCL
```

Load configuration:
```bash
./hydrosis --config my_simulation.ini --test 0
```

---

## 4. Running Simulations

### Example 1: Simple Dam Break

```bash
./hydrosis --test 0 --nx 512 --ny 256 --tend 5.0 --vtk
```

This runs a 1D dam break with VTK output for visualization.

### Example 2: Lake at Rest (Validation)

```bash
./hydrosis --test 5 --nx 256 --ny 256 --tend 10.0 --validate
```

This tests the well-balanced property. Velocities should remain near zero.

### Example 3: Multi-GPU Simulation

```bash
mpirun -np 4 ./hydrosis --multi-gpu --test 1 --nx 2048 --ny 2048 --tend 10.0
```

This runs a 2D circular dam break on 4 GPUs.

### Example 4: Using Configuration File

```bash
./hydrosis --config examples/config_dam_break.ini --test 0 --vtk
```

### Test Case Selection

| ID | Test Case | Description |
|----|-----------|-------------|
| 0 | 1D Dam Break | Ritter's analytical solution |
| 1 | 2D Circular Dam | Radial symmetry test |
| 2 | Partial Dam Break | Asymmetric flow |
| 3 | Thacker's Beach | Wetting/drying benchmark |
| 4 | MacDonald Test | Standard benchmark with friction |
| 5 | Lake at Rest | Well-balanced test |
| 6 | Small Perturbation | Stability test |
| 7 | Flow Over Bump | Transcritical flow |
| 8 | Oblique Jump | 2D shock capturing |

---

## 5. Visualization

### ParaView (Recommended)

1. **Run simulation with VTK output**:
   ```bash
   ./hydrosis --test 0 --vtk --tend 5.0
   ```

2. **Open in ParaView**:
   - Launch ParaView
   - File → Open → Select `output_*.vtk` files
   - Click "Apply"
   - Select field to visualize (depth, velocity, etc.)

3. **Create animation**:
   - View → Animation View
   - Play to preview
   - File → Save Animation

### Python Visualization

```bash
# Plot final snapshot
./tools/visualize.py --field h --output depth.png

# Plot 1D profile
./tools/visualize.py --profile --direction x --output profile.png

# Create animation
./tools/visualize.py --animate --field h --output animation.mp4
```

### Python Options

```bash
./tools/visualize.py [options]

Options:
  --dir <path>        Output directory
  --file <path>       Specific file to plot
  --field <name>      Field to visualize (h, u, v, vel, eta)
  --profile           Plot 1D profile
  --direction <x|y>   Direction for profile
  --animate           Create animation
  --output <file>     Output filename
  --fps <value>       Animation frame rate
```

---

## 6. Analysis Tools

### Mass Conservation

```bash
./tools/analyze_results.py --mass
```

Output:
```
Mass Conservation Analysis:
  Initial mass: 1234.567890 m³
  Final mass:   1234.567891 m³
  Relative error: 1.23e-07 (0.0000%)
```

### Field Statistics

```bash
./tools/analyze_results.py --stats
```

Output:
```
Statistics at t = 5.00 s:

Water Depth (h):
  Min:    0.000000 m
  Max:    8.234567 m
  Mean:   2.456789 m
  Std:    1.234567 m

Velocity Magnitude:
  Min:    0.000000 m/s
  Max:    5.678901 m/s
  Mean:   1.234567 m/s
```

### Analytical Comparison

```bash
./tools/analyze_results.py --analytical --test dam_break
```

Output:
```
Comparison with Analytical Solution:
  L1 error:   0.052341
  L2 error:   0.073421
  Linf error: 0.156789
```

---

## 7. Advanced Topics

### Multi-GPU Considerations

1. **Domain Decomposition**:
   - Automatic 2D decomposition
   - Optimal for square domains
   - Halo exchange overhead ~5-10%

2. **MPI Setup**:
   ```bash
   # Check MPI configuration
   mpirun --version

   # Run on specific GPUs
   CUDA_VISIBLE_DEVICES=0,1,2,3 mpirun -np 4 ./hydrosis --multi-gpu
   ```

3. **Performance Tips**:
   - Use power-of-2 GPU counts (2, 4, 8)
   - Larger grids scale better (>1M cells/GPU)
   - Enable CUDA-Aware MPI for best performance

### Performance Tuning

1. **CFL Number**:
   - Higher CFL = fewer time steps
   - Typical range: 0.5-0.7
   - Too high (>0.9) may cause instability

2. **Grid Resolution**:
   - Coarser grid = faster computation
   - Finer grid = better accuracy
   - Balance based on application needs

3. **GPU Selection**:
   ```bash
   # Check available GPUs
   nvidia-smi

   # Use specific GPU
   CUDA_VISIBLE_DEVICES=0 ./hydrosis --test 0
   ```

### Custom Test Cases

To add a custom test case:

1. Edit `include/test_cases.h`
2. Add new test case struct with `initialize()` method
3. Update `get_test_name()` function
4. Recompile

---

## 8. Troubleshooting

### Common Issues

#### 1. CUDA Not Found

**Error**: `CMake Error: Failed to find nvcc`

**Solution**:
```bash
# Add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Reconfigure
cd build
cmake .. -DCUDAToolkit_ROOT=/usr/local/cuda
```

#### 2. Out of Memory

**Error**: `CUDA error: out of memory`

**Solutions**:
- Reduce grid size: `--nx 256 --ny 256`
- Use multiple GPUs: `--multi-gpu`
- Close other GPU applications

#### 3. MPI Not Working

**Error**: `MPI_Init failed`

**Solution**:
```bash
# Rebuild OpenMPI with CUDA support
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --with-cuda=/usr/local/cuda
make -j$(nproc)
sudo make install
```

#### 4. Simulation Unstable

**Symptoms**: NaN values, negative depths

**Solutions**:
- Reduce CFL number: `--cfl 0.3`
- Check initial conditions
- Verify boundary conditions
- Use first-order scheme temporarily

#### 5. Slow Performance

**Symptoms**: Much slower than expected

**Checks**:
```bash
# Check GPU utilization
nvidia-smi -l 1

# Should show ~90-100% GPU utilization
# If low, check:
# - Grid size too small (<100K cells)
# - CPU-GPU transfer overhead
# - Multiple processes on same GPU
```

### Getting Help

1. **Check documentation**:
   - `docs/TECHNICAL_DESIGN.md`
   - `docs/BUILD_GUIDE.md`
   - `docs/TESTING_GUIDE.md`

2. **Run diagnostics**:
   ```bash
   # Check CUDA
   nvidia-smi
   nvcc --version

   # Check MPI
   mpirun --version

   # Check build
   ldd build/hydrosis
   ```

3. **Report issues**:
   - Include error messages
   - System configuration
   - Steps to reproduce

---

## Appendix A: Output File Formats

### ASCII Output (.dat)

Format: Space-separated values
```
# Time: 5.00 s
# x y h u v z
0.0 0.0 1.234 0.567 0.234 0.000
0.5 0.0 1.456 0.678 0.345 0.000
...
```

Columns:
1. x: X-coordinate (m)
2. y: Y-coordinate (m)
3. h: Water depth (m)
4. u: X-velocity (m/s)
5. v: Y-velocity (m/s)
6. z: Bed elevation (m)

### VTK Output (.vtk)

ParaView-compatible structured grid format.

Available fields:
- `depth`: Water depth (m)
- `velocity`: Velocity vector (m/s)
- `velocity_magnitude`: Speed (m/s)
- `bed_elevation`: Bed elevation (m)
- `surface_elevation`: Water surface elevation (m)
- `froude_number`: Froude number (-)
- `wet_dry`: Wet (1) or dry (0) flag

---

## Appendix B: Performance Benchmarks

### Expected Performance (Single GPU)

| GPU Model | Grid Size | Performance |
|-----------|-----------|-------------|
| RTX 3090 | 1024² | 1.2 gigacells/s |
| A100 | 2048² | 2.5 gigacells/s |
| V100 | 1024² | 0.8 gigacells/s |

### Multi-GPU Scaling

| GPUs | Speedup | Efficiency |
|------|---------|------------|
| 1 | 1.0× | 100% |
| 2 | 1.9× | 95% |
| 4 | 3.7× | 93% |
| 8 | 7.2× | 90% |

---

## Appendix C: References

1. Toro, E. F. (2001). "Shock-Capturing Methods for Free-Surface Shallow Flows"
2. Liang, Q., & Borthwick, A. G. (2009). "Adaptive quadtree simulation of shallow flows"
3. NVIDIA CUDA Programming Guide
4. ParaView User's Guide

---

**Last Updated**: January 2025
**Version**: 1.0
**Contact**: See project repository for support
