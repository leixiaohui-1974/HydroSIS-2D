# HydroSIS-2D: Multi-GPU Accelerated 2D Hydrodynamic Model

## Overview
HydroSIS-2D is a high-performance GPU-accelerated two-dimensional hydrodynamic model for simulating flood inundation, dam breaks, and rainfall-runoff processes. It achieves 640-900x speedup on multi-GPU systems.

## Key Features
- **Multi-GPU Acceleration**: CUDA + MPI for distributed parallel computing
- **Advanced Numerics**: 2D Shallow Water Equations with MUSCL-Hancock scheme
- **Adaptive Time Stepping**: Local Time Step (LTS) for enhanced efficiency
- **Scalability**: Near-perfect weak/strong scaling up to 8+ GPUs
- **Performance**: >1.2 gigacells/s on 4 GPUs

## Technical Highlights
- **Solver**: Finite Volume Method with HLLC Riemann solver
- **Domain Decomposition**: METIS-based intelligent partitioning
- **Communication**: CUDA-Aware MPI for GPU-to-GPU direct transfer
- **Optimization**: CUDA Streams, Shared Memory, Warp-level primitives

## Requirements
- CUDA Toolkit >= 11.0
- MPI (OpenMPI or MPICH with CUDA support)
- GCC/G++ >= 7.0
- CMake >= 3.18

## Directory Structure
```
HydroSIS-2D/
├── src/               # Source code
│   ├── cuda/          # CUDA kernels
│   ├── solver/        # Core solver
│   ├── parallel/      # MPI parallelization
│   └── utils/         # Utilities
├── include/           # Header files
├── examples/          # Test cases
├── docs/              # Documentation
└── CMakeLists.txt     # Build configuration
```

## Performance Benchmarks
- Single GPU: 75-200x speedup vs CPU
- 4 GPUs: 640x speedup
- 8 GPUs: 900x+ speedup

## Citation
GPU-accelerated 2D hydrodynamic model based on shallow water equations with advanced numerical schemes (2025)