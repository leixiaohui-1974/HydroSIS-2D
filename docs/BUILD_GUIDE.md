# HydroSIS-2D Build Guide

## Prerequisites

### Required Software

1. **CUDA Toolkit** (>= 11.0)
   ```bash
   # Check CUDA version
   nvcc --version
   ```

2. **C++ Compiler** (GCC >= 7.0 or Clang >= 5.0)
   ```bash
   g++ --version
   ```

3. **CMake** (>= 3.18)
   ```bash
   cmake --version
   ```

4. **MPI** (Optional, for multi-GPU)
   - OpenMPI with CUDA support, or
   - MPICH with CUDA support
   ```bash
   mpirun --version
   ```

### Hardware Requirements

**Minimum:**
- NVIDIA GPU with compute capability >= 7.0 (Volta or newer)
- 4 GB GPU memory
- 8 GB system RAM

**Recommended:**
- NVIDIA GPU with compute capability >= 8.0 (Ampere or newer)
- 8+ GB GPU memory
- Multiple GPUs for parallel execution
- InfiniBand for multi-node setups

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/your-repo/HydroSIS-2D.git
cd HydroSIS-2D
```

### Step 2: Configure Build

#### Single GPU (no MPI)

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=OFF
```

#### Multi-GPU with MPI

```bash
mkdir build
cd build
cmake .. -DENABLE_MPI=ON
```

#### Advanced Options

```bash
# Use double precision
cmake .. -DUSE_DOUBLE_PRECISION=ON

# Specify CUDA architecture (adjust for your GPU)
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86"

# Common architectures:
#   70 - V100
#   75 - RTX 2080, T4
#   80 - A100
#   86 - RTX 3090
#   89 - RTX 4090
#   90 - H100

# Release build (optimized)
cmake .. -DCMAKE_BUILD_TYPE=Release

# Debug build
cmake .. -DCMAKE_BUILD_TYPE=Debug
```

### Step 3: Build

```bash
make -j$(nproc)
```

This will create the `hydrosis` executable in the `build` directory.

### Step 4: Install (Optional)

```bash
sudo make install
```

This installs the executable to `/usr/local/bin/hydrosis`.

## Verification

### Check GPU

```bash
nvidia-smi
```

### Test Single GPU

```bash
./hydrosis --nx 512 --ny 512 --tend 1.0 --test 0
```

Expected output:
```
╔════════════════════════════════════════════════════════════╗
║         HydroSIS-2D: GPU-Accelerated 2D Hydrodynamic       ║
║                     Single GPU Mode                         ║
╚════════════════════════════════════════════════════════════╝

Grid size: 512 x 512
...
Performance: X.XXX gigacells/s
```

### Test Multi-GPU (if MPI enabled)

```bash
mpirun -np 2 ./hydrosis --multi-gpu --nx 1024 --ny 1024 --tend 1.0 --test 1
```

Expected output:
```
Multi-GPU initialization:
  Number of MPI processes: 2
  Number of GPUs: X
Rank 0 on hostname using GPU 0 (GPU Name)
Rank 1 on hostname using GPU 1 (GPU Name)
...
```

## Troubleshooting

### Problem: CUDA not found

**Solution:**
```bash
# Set CUDA path
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

### Problem: MPI not found

**Solution:**
```bash
# Install OpenMPI
sudo apt-get install libopenmpi-dev openmpi-bin

# Or build from source with CUDA support
wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.gz
tar xzf openmpi-4.1.5.tar.gz
cd openmpi-4.1.5
./configure --with-cuda=/usr/local/cuda
make -j$(nproc)
sudo make install
```

### Problem: CMake version too old

**Solution:**
```bash
# Install newer CMake via pip
pip install cmake --upgrade

# Or download binary from cmake.org
```

### Problem: Compute capability error

**Error:** `nvcc fatal : Unsupported gpu architecture 'compute_XX'`

**Solution:**
```bash
# Check your GPU compute capability
nvidia-smi --query-gpu=compute_cap --format=csv

# Update CMakeLists.txt:
set(CMAKE_CUDA_ARCHITECTURES XX)  # Replace XX with your compute capability
```

### Problem: Out of memory

**Error:** `CUDA error: out of memory`

**Solution:**
- Reduce grid size: `--nx 256 --ny 256`
- Use single precision (default)
- Close other GPU applications

### Problem: MPI_CUDA not working

**Error:** MPI cannot access GPU memory directly

**Solution:**
1. Ensure MPI was built with CUDA support
2. Check if CUDA-Aware MPI is enabled:
   ```bash
   ompi_info --parsable --all | grep mpi_built_with_cuda_support:value
   ```
3. Rebuild OpenMPI with `--with-cuda` flag

## Performance Tuning

### 1. Grid Size

Optimal grid size depends on GPU memory:
- **4 GB GPU:** up to 2048×2048
- **8 GB GPU:** up to 4096×4096
- **16 GB GPU:** up to 8192×8192

### 2. CFL Number

- Higher CFL = larger timesteps = faster simulation
- Too high CFL = instability
- Recommended: 0.5-0.7
- Maximum stable: ~0.9

### 3. Block Size

Default: 16×16 (256 threads/block)
- Optimal for most GPUs
- Can experiment with 8×8 or 32×32

To change, edit `get_kernel_dims()` in `hydrosis_solver.cpp`:
```cpp
block = dim3(32, 32);  // Try different values
```

### 4. Multi-GPU Scaling

For best performance:
- Use power-of-2 GPU counts (2, 4, 8)
- Ensure GPUs are on same node (NVLink > PCIe > Network)
- Large domains (>1M cells/GPU) scale better

## Example Workflows

### Small Test Run

```bash
./hydrosis --nx 256 --ny 256 --cfl 0.5 --tend 5.0 --test 0
```

### Large Single-GPU Run

```bash
./hydrosis --nx 4096 --ny 4096 --cfl 0.6 --tend 10.0 --test 1
```

### Multi-GPU Production Run

```bash
mpirun -np 4 ./hydrosis --multi-gpu --nx 8192 --ny 8192 --cfl 0.6 --tend 100.0 --test 1
```

### Multi-Node Cluster

```bash
# Create hostfile
cat > hostfile << EOF
node1 slots=4
node2 slots=4
EOF

# Run on 2 nodes, 8 GPUs total
mpirun -np 8 --hostfile hostfile ./hydrosis --multi-gpu --nx 16384 --ny 16384 --tend 100.0
```

## Benchmarking

### Single GPU Benchmark

```bash
#!/bin/bash
for size in 512 1024 2048 4096; do
    echo "Grid size: $size x $size"
    ./hydrosis --nx $size --ny $size --tend 1.0 --test 0 | grep "Performance"
done
```

### Multi-GPU Scaling Test

```bash
#!/bin/bash
for ngpu in 1 2 4 8; do
    echo "Number of GPUs: $ngpu"
    mpirun -np $ngpu ./hydrosis --multi-gpu --nx 4096 --ny 4096 --tend 1.0 --test 1 | grep "Performance"
done
```

---

**For more help, consult TECHNICAL_DESIGN.md or open an issue on GitHub.**
