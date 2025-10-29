# HydroSIS-2D Performance Analysis Guide

## Overview

This guide explains how to analyze HydroSIS-2D performance, identify bottlenecks, and validate optimizations.

---

## Performance Metrics

### Key Performance Indicators (KPIs)

1. **Throughput**
   - Cell updates per second
   - Time steps per second
   - Simulation time vs. wall-clock time ratio

2. **Efficiency**
   - GPU utilization (%)
   - Memory bandwidth utilization (%)
   - Parallel efficiency (multi-GPU)

3. **Scalability**
   - Strong scaling (fixed problem, more GPUs)
   - Weak scaling (problem grows with GPUs)
   - Communication overhead

4. **Memory**
   - Total memory usage
   - Peak memory usage
   - Memory transfer volume
   - Cache efficiency

---

## Benchmarking

### Running Benchmarks

**Basic benchmark:**
```bash
chmod +x scripts/benchmark.sh
./scripts/benchmark.sh --sizes "100 500 1000" --gpus "1" --cases "dam_break"
```

**Multi-GPU scaling study:**
```bash
./scripts/benchmark.sh --sizes "1000 2000" --gpus "1 2 4 8" --cases "dam_break"
```

**Application benchmarks:**
```bash
./scripts/benchmark.sh \
    --sizes "500" \
    --gpus "1 2 4" \
    --cases "dam_break urban river"
```

### Visualizing Results

```bash
chmod +x scripts/plot_performance.py
python3 scripts/plot_performance.py --input results/performance/benchmark_*.csv
```

**Output plots:**
- `scaling_efficiency.png` - Speedup and parallel efficiency
- `performance_vs_size.png` - Throughput vs problem size
- `wall_time_comparison.png` - Wall time breakdown
- `memory_usage.png` - Memory consumption
- `benchmark_summary.txt` - Text summary

---

## Profiling with NVIDIA Tools

### Nsight Systems (System-Level Profiling)

**Purpose:** Identify CPU/GPU activity, kernel launches, memory transfers

**Basic profiling:**
```bash
nsys profile -o baseline_profile ./hydrosis --config examples/config_dam_break.ini
```

**Advanced profiling:**
```bash
nsys profile \
    --trace=cuda,nvtx,osrt \
    --sample=cpu \
    --cpuctxsw=true \
    --output=detailed_profile \
    ./hydrosis --config examples/applications/urban_flooding.ini
```

**View results:**
```bash
nsys-ui baseline_profile.qdrep
```

**Key metrics to check:**
- Kernel execution timeline
- Memory transfer timeline
- CPU-GPU synchronization points
- Kernel launch overhead

### Nsight Compute (Kernel-Level Profiling)

**Purpose:** Detailed kernel analysis, occupancy, memory patterns

**Profile specific kernel:**
```bash
ncu \
    --kernel-name compute_fluxes_kernel \
    --launch-skip 100 \
    --launch-count 10 \
    --metrics all \
    --target-processes all \
    -o kernel_profile \
    ./hydrosis --config examples/config_dam_break.ini
```

**Interactive profiling:**
```bash
ncu --mode=launch-and-attach ./hydrosis --config examples/config_dam_break.ini
```

**View results:**
```bash
ncu-ui kernel_profile.ncu-rep
```

**Key metrics:**
- GPU SM efficiency
- Memory bandwidth utilization
- Achieved occupancy
- Register usage
- Shared memory usage
- Cache hit rates

### nvprof (Legacy Profiler)

**Basic profiling:**
```bash
nvprof ./hydrosis --config examples/config_dam_break.ini
```

**Detailed metrics:**
```bash
nvprof --metrics all --analysis-metrics \
    ./hydrosis --config examples/config_dam_break.ini > profile_metrics.txt
```

**Important nvprof metrics:**
- `gld_efficiency` - Global load efficiency
- `gst_efficiency` - Global store efficiency
- `sm_efficiency` - SM efficiency
- `achieved_occupancy` - Achieved occupancy
- `dram_utilization` - DRAM utilization

---

## Identifying Bottlenecks

### Common Bottlenecks

#### 1. Memory Bandwidth Limited

**Symptoms:**
- Low GPU SM efficiency (<50%)
- High memory transfer times
- Low compute-to-memory ratio

**Diagnosis:**
```bash
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed \
    ./hydrosis --config examples/config_dam_break.ini
```

**Solutions:**
- Coalesced memory access
- Shared memory usage
- Reduce global memory transactions
- Texture memory for read-only data

#### 2. Compute Bound

**Symptoms:**
- High GPU SM efficiency (>80%)
- Many arithmetic operations per memory access
- Low memory bandwidth utilization

**Diagnosis:**
```bash
ncu --metrics smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct \
    ./hydrosis --config examples/config_dam_break.ini
```

**Solutions:**
- Use intrinsic functions (fast math)
- Reduce register pressure
- Optimize arithmetic operations
- Consider lower precision where acceptable

#### 3. Launch Overhead

**Symptoms:**
- Many small kernels
- High CPU time in kernel launches
- Large gap between kernel executions

**Diagnosis:**
- Look at Nsight Systems timeline
- Count kernel launches
- Measure inter-kernel gaps

**Solutions:**
- Kernel fusion
- Persistent kernels
- Grid-stride loops
- Reduce synchronization points

#### 4. Low Occupancy

**Symptoms:**
- Achieved occupancy <50%
- High register usage per thread
- Large shared memory usage

**Diagnosis:**
```bash
ncu --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
    ./hydrosis --config examples/config_dam_break.ini
```

**Solutions:**
- Reduce register usage (simplify kernel)
- Reduce shared memory (if excessive)
- Adjust block size
- Split complex kernels

#### 5. Divergent Branches

**Symptoms:**
- Low warp execution efficiency
- Many if-else branches
- Irregular control flow

**Diagnosis:**
```bash
ncu --metrics smsp__sass_average_branch_targets_threads_uniform.pct \
    ./hydrosis --config examples/config_dam_break.ini
```

**Solutions:**
- Minimize branching
- Use predication instead of branches
- Reorder data for better thread coherence
- Separate kernels for different cases

---

## Performance Analysis Workflow

### Step 1: Establish Baseline

```bash
# Run benchmark suite
./scripts/benchmark.sh --sizes "500 1000" --gpus "1" --output results/baseline.csv

# Profile with Nsight Systems
nsys profile -o baseline ./hydrosis --config examples/config_dam_break.ini

# Generate report
python3 scripts/plot_performance.py --input results/baseline.csv
```

**Document:**
- Throughput (cells/s)
- GPU utilization (%)
- Memory bandwidth (%)
- Kernel times

### Step 2: Identify Top Bottlenecks

```bash
# Profile all kernels
ncu --set full -o detailed ./hydrosis --config examples/config_dam_break.ini

# Analyze timeline
nsys-ui baseline.qdrep
```

**Look for:**
- Longest running kernels (optimize first)
- Memory-bound kernels (bandwidth limited)
- Low occupancy kernels
- Excessive memory transfers
- Synchronization overhead

### Step 3: Prioritize Optimizations

**Scoring formula:**
```
Priority = (Kernel Time %) × (Optimization Potential %)
```

**Example:**
- `compute_fluxes_kernel`: 40% time, 50% potential → Priority: 20
- `muscl_reconstruction`: 25% time, 70% potential → Priority: 17.5
- `time_integration`: 20% time, 30% potential → Priority: 6

**Focus on top 2-3 kernels for maximum impact.**

### Step 4: Implement Optimization

**For each optimization:**

1. **Create branch:**
   ```bash
   git checkout -b optimize-flux-kernel
   ```

2. **Make changes:**
   - Modify kernel code
   - Add comments explaining changes
   - Keep old code in comments for reference

3. **Test correctness:**
   ```bash
   # Run all validation tests
   ./scripts/run_tests.sh
   ```

4. **Measure performance:**
   ```bash
   ./scripts/benchmark.sh --sizes "500" --gpus "1" --output results/optimized.csv
   python3 scripts/compare_performance.py results/baseline.csv results/optimized.csv
   ```

5. **Validate improvement:**
   - Check speedup ≥ 1.05× (5% minimum)
   - Ensure correctness maintained
   - Verify no regressions in other areas

6. **Commit if successful:**
   ```bash
   git add <modified files>
   git commit -m "Optimize flux kernel: 1.3× speedup via coalesced memory access"
   ```

### Step 5: Iterate

Repeat steps 2-4 for next highest priority bottleneck.

---

## Optimization Techniques

### Memory Optimization

#### Coalesced Global Memory Access

**Bad (uncoalesced):**
```cuda
__global__ void bad_kernel(float* data, int stride) {
    int idx = threadIdx.x;
    float val = data[idx * stride];  // Non-contiguous access
}
```

**Good (coalesced):**
```cuda
__global__ void good_kernel(float* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = data[idx];  // Contiguous access
}
```

#### Shared Memory for Reuse

**Bad (multiple global memory reads):**
```cuda
__global__ void bad_kernel(float* data) {
    int idx = threadIdx.x;
    float sum = data[idx] + data[idx+1] + data[idx+2];
}
```

**Good (shared memory):**
```cuda
__global__ void good_kernel(float* data) {
    __shared__ float s_data[BLOCK_SIZE + 2];
    int idx = threadIdx.x;

    // Load into shared memory
    s_data[idx] = data[blockIdx.x * blockDim.x + idx];
    if (idx < 2) s_data[blockDim.x + idx] = data[blockIdx.x * blockDim.x + blockDim.x + idx];
    __syncthreads();

    // Use shared memory
    float sum = s_data[idx] + s_data[idx+1] + s_data[idx+2];
}
```

### Computational Optimization

#### Use Intrinsic Functions

**Bad (slow):**
```cuda
float result = sqrtf(x * x + y * y);
```

**Good (fast):**
```cuda
float result = __fsqrt_rn(x * x + y * y);  // Fast intrinsic
// or
float result = norm2df(x, y);  // Even faster specialized function
```

#### Minimize Branching

**Bad (divergent):**
```cuda
if (condition) {
    result = expensiveOp1();
} else {
    result = expensiveOp2();
}
```

**Good (predication):**
```cuda
float val1 = expensiveOp1();
float val2 = expensiveOp2();
result = condition ? val1 : val2;  // Compiler may use predication
```

### Occupancy Optimization

#### Reduce Register Usage

**Check register usage:**
```bash
nvcc --ptxas-options=-v kernel.cu
```

**Techniques:**
- Simplify complex expressions
- Use smaller data types where possible
- Split large kernels
- Use `__launch_bounds__` directive

#### Adjust Block Size

**Test different sizes:**
```cuda
// Test block sizes: 64, 128, 256, 512, 1024
dim3 blockSize(256);  // Often optimal for most GPUs
dim3 gridSize((N + blockSize.x - 1) / blockSize.x);
kernel<<<gridSize, blockSize>>>(...);
```

---

## Multi-GPU Performance

### Strong Scaling Analysis

**Run scaling study:**
```bash
./scripts/benchmark.sh --sizes "2000" --gpus "1 2 4 8"
```

**Calculate efficiency:**
```
Speedup(N) = T(1) / T(N)
Efficiency(N) = Speedup(N) / N × 100%
```

**Ideal: Efficiency > 90% for 2-4 GPUs**

### Weak Scaling Analysis

**Scale problem with GPUs:**
```bash
# 1 GPU: 1000×1000
# 2 GPUs: 1414×1414 (2× cells)
# 4 GPUs: 2000×2000 (4× cells)
# 8 GPUs: 2828×2828 (8× cells)
```

**Ideal: Wall time remains constant**

### Communication Profiling

**Profile MPI communication:**
```bash
nsys profile \
    --trace=mpi,cuda \
    mpirun -np 4 ./hydrosis --config examples/config_multi_gpu.ini
```

**Key metrics:**
- MPI call frequency
- Data volume per exchange
- Communication time %
- Overlap with computation

**Optimization strategies:**
- Minimize halo exchange frequency
- Overlap communication with computation
- Use CUDA-aware MPI
- Compress halo data if bandwidth limited

---

## Validation

### Correctness Checks

**After every optimization:**

1. **Run standard test cases:**
   ```bash
   ./scripts/run_tests.sh
   ```

2. **Check mass conservation:**
   ```python
   # Should be < 1e-10
   mass_error = abs(final_mass - initial_mass) / initial_mass
   ```

3. **Compare with baseline:**
   ```bash
   # Relative error should be < 1e-6
   python3 scripts/compare_solutions.py baseline.vtk optimized.vtk
   ```

4. **Visual inspection:**
   - Load results in ParaView
   - Check for artifacts
   - Verify physical behavior

### Performance Regression Testing

**Automated checks:**
```bash
# Run before/after comparison
python3 scripts/regression_test.py \
    --baseline results/baseline.csv \
    --current results/optimized.csv \
    --threshold 0.95  # Flag if performance drops below 95%
```

**Monitor:**
- Performance metrics
- Memory usage
- Numerical accuracy
- Multi-GPU efficiency

---

## Reporting

### Performance Report Template

```markdown
## Optimization: [Brief Description]

### Baseline Performance
- Throughput: X.XX M cells/s
- GPU Utilization: XX%
- Memory Bandwidth: XX%

### Bottleneck Identified
- [Description of bottleneck]
- [Profiling evidence]

### Optimization Applied
- [Description of changes]
- [Code snippets if relevant]

### Results
- New Throughput: X.XX M cells/s
- Speedup: X.XX×
- GPU Utilization: XX%
- Memory Bandwidth: XX%

### Validation
- ✓ All tests pass
- ✓ Mass conservation < 1e-10
- ✓ Solution error < 1e-6

### Impact
- [Overall impact on application performance]
```

---

## Tools Summary

| Tool | Purpose | Command |
|------|---------|---------|
| benchmark.sh | Automated benchmarking | `./scripts/benchmark.sh` |
| plot_performance.py | Visualization | `python3 scripts/plot_performance.py` |
| Nsight Systems | System profiling | `nsys profile -o out ./hydrosis` |
| Nsight Compute | Kernel profiling | `ncu -o out ./hydrosis` |
| nvprof | Legacy profiling | `nvprof ./hydrosis` |

---

## Best Practices

1. **Always establish baseline** before optimizing
2. **Optimize highest impact** bottlenecks first
3. **Validate correctness** after every change
4. **Measure performance** quantitatively
5. **Use version control** (Git branches)
6. **Document optimizations** clearly
7. **Test on target hardware** (production GPUs)
8. **Consider maintainability** vs performance

---

## References

- NVIDIA CUDA Best Practices Guide
- Nsight Systems Documentation
- Nsight Compute Documentation
- "CUDA Handbook" by Nicholas Wilt
- "Programming Massively Parallel Processors" by Kirk & Hwu

---

**Last Updated:** 2025-10-29
**Version:** 1.0
