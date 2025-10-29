# HydroSIS-2D Technical Design Document

## 1. Introduction

HydroSIS-2D is a state-of-the-art GPU-accelerated 2D hydrodynamic model designed for high-performance flood simulation. Based on the latest research (2023-2025), it implements the most advanced techniques for multi-GPU parallel computing.

## 2. Mathematical Model

### 2.1 Governing Equations

The model solves the 2D Shallow Water Equations (SWE) in conservative form:

```
∂U/∂t + ∂F(U)/∂x + ∂G(U)/∂y = S(U)
```

Where:
- **U** = [h, qx, qy]ᵀ - Conservative variables (depth, x-discharge, y-discharge)
- **F** - Flux in x-direction
- **G** - Flux in y-direction
- **S** - Source terms (bed slope, friction)

### 2.2 Flux Functions

**X-direction flux:**
```
F(U) = [qx, qx²/h + ½gh², qxqy/h]ᵀ
```

**Y-direction flux:**
```
G(U) = [qy, qxqy/h, qy²/h + ½gh²]ᵀ
```

### 2.3 Source Terms

1. **Bed slope:** -gh ∂z/∂x, -gh ∂z/∂y
2. **Manning friction:** Sf = n²|V|V / h^(4/3)

## 3. Numerical Methods

### 3.1 Finite Volume Method

- **Discretization:** Cell-centered finite volume method
- **Grid:** Structured Cartesian grid
- **Order:** 1st or 2nd order (MUSCL reconstruction)

### 3.2 MUSCL-Hancock Scheme

For second-order accuracy in space:
1. **Reconstruction:** Piecewise linear reconstruction with slope limiters
2. **Limiters:** Minmod, Superbee, MC limiter
3. **Evolution:** Half-time step predictor

### 3.3 HLLC Riemann Solver

The HLLC (Harten-Lax-van Leer-Contact) solver provides:
- **Robustness:** Handles dry/wet interfaces
- **Accuracy:** Resolves contact discontinuities
- **Efficiency:** Suitable for GPU parallelization

Wave speed estimates using Roe averages:
```
S_L = min(u_L - c_L, u_roe - c_roe)
S_R = max(u_R + c_R, u_roe + c_roe)
S_* = (S_R u_R - S_L u_L + ½g(h_L² - h_R²)) / (S_R - S_L)
```

### 3.4 Time Integration

**CFL condition:**
```
dt = CFL × min(dx/(|u| + c), dy/(|v| + c))
```

Where:
- CFL ≈ 0.5-0.9
- c = √(gh) - wave celerity

## 4. GPU Acceleration Strategy

### 4.1 CUDA Kernel Design

#### Memory Hierarchy Optimization
- **Global memory:** Cell data storage
- **Shared memory:** Halo cells, flux computation
- **Registers:** Intermediate calculations

#### Thread Organization
- **Block size:** 16×16 threads (256 threads/block)
- **Grid size:** Covers entire computational domain
- **Warp efficiency:** Minimize divergence

### 4.2 Computational Pipeline

1. **Timestep calculation:** Parallel reduction for global minimum
2. **Flux computation:** Thread-per-cell, HLLC solver
3. **Cell update:** Explicit finite volume update
4. **Source terms:** Bed slope and friction
5. **Boundary conditions:** Ghost cell method

### 4.3 Performance Optimizations

1. **Coalesced memory access:** Aligned data structures
2. **Fast math:** `-use_fast_math` compiler flag
3. **Constant memory:** Simulation parameters
4. **Texture memory:** Bed elevation data (optional)
5. **Warp-level primitives:** Reduction operations

## 5. Multi-GPU Parallelization

### 5.1 Domain Decomposition

**2D Cartesian decomposition:**
- Minimize surface-to-volume ratio
- Balance load across GPUs
- Consider GPU topology

**Optimal decomposition:**
```
npx × npy = num_GPUs
npx/npy ≈ nx/ny (aspect ratio matching)
```

### 5.2 MPI Communication

**CUDA-Aware MPI benefits:**
- Direct GPU-to-GPU transfers
- No host staging required
- Reduced latency

**Communication pattern:**
```
1. Pack halos (CUDA kernel)
2. MPI_Isend/Irecv (non-blocking)
3. Overlap communication with computation
4. MPI_Waitall
5. Unpack halos (CUDA kernel)
```

### 5.3 Halo Exchange

**Halo width:** 2 cells (for 2nd order)

**Exchange directions:**
- Left/Right neighbors (X-direction)
- Bottom/Top neighbors (Y-direction)

**Implementation:**
```cuda
// Pack left halo
pack_left_halo<<<grid, block>>>(cells, halo_buffer);

// Send/Recv with MPI
MPI_Isend(halo_buffer, size, datatype, left_rank, tag, comm, &request);
MPI_Irecv(recv_buffer, size, datatype, left_rank, tag, comm, &request);

// Unpack after receive
unpack_left_halo<<<grid, block>>>(cells, recv_buffer);
```

### 5.4 Load Balancing

- **Static:** Equal domain sizes
- **Dynamic:** Adaptive based on wet cell count (future work)

## 6. Performance Expectations

### 6.1 Single GPU Performance

Based on literature review:
- **Speedup:** 75-200× vs single-thread CPU
- **Throughput:** ~0.3-0.5 gigacells/s (consumer GPU)
- **Memory bandwidth:** 80-90% peak utilization

### 6.2 Multi-GPU Scaling

**Strong scaling:**
- 2 GPUs: 1.8-1.95× speedup
- 4 GPUs: 3.5-3.8× speedup (640× vs CPU)
- 8 GPUs: 6.8-7.5× speedup (900× vs CPU)

**Weak scaling:**
- Near-perfect scaling up to 8 GPUs
- Limited by communication overhead

**Communication overhead:**
- ~5-10% for large domains (>1M cells/GPU)
- ~15-20% for small domains (<100k cells/GPU)

## 7. Validation and Verification

### 7.1 Test Cases

1. **1D Dam Break:**
   - Analytical solution available
   - Tests shock capturing

2. **2D Circular Dam:**
   - Radial symmetry test
   - Tests isotropy

3. **MacDonald Benchmark:**
   - Standard hydrodynamic test
   - Wetting/drying validation

### 7.2 Accuracy Metrics

- **L1 error:** ∫|h_computed - h_exact| dx dy
- **L2 error:** √(∫(h_computed - h_exact)² dx dy)
- **Mass conservation:** ∫h dx dy = constant

## 8. Implementation Details

### 8.1 Data Structures

```cpp
struct ConservativeVars {
    real_t h;   // Depth
    real_t qx;  // X-discharge
    real_t qy;  // Y-discharge
};

struct CellData {
    ConservativeVars U;  // State
    real_t z;            // Bed elevation
    real_t n;            // Manning coefficient
    real_t dt_local;     // Local timestep
    bool is_wet;         // Wet/dry flag
};
```

### 8.2 Memory Layout

**AoS (Array of Structures):**
- Better for: Small strides, complex operations
- Used for: CellData

**Advantages:**
- Cache-friendly for flux computation
- Simpler code

### 8.3 Precision

**Single precision (float):**
- 4 bytes per value
- Sufficient for most applications
- 2× memory bandwidth vs double

**Double precision (double):**
- 8 bytes per value
- Required for: Very large domains, long simulations
- Compile-time option: `-DUSE_DOUBLE_PRECISION`

## 9. Future Enhancements

1. **Local Time Stepping (LTS):**
   - Adaptive dt per cell
   - 2-3× additional speedup

2. **CUDA Dynamic Parallelism:**
   - Nested kernel launches
   - Better load balancing

3. **GPU-Direct RDMA:**
   - Direct GPU-to-GPU via InfiniBand
   - Reduced MPI overhead

4. **Unstructured grids:**
   - Mesh adaptation
   - Complex geometries

5. **Multi-physics coupling:**
   - Sediment transport
   - Water quality
   - Rainfall-runoff

## 10. References

1. GPU-accelerated 2D hydrodynamic models (2023-2025)
2. Multi-GPU shallow water solvers with CUDA+MPI (2024)
3. HLLC Riemann solver for shallow water equations
4. MUSCL-Hancock scheme for high-order accuracy
5. CUDA-Aware MPI for GPU-to-GPU communication

---

**Document Version:** 1.0
**Last Updated:** 2025-01-29
**Author:** Claude (Anthropic)
