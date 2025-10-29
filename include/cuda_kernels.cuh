#ifndef CUDA_KERNELS_CUH
#define CUDA_KERNELS_CUH

#include "hydrosis_types.h"
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

namespace CudaKernels {

// ============================================================================
// Riemann Solver Kernels
// ============================================================================

/**
 * @brief HLLC Riemann solver for computing interface fluxes
 * @param U_L Left state (conservative variables)
 * @param U_R Right state (conservative variables)
 * @param z_L Left bed elevation
 * @param z_R Right bed elevation
 * @param g Gravitational acceleration
 * @param dir Direction (0: x, 1: y)
 * @return Flux vector
 */
__device__ FluxVector hllc_riemann_solver(
    const ConservativeVars& U_L,
    const ConservativeVars& U_R,
    real_t z_L, real_t z_R,
    real_t g, int dir);

/**
 * @brief Compute conservative flux F(U) in x-direction
 */
__device__ FluxVector compute_flux_x(
    const ConservativeVars& U, real_t g);

/**
 * @brief Compute conservative flux G(U) in y-direction
 */
__device__ FluxVector compute_flux_y(
    const ConservativeVars& U, real_t g);

// ============================================================================
// Reconstruction Kernels (MUSCL)
// ============================================================================

/**
 * @brief Minmod slope limiter
 */
__device__ real_t minmod(real_t a, real_t b);

/**
 * @brief Superbee slope limiter
 */
__device__ real_t superbee(real_t a, real_t b);

/**
 * @brief MUSCL reconstruction for second-order accuracy
 * @param U_m Cell at i-1
 * @param U_c Cell at i (center)
 * @param U_p Cell at i+1
 * @param U_L Left reconstructed state (output)
 * @param U_R Right reconstructed state (output)
 * @param limiter Slope limiter type
 */
__device__ void muscl_reconstruction(
    const ConservativeVars& U_m,
    const ConservativeVars& U_c,
    const ConservativeVars& U_p,
    ConservativeVars& U_L,
    ConservativeVars& U_R,
    int limiter);

// ============================================================================
// Main Solver Kernels
// ============================================================================

/**
 * @brief Compute maximum wave speed for CFL condition
 * @param cells Grid cells array
 * @param nx, ny Grid dimensions
 * @param g Gravitational acceleration
 * @param max_speed Output maximum wave speed
 */
__global__ void compute_max_wavespeed_kernel(
    const CellData* cells,
    int nx, int ny,
    real_t g,
    real_t* max_speed);

/**
 * @brief Compute time step based on CFL condition
 * @param cells Grid cells array
 * @param nx, ny Grid dimensions
 * @param dx, dy Grid spacing
 * @param cfl CFL number
 * @param g Gravitational acceleration
 * @param dt_global Output global time step
 */
__global__ void compute_timestep_kernel(
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t cfl, real_t g,
    real_t* dt_global);

/**
 * @brief Compute local time steps for LTS (Local Time Stepping)
 */
__global__ void compute_local_timestep_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t cfl, real_t g);

/**
 * @brief Update cell states using finite volume method
 * @param cells Current cell data
 * @param cells_new New cell data (output)
 * @param nx, ny Grid dimensions
 * @param dx, dy Grid spacing
 * @param dt Time step
 * @param params Simulation parameters
 */
__global__ void update_cells_kernel(
    const CellData* cells,
    CellData* cells_new,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t dt,
    const SimParams params);

/**
 * @brief Apply source terms (bed slope, friction)
 */
__global__ void apply_source_terms_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t dt,
    real_t g);

/**
 * @brief Apply Manning's friction
 */
__device__ void apply_manning_friction(
    ConservativeVars& U,
    real_t n,
    real_t dt,
    real_t g);

/**
 * @brief Apply boundary conditions
 */
__global__ void apply_boundary_conditions_kernel(
    CellData* cells,
    int nx, int ny,
    const int* bc_types);

/**
 * @brief Wet/dry treatment
 */
__global__ void wet_dry_treatment_kernel(
    CellData* cells,
    int nx, int ny,
    real_t h_dry);

// ============================================================================
// Utility Kernels
// ============================================================================

/**
 * @brief Initialize grid with test case
 */
__global__ void initialize_grid_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin, real_t ymin,
    int test_case);

/**
 * @brief Copy cell data between buffers
 */
__global__ void copy_cells_kernel(
    const CellData* src,
    CellData* dst,
    int n);

/**
 * @brief Compute conservative to primitive variables
 */
__device__ PrimitiveVars conservative_to_primitive(
    const ConservativeVars& U);

/**
 * @brief Compute primitive to conservative variables
 */
__device__ ConservativeVars primitive_to_conservative(
    const PrimitiveVars& W);

} // namespace CudaKernels

#endif // CUDA_KERNELS_CUH
