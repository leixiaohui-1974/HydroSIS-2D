/**
 * @file flux_kernels.cu
 * @brief CUDA kernels for flux computation
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "../ShallowWaterSolver.cuh"
#include "../RiemannSolver.cuh"

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Compute fluxes in x-direction (first-order)
 *
 * Each thread computes flux at one x-interface.
 * Interface (i, j) connects cells (i-1, j) and (i, j).
 */
__global__ void computeFluxesX_FirstOrder(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ flux_x,  // [3 * (nx+1) * ny]
    int nx, int ny,
    double dx, double dy,
    double g,
    int riemann_type  // 0=HLL, 1=HLLC
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x-interface index [0, nx]
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y-cell index [0, ny-1]

    if (i > nx || j >= ny) return;

    // Boundary fluxes (wall BC: zero flux)
    if (i == 0 || i == nx) {
        int idx = i + j * (nx + 1);
        flux_x[3 * idx + 0] = 0.0;  // Mass flux
        flux_x[3 * idx + 1] = 0.0;  // x-momentum flux
        flux_x[3 * idx + 2] = 0.0;  // y-momentum flux
        return;
    }

    // Left and right cell indices
    int iL = i - 1;
    int iR = i;
    int cell_L = iL + j * nx;
    int cell_R = iR + j * nx;

    // Left state
    double hL = h[cell_L];
    double uL = u[cell_L];
    double vL = v[cell_L];

    // Right state
    double hR = h[cell_R];
    double uR = u[cell_R];
    double vR = v[cell_R];

    // Compute flux using Riemann solver
    double F_mass, F_mom_x, F_mom_y;

    // Normal direction: (1, 0) for x-interface
    hllc_solver_2d(hL, uL, vL, hR, uR, vR,
                   1.0, 0.0,  // nx, ny
                   g,
                   F_mass, F_mom_x, F_mom_y);

    // Store fluxes
    int idx = i + j * (nx + 1);
    flux_x[3 * idx + 0] = F_mass;
    flux_x[3 * idx + 1] = F_mom_x;
    flux_x[3 * idx + 2] = F_mom_y;
}

/**
 * @brief Compute fluxes in y-direction (first-order)
 */
__global__ void computeFluxesY_FirstOrder(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ flux_y,  // [3 * nx * (ny+1)]
    int nx, int ny,
    double dx, double dy,
    double g,
    int riemann_type
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // x-cell index [0, nx-1]
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // y-interface index [0, ny]

    if (i >= nx || j > ny) return;

    // Boundary fluxes (wall BC: zero flux)
    if (j == 0 || j == ny) {
        int idx = i + j * nx;
        flux_y[3 * idx + 0] = 0.0;
        flux_y[3 * idx + 1] = 0.0;
        flux_y[3 * idx + 2] = 0.0;
        return;
    }

    // Bottom and top cell indices
    int jB = j - 1;
    int jT = j;
    int cell_B = i + jB * nx;
    int cell_T = i + jT * nx;

    // Bottom state
    double hB = h[cell_B];
    double uB = u[cell_B];
    double vB = v[cell_B];

    // Top state
    double hT = h[cell_T];
    double uT = u[cell_T];
    double vT = v[cell_T];

    // Compute flux
    double F_mass, F_mom_x, F_mom_y;

    // Normal direction: (0, 1) for y-interface
    hllc_solver_2d(hB, uB, vB, hT, uT, vT,
                   0.0, 1.0,  // nx, ny
                   g,
                   F_mass, F_mom_x, F_mom_y);

    // Store fluxes
    int idx = i + j * nx;
    flux_y[3 * idx + 0] = F_mass;
    flux_y[3 * idx + 1] = F_mom_x;
    flux_y[3 * idx + 2] = F_mom_y;
}

/**
 * @brief MUSCL reconstruction kernel (second-order)
 *
 * Computes left and right states at interfaces using MUSCL reconstruction.
 * Uses shared memory for efficient neighbor access.
 */
__global__ void computeFluxesX_MUSCL(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ flux_x,
    int nx, int ny,
    double dx, double dy,
    double g,
    int limiter_type  // 0=none, 1=minmod, 2=vanleer, 3=superbee
) {
    // Shared memory for efficient access (including ghost cells)
    extern __shared__ double s_data[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;

    // TODO: Implement MUSCL reconstruction
    // This is a placeholder - full implementation requires:
    // 1. Load data into shared memory with halo
    // 2. Compute gradients
    // 3. Apply limiter
    // 4. Reconstruct left/right states
    // 5. Call Riemann solver
    // 6. Store fluxes

    // For now, fall back to first-order
    if (i > nx || j >= ny) return;

    // Simple first-order for now
    // (Full MUSCL implementation to be added)
}

} // namespace cuda
} // namespace hydrosis2d
