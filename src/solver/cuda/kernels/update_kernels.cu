/**
 * @file update_kernels.cu
 * @brief CUDA kernels for updating conservative variables
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "../ShallowWaterSolver.cuh"

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Update conservative variables using forward Euler
 *
 * Implements: Q^(n+1) = Q^n - dt/dx * (F_x^(n+1/2) - F_x^(n-1/2))
 *                            - dt/dy * (F_y^(n+1/2) - F_y^(n-1/2))
 *
 * @param h Water depth [nx*ny]
 * @param hu X-momentum [nx*ny]
 * @param hv Y-momentum [nx*ny]
 * @param flux_x X-direction fluxes [3*(nx+1)*ny]
 * @param flux_y Y-direction fluxes [3*nx*(ny+1)]
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Cell spacing in x
 * @param dy Cell spacing in y
 * @param dt Time step size
 */
__global__ void updateConservativeVariables_Euler(
    double* __restrict__ h,
    double* __restrict__ hu,
    double* __restrict__ hv,
    const double* __restrict__ flux_x,
    const double* __restrict__ flux_y,
    int nx, int ny,
    double dx, double dy,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int cell_idx = i + j * nx;

    // X-direction fluxes (at i and i+1)
    int flux_left  = i + j * (nx + 1);
    int flux_right = (i + 1) + j * (nx + 1);

    double F_left_h  = flux_x[3 * flux_left + 0];
    double F_left_hu = flux_x[3 * flux_left + 1];
    double F_left_hv = flux_x[3 * flux_left + 2];

    double F_right_h  = flux_x[3 * flux_right + 0];
    double F_right_hu = flux_x[3 * flux_right + 1];
    double F_right_hv = flux_x[3 * flux_right + 2];

    // Y-direction fluxes (at j and j+1)
    int flux_bottom = i + j * nx;
    int flux_top    = i + (j + 1) * nx;

    double G_bottom_h  = flux_y[3 * flux_bottom + 0];
    double G_bottom_hu = flux_y[3 * flux_bottom + 1];
    double G_bottom_hv = flux_y[3 * flux_bottom + 2];

    double G_top_h  = flux_y[3 * flux_top + 0];
    double G_top_hu = flux_y[3 * flux_top + 1];
    double G_top_hv = flux_y[3 * flux_top + 2];

    // Flux differences
    double dF_h  = F_right_h  - F_left_h;
    double dF_hu = F_right_hu - F_left_hu;
    double dF_hv = F_right_hv - F_left_hv;

    double dG_h  = G_top_h  - G_bottom_h;
    double dG_hu = G_top_hu - G_bottom_hu;
    double dG_hv = G_top_hv - G_bottom_hv;

    // Update conservative variables (forward Euler)
    h[cell_idx]  -= dt * (dF_h / dx  + dG_h / dy);
    hu[cell_idx] -= dt * (dF_hu / dx + dG_hu / dy);
    hv[cell_idx] -= dt * (dF_hv / dx + dG_hv / dy);

    // Positivity preserving: ensure h >= 0
    if (h[cell_idx] < 0.0) {
        h[cell_idx] = 0.0;
        hu[cell_idx] = 0.0;
        hv[cell_idx] = 0.0;
    }
}

/**
 * @brief Update conservative variables using RK2 (predictor step)
 *
 * Computes intermediate state: Q* = Q^n - dt * L(Q^n)
 * where L is the spatial operator (flux divergence)
 *
 * @param h Current water depth
 * @param hu Current x-momentum
 * @param hv Current y-momentum
 * @param h_temp Temporary storage for Q*
 * @param hu_temp Temporary storage
 * @param hv_temp Temporary storage
 * @param flux_x X-fluxes
 * @param flux_y Y-fluxes
 * @param nx, ny Grid dimensions
 * @param dx, dy Cell spacing
 * @param dt Time step
 */
__global__ void updateConservativeVariables_RK2_Predictor(
    const double* __restrict__ h,
    const double* __restrict__ hu,
    const double* __restrict__ hv,
    double* __restrict__ h_temp,
    double* __restrict__ hu_temp,
    double* __restrict__ hv_temp,
    const double* __restrict__ flux_x,
    const double* __restrict__ flux_y,
    int nx, int ny,
    double dx, double dy,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int cell_idx = i + j * nx;

    // Compute flux divergence (same as Euler)
    int flux_left  = i + j * (nx + 1);
    int flux_right = (i + 1) + j * (nx + 1);
    int flux_bottom = i + j * nx;
    int flux_top    = i + (j + 1) * nx;

    double dF_h  = flux_x[3 * flux_right + 0] - flux_x[3 * flux_left + 0];
    double dF_hu = flux_x[3 * flux_right + 1] - flux_x[3 * flux_left + 1];
    double dF_hv = flux_x[3 * flux_right + 2] - flux_x[3 * flux_left + 2];

    double dG_h  = flux_y[3 * flux_top + 0] - flux_y[3 * flux_bottom + 0];
    double dG_hu = flux_y[3 * flux_top + 1] - flux_y[3 * flux_bottom + 1];
    double dG_hv = flux_y[3 * flux_top + 2] - flux_y[3 * flux_bottom + 2];

    // RK2 predictor: Q* = Q^n - dt * L(Q^n)
    h_temp[cell_idx]  = h[cell_idx]  - dt * (dF_h / dx  + dG_h / dy);
    hu_temp[cell_idx] = hu[cell_idx] - dt * (dF_hu / dx + dG_hu / dy);
    hv_temp[cell_idx] = hv[cell_idx] - dt * (dF_hv / dx + dG_hv / dy);

    // Positivity preserving
    if (h_temp[cell_idx] < 0.0) {
        h_temp[cell_idx] = 0.0;
        hu_temp[cell_idx] = 0.0;
        hv_temp[cell_idx] = 0.0;
    }
}

/**
 * @brief Update conservative variables using RK2 (corrector step)
 *
 * Computes final state: Q^(n+1) = Q^n - dt/2 * (L(Q^n) + L(Q*))
 *
 * @param h Current water depth
 * @param hu Current x-momentum
 * @param hv Current y-momentum
 * @param flux_x_n X-fluxes at time n
 * @param flux_y_n Y-fluxes at time n
 * @param flux_x_star X-fluxes at intermediate state
 * @param flux_y_star Y-fluxes at intermediate state
 * @param nx, ny Grid dimensions
 * @param dx, dy Cell spacing
 * @param dt Time step
 */
__global__ void updateConservativeVariables_RK2_Corrector(
    double* __restrict__ h,
    double* __restrict__ hu,
    double* __restrict__ hv,
    const double* __restrict__ flux_x_n,
    const double* __restrict__ flux_y_n,
    const double* __restrict__ flux_x_star,
    const double* __restrict__ flux_y_star,
    int nx, int ny,
    double dx, double dy,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int cell_idx = i + j * nx;

    // Indices for fluxes
    int flux_left  = i + j * (nx + 1);
    int flux_right = (i + 1) + j * (nx + 1);
    int flux_bottom = i + j * nx;
    int flux_top    = i + (j + 1) * nx;

    // L(Q^n) - flux divergence at time n
    double L_h_n  = (flux_x_n[3 * flux_right + 0] - flux_x_n[3 * flux_left + 0]) / dx
                  + (flux_y_n[3 * flux_top + 0] - flux_y_n[3 * flux_bottom + 0]) / dy;

    double L_hu_n = (flux_x_n[3 * flux_right + 1] - flux_x_n[3 * flux_left + 1]) / dx
                  + (flux_y_n[3 * flux_top + 1] - flux_y_n[3 * flux_bottom + 1]) / dy;

    double L_hv_n = (flux_x_n[3 * flux_right + 2] - flux_x_n[3 * flux_left + 2]) / dx
                  + (flux_y_n[3 * flux_top + 2] - flux_y_n[3 * flux_bottom + 2]) / dy;

    // L(Q*) - flux divergence at intermediate state
    double L_h_star  = (flux_x_star[3 * flux_right + 0] - flux_x_star[3 * flux_left + 0]) / dx
                     + (flux_y_star[3 * flux_top + 0] - flux_y_star[3 * flux_bottom + 0]) / dy;

    double L_hu_star = (flux_x_star[3 * flux_right + 1] - flux_x_star[3 * flux_left + 1]) / dx
                     + (flux_y_star[3 * flux_top + 1] - flux_y_star[3 * flux_bottom + 1]) / dy;

    double L_hv_star = (flux_x_star[3 * flux_right + 2] - flux_x_star[3 * flux_left + 2]) / dx
                     + (flux_y_star[3 * flux_top + 2] - flux_y_star[3 * flux_bottom + 2]) / dy;

    // RK2 corrector: Q^(n+1) = Q^n - dt/2 * (L(Q^n) + L(Q*))
    h[cell_idx]  -= 0.5 * dt * (L_h_n  + L_h_star);
    hu[cell_idx] -= 0.5 * dt * (L_hu_n + L_hu_star);
    hv[cell_idx] -= 0.5 * dt * (L_hv_n + L_hv_star);

    // Positivity preserving
    if (h[cell_idx] < 0.0) {
        h[cell_idx] = 0.0;
        hu[cell_idx] = 0.0;
        hv[cell_idx] = 0.0;
    }
}

/**
 * @brief Convert conservative variables to primitive variables
 *
 * Computes velocity from momentum: u = hu/h, v = hv/h
 * Handles dry cells (h < dry_threshold) by setting velocity to zero.
 *
 * @param h Water depth [nx*ny]
 * @param hu X-momentum [nx*ny]
 * @param hv Y-momentum [nx*ny]
 * @param u X-velocity output [nx*ny]
 * @param v Y-velocity output [nx*ny]
 * @param nx, ny Grid dimensions
 * @param dry_threshold Dry/wet threshold
 */
__global__ void conservativeToPrimitive(
    const double* __restrict__ h,
    const double* __restrict__ hu,
    const double* __restrict__ hv,
    double* __restrict__ u,
    double* __restrict__ v,
    int nx, int ny,
    double dry_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    if (h[idx] > dry_threshold) {
        u[idx] = hu[idx] / h[idx];
        v[idx] = hv[idx] / h[idx];
    } else {
        u[idx] = 0.0;
        v[idx] = 0.0;
    }
}

/**
 * @brief Convert primitive variables to conservative variables
 *
 * Computes momentum from velocity: hu = h*u, hv = h*v
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param hu X-momentum output [nx*ny]
 * @param hv Y-momentum output [nx*ny]
 * @param nx, ny Grid dimensions
 */
__global__ void primitiveToConservative(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ hu,
    double* __restrict__ hv,
    int nx, int ny
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    hu[idx] = h[idx] * u[idx];
    hv[idx] = h[idx] * v[idx];
}

} // namespace cuda
} // namespace hydrosis2d
