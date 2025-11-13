/**
 * @file source_kernels.cu
 * @brief CUDA kernels for source term computation
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "../ShallowWaterSolver.cuh"
#include <cmath>

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Compute bed slope source terms
 *
 * Implements source terms for non-flat bed:
 *   S_h = 0
 *   S_hu = -g * h * ∂z/∂x
 *   S_hv = -g * h * ∂z/∂y
 *
 * Uses central differences for bed slope gradients:
 *   ∂z/∂x ≈ (z[i+1,j] - z[i-1,j]) / (2*dx)
 *   ∂z/∂y ≈ (z[i,j+1] - z[i,j-1]) / (2*dy)
 *
 * For boundary cells, uses one-sided differences.
 *
 * @param h Water depth [nx*ny]
 * @param z Bed elevation [nx*ny]
 * @param S_hu X-momentum source term output [nx*ny]
 * @param S_hv Y-momentum source term output [nx*ny]
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dx Cell spacing in x
 * @param dy Cell spacing in y
 * @param g Gravitational acceleration
 */
__global__ void computeBedSlopeSource(
    const double* __restrict__ h,
    const double* __restrict__ z,
    double* __restrict__ S_hu,
    double* __restrict__ S_hv,
    int nx, int ny,
    double dx, double dy,
    double g
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    // Compute bed slope gradients using central differences
    double dz_dx, dz_dy;

    // X-direction gradient
    if (i == 0) {
        // Left boundary: forward difference
        dz_dx = (z[idx + 1] - z[idx]) / dx;
    } else if (i == nx - 1) {
        // Right boundary: backward difference
        dz_dx = (z[idx] - z[idx - 1]) / dx;
    } else {
        // Interior: central difference
        dz_dx = (z[idx + 1] - z[idx - 1]) / (2.0 * dx);
    }

    // Y-direction gradient
    if (j == 0) {
        // Bottom boundary: forward difference
        dz_dy = (z[idx + nx] - z[idx]) / dy;
    } else if (j == ny - 1) {
        // Top boundary: backward difference
        dz_dy = (z[idx] - z[idx - nx]) / dy;
    } else {
        // Interior: central difference
        dz_dy = (z[idx + nx] - z[idx - nx]) / (2.0 * dy);
    }

    // Bed slope source terms
    S_hu[idx] = -g * h[idx] * dz_dx;
    S_hv[idx] = -g * h[idx] * dz_dy;
}

/**
 * @brief Compute Manning friction source terms
 *
 * Implements Manning friction:
 *   τ_x = ρ * g * n² * |V| * u / h^(4/3)
 *   τ_y = ρ * g * n² * |V| * v / h^(4/3)
 *
 * where:
 *   |V| = sqrt(u² + v²) is velocity magnitude
 *   n is Manning's roughness coefficient
 *
 * Source terms:
 *   S_hu = -g * n² * |V| * hu / h^(4/3)
 *   S_hv = -g * n² * |V| * hv / h^(4/3)
 *
 * Handles dry cells (h < dry_threshold) by setting S = 0.
 *
 * @param h Water depth [nx*ny]
 * @param hu X-momentum [nx*ny]
 * @param hv Y-momentum [nx*ny]
 * @param manning Manning's n coefficient [nx*ny] or scalar
 * @param S_hu X-momentum source term output [nx*ny]
 * @param S_hv Y-momentum source term output [nx*ny]
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param g Gravitational acceleration
 * @param dry_threshold Dry/wet threshold
 * @param uniform_manning If true, manning[0] is used for all cells
 */
__global__ void computeManningFriction(
    const double* __restrict__ h,
    const double* __restrict__ hu,
    const double* __restrict__ hv,
    const double* __restrict__ manning,
    double* __restrict__ S_hu,
    double* __restrict__ S_hv,
    int nx, int ny,
    double g,
    double dry_threshold,
    bool uniform_manning
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    // Handle dry cells
    if (h[idx] < dry_threshold) {
        S_hu[idx] = 0.0;
        S_hv[idx] = 0.0;
        return;
    }

    // Get Manning coefficient
    double n = uniform_manning ? manning[0] : manning[idx];

    // Compute velocity
    double u = hu[idx] / h[idx];
    double v = hv[idx] / h[idx];

    // Velocity magnitude
    double vel_mag = sqrt(u * u + v * v);

    // Manning friction coefficient: C_f = g * n² / h^(1/3)
    double h_13 = cbrt(h[idx]);  // h^(1/3)
    double h_43 = h_13 * h[idx]; // h^(4/3)

    double C_f = g * n * n / h_43;

    // Friction source terms
    S_hu[idx] = -C_f * vel_mag * hu[idx];
    S_hv[idx] = -C_f * vel_mag * hv[idx];
}

/**
 * @brief Apply source terms to conservative variables (explicit)
 *
 * Updates conservative variables by adding source term contribution:
 *   h[i]  += dt * S_h[i]   (typically S_h = 0)
 *   hu[i] += dt * S_hu[i]
 *   hv[i] += dt * S_hv[i]
 *
 * @param h Water depth [nx*ny] (in/out)
 * @param hu X-momentum [nx*ny] (in/out)
 * @param hv Y-momentum [nx*ny] (in/out)
 * @param S_hu X-momentum source terms [nx*ny]
 * @param S_hv Y-momentum source terms [nx*ny]
 * @param nx Number of cells in x
 * @param ny Number of cells in y
 * @param dt Time step size
 */
__global__ void applySourceTerms(
    double* __restrict__ h,
    double* __restrict__ hu,
    double* __restrict__ hv,
    const double* __restrict__ S_hu,
    const double* __restrict__ S_hv,
    int nx, int ny,
    double dt
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    // Apply source terms (S_h = 0 for shallow water)
    hu[idx] += dt * S_hu[idx];
    hv[idx] += dt * S_hv[idx];

    // Positivity preserving
    if (h[idx] < 0.0) {
        h[idx] = 0.0;
        hu[idx] = 0.0;
        hv[idx] = 0.0;
    }
}

/**
 * @brief Combined source term computation and application
 *
 * Computes bed slope + Manning friction source terms and applies them
 * in a single kernel launch for efficiency.
 *
 * Source term splitting (operator splitting):
 *   Q^(n+1) = Q^n + dt * (S_bed + S_friction)
 *
 * @param h Water depth [nx*ny] (in/out)
 * @param hu X-momentum [nx*ny] (in/out)
 * @param hv Y-momentum [nx*ny] (in/out)
 * @param z Bed elevation [nx*ny]
 * @param manning Manning's n [nx*ny or scalar]
 * @param nx, ny Grid dimensions
 * @param dx, dy Cell spacing
 * @param dt Time step
 * @param g Gravitational acceleration
 * @param dry_threshold Dry/wet threshold
 * @param uniform_manning Manning coefficient flag
 */
__global__ void computeAndApplySourceTerms(
    double* __restrict__ h,
    double* __restrict__ hu,
    double* __restrict__ hv,
    const double* __restrict__ z,
    const double* __restrict__ manning,
    int nx, int ny,
    double dx, double dy,
    double dt,
    double g,
    double dry_threshold,
    bool uniform_manning
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = i + j * nx;

    // ===== Bed slope source term =====
    double dz_dx, dz_dy;

    // X-direction gradient
    if (i == 0) {
        dz_dx = (z[idx + 1] - z[idx]) / dx;
    } else if (i == nx - 1) {
        dz_dx = (z[idx] - z[idx - 1]) / dx;
    } else {
        dz_dx = (z[idx + 1] - z[idx - 1]) / (2.0 * dx);
    }

    // Y-direction gradient
    if (j == 0) {
        dz_dy = (z[idx + nx] - z[idx]) / dy;
    } else if (j == ny - 1) {
        dz_dy = (z[idx] - z[idx - nx]) / dy;
    } else {
        dz_dy = (z[idx + nx] - z[idx - nx]) / (2.0 * dy);
    }

    double S_bed_hu = -g * h[idx] * dz_dx;
    double S_bed_hv = -g * h[idx] * dz_dy;

    // ===== Manning friction source term =====
    double S_friction_hu = 0.0;
    double S_friction_hv = 0.0;

    if (h[idx] >= dry_threshold) {
        double n = uniform_manning ? manning[0] : manning[idx];
        double u = hu[idx] / h[idx];
        double v = hv[idx] / h[idx];
        double vel_mag = sqrt(u * u + v * v);

        double h_13 = cbrt(h[idx]);
        double h_43 = h_13 * h[idx];
        double C_f = g * n * n / h_43;

        S_friction_hu = -C_f * vel_mag * hu[idx];
        S_friction_hv = -C_f * vel_mag * hv[idx];
    }

    // ===== Apply combined source terms =====
    hu[idx] += dt * (S_bed_hu + S_friction_hu);
    hv[idx] += dt * (S_bed_hv + S_friction_hv);

    // Positivity preserving
    if (h[idx] < 0.0) {
        h[idx] = 0.0;
        hu[idx] = 0.0;
        hv[idx] = 0.0;
    }
}

/**
 * @brief Compute CFL time step based on maximum wave speed
 *
 * CFL condition: dt ≤ CFL * min(dx, dy) / max(|u| + sqrt(gh), |v| + sqrt(gh))
 *
 * Uses parallel reduction to find maximum wave speed across all cells.
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param dx Cell spacing in x
 * @param dy Cell spacing in y
 * @param g Gravitational acceleration
 * @param CFL CFL number (typically 0.5-0.9)
 * @param nx, ny Grid dimensions
 * @return Computed time step
 */
__global__ void computeCFLTimeStep_kernel(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double dx, double dy,
    double g,
    int nx, int ny,
    double* __restrict__ max_wave_speed
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    __shared__ double shared_max[256];  // Assume blockDim.x * blockDim.y <= 256

    double local_max = 0.0;

    if (i < nx && j < ny) {
        int idx = i + j * nx;

        double c = sqrt(g * h[idx]);  // Shallow water wave speed
        double max_speed_x = fabs(u[idx]) + c;
        double max_speed_y = fabs(v[idx]) + c;

        local_max = fmax(max_speed_x / dx, max_speed_y / dy);
    }

    // Parallel reduction within block
    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    shared_max[tid] = local_max;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_max[tid] = fmax(shared_max[tid], shared_max[tid + s]);
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        atomicMax_double(max_wave_speed, shared_max[0]);
    }
}

/**
 * @brief Atomic max for double precision
 *
 * CUDA doesn't provide atomicMax for double, so we implement it
 * using atomicCAS (compare-and-swap).
 */
__device__ void atomicMax_double(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                       __double_as_longlong(fmax(val, __longlong_as_double(assumed))));
    } while (assumed != old);
}

} // namespace cuda
} // namespace hydrosis2d
