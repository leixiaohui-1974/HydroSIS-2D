/**
 * @file muscl_kernels.cu
 * @brief CUDA kernels for MUSCL reconstruction (2nd order spatial accuracy)
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "../ShallowWaterSolver.cuh"
#include <cmath>

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Slope limiter types for MUSCL reconstruction
 */
enum class LimiterType {
    NONE = 0,      // No limiting (2nd order, may oscillate)
    MINMOD = 1,    // Most diffusive, most stable
    VANLEER = 2,   // Smooth, good balance
    SUPERBEE = 3,  // Least diffusive, sharpest
    MC = 4         // Monotonized central (recommended)
};

/**
 * @brief Minmod slope limiter
 *
 * φ(r) = max(0, min(1, r))
 *
 * Most diffusive but very stable. Good for problems with shocks.
 */
__device__ inline double minmod_limiter(double r) {
    return fmax(0.0, fmin(1.0, r));
}

/**
 * @brief Van Leer slope limiter
 *
 * φ(r) = (r + |r|) / (1 + |r|)
 *
 * Smooth and differentiable. Good general-purpose limiter.
 */
__device__ inline double vanleer_limiter(double r) {
    if (r <= 0.0) return 0.0;
    return (r + fabs(r)) / (1.0 + fabs(r));
}

/**
 * @brief Superbee slope limiter
 *
 * φ(r) = max(0, min(2r, 1), min(r, 2))
 *
 * Least diffusive, sharpest gradients. Can be aggressive.
 */
__device__ inline double superbee_limiter(double r) {
    if (r <= 0.0) return 0.0;
    return fmax(fmin(2.0 * r, 1.0), fmin(r, 2.0));
}

/**
 * @brief Monotonized Central (MC) slope limiter
 *
 * φ(r) = max(0, min((1+r)/2, 2, 2r))
 *
 * Recommended limiter: less diffusive than minmod, more stable than superbee.
 */
__device__ inline double mc_limiter(double r) {
    if (r <= 0.0) return 0.0;
    return fmax(0.0, fmin(fmin(0.5 * (1.0 + r), 2.0), 2.0 * r));
}

/**
 * @brief Apply slope limiter based on type
 *
 * @param r Ratio of consecutive gradients
 * @param limiter_type Limiter type
 * @return Limited slope coefficient φ(r)
 */
__device__ inline double apply_limiter(double r, int limiter_type) {
    switch (limiter_type) {
        case static_cast<int>(LimiterType::MINMOD):
            return minmod_limiter(r);
        case static_cast<int>(LimiterType::VANLEER):
            return vanleer_limiter(r);
        case static_cast<int>(LimiterType::SUPERBEE):
            return superbee_limiter(r);
        case static_cast<int>(LimiterType::MC):
            return mc_limiter(r);
        case static_cast<int>(LimiterType::NONE):
            return 1.0;  // No limiting
        default:
            return mc_limiter(r);  // Default to MC
    }
}

/**
 * @brief Compute MUSCL reconstruction in X direction
 *
 * For each cell i, computes left and right states at the cell interface:
 *   Q_L(i+1/2) = Q[i] + 0.5 * φ(r) * ΔQ[i]
 *   Q_R(i+1/2) = Q[i+1] - 0.5 * φ(r) * ΔQ[i+1]
 *
 * where:
 *   ΔQ[i] = Q[i] - Q[i-1]  (backward difference)
 *   r = ΔQ[i+1] / ΔQ[i]    (smoothness indicator)
 *
 * Output arrays store reconstructed states at interfaces:
 *   Q_left[i + j*(nx+1)] = left state at interface (i, j)
 *   Q_right[i + j*(nx+1)] = right state at interface (i, j)
 *
 * @param Q Field to reconstruct (h, u, or v) [nx*ny]
 * @param Q_left Left state at interfaces [(nx+1)*ny]
 * @param Q_right Right state at interfaces [(nx+1)*ny]
 * @param nx, ny Grid dimensions
 * @param limiter_type Slope limiter type
 * @param dry_threshold Dry cell threshold (reconstruct as zero if h < threshold)
 */
__global__ void musclReconstructX(
    const double* __restrict__ Q,
    double* __restrict__ Q_left,
    double* __restrict__ Q_right,
    int nx, int ny,
    int limiter_type,
    double dry_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx + 1 || j >= ny) return;

    int face_idx = i + j * (nx + 1);

    // Boundary faces: use first-order (no reconstruction)
    if (i == 0) {
        // Left boundary face
        int cell_R = 0 + j * nx;
        Q_left[face_idx] = Q[cell_R];
        Q_right[face_idx] = Q[cell_R];
        return;
    }

    if (i == nx) {
        // Right boundary face
        int cell_L = (nx - 1) + j * nx;
        Q_left[face_idx] = Q[cell_L];
        Q_right[face_idx] = Q[cell_L];
        return;
    }

    // Interior face: MUSCL reconstruction
    int cell_L = (i - 1) + j * nx;  // Cell to the left of face
    int cell_R = i + j * nx;        // Cell to the right of face

    // Check for dry cells - use first order if either cell is dry
    if (Q[cell_L] < dry_threshold || Q[cell_R] < dry_threshold) {
        Q_left[face_idx] = Q[cell_L];
        Q_right[face_idx] = Q[cell_R];
        return;
    }

    // Compute gradients
    // Left cell gradient
    double dQ_L;
    if (i > 1) {
        int cell_LL = (i - 2) + j * nx;
        dQ_L = Q[cell_L] - Q[cell_LL];
    } else {
        dQ_L = Q[cell_R] - Q[cell_L];  // Forward difference at boundary
    }

    // Right cell gradient
    double dQ_R;
    if (i < nx - 1) {
        int cell_RR = (i + 1) + j * nx;
        dQ_R = Q[cell_RR] - Q[cell_R];
    } else {
        dQ_R = Q[cell_R] - Q[cell_L];  // Backward difference at boundary
    }

    // Central gradient for this face
    double dQ_central = Q[cell_R] - Q[cell_L];

    // Apply limiters
    double phi_L = 1.0;
    double phi_R = 1.0;

    if (fabs(dQ_L) > 1e-12) {
        double r_L = dQ_central / dQ_L;
        phi_L = apply_limiter(r_L, limiter_type);
    }

    if (fabs(dQ_R) > 1e-12) {
        double r_R = dQ_L / dQ_central;  // Note: reversed ratio for right state
        phi_R = apply_limiter(r_R, limiter_type);
    }

    // Reconstruct left and right states at face
    Q_left[face_idx] = Q[cell_L] + 0.5 * phi_L * dQ_L;
    Q_right[face_idx] = Q[cell_R] - 0.5 * phi_R * dQ_R;
}

/**
 * @brief Compute MUSCL reconstruction in Y direction
 *
 * Similar to musclReconstructX but for Y-direction faces.
 *
 * Output arrays:
 *   Q_bottom[i + j*nx] = bottom state at interface (i, j)
 *   Q_top[i + j*nx] = top state at interface (i, j)
 *
 * @param Q Field to reconstruct [nx*ny]
 * @param Q_bottom Bottom state at interfaces [nx*(ny+1)]
 * @param Q_top Top state at interfaces [nx*(ny+1)]
 * @param nx, ny Grid dimensions
 * @param limiter_type Slope limiter type
 * @param dry_threshold Dry cell threshold
 */
__global__ void musclReconstructY(
    const double* __restrict__ Q,
    double* __restrict__ Q_bottom,
    double* __restrict__ Q_top,
    int nx, int ny,
    int limiter_type,
    double dry_threshold
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny + 1) return;

    int face_idx = i + j * nx;

    // Boundary faces: use first-order
    if (j == 0) {
        // Bottom boundary face
        int cell_T = i + 0 * nx;
        Q_bottom[face_idx] = Q[cell_T];
        Q_top[face_idx] = Q[cell_T];
        return;
    }

    if (j == ny) {
        // Top boundary face
        int cell_B = i + (ny - 1) * nx;
        Q_bottom[face_idx] = Q[cell_B];
        Q_top[face_idx] = Q[cell_B];
        return;
    }

    // Interior face: MUSCL reconstruction
    int cell_B = i + (j - 1) * nx;  // Cell below face
    int cell_T = i + j * nx;        // Cell above face

    // Check for dry cells
    if (Q[cell_B] < dry_threshold || Q[cell_T] < dry_threshold) {
        Q_bottom[face_idx] = Q[cell_B];
        Q_top[face_idx] = Q[cell_T];
        return;
    }

    // Compute gradients
    // Bottom cell gradient
    double dQ_B;
    if (j > 1) {
        int cell_BB = i + (j - 2) * nx;
        dQ_B = Q[cell_B] - Q[cell_BB];
    } else {
        dQ_B = Q[cell_T] - Q[cell_B];
    }

    // Top cell gradient
    double dQ_T;
    if (j < ny - 1) {
        int cell_TT = i + (j + 1) * nx;
        dQ_T = Q[cell_TT] - Q[cell_T];
    } else {
        dQ_T = Q[cell_T] - Q[cell_B];
    }

    // Central gradient
    double dQ_central = Q[cell_T] - Q[cell_B];

    // Apply limiters
    double phi_B = 1.0;
    double phi_T = 1.0;

    if (fabs(dQ_B) > 1e-12) {
        double r_B = dQ_central / dQ_B;
        phi_B = apply_limiter(r_B, limiter_type);
    }

    if (fabs(dQ_T) > 1e-12) {
        double r_T = dQ_B / dQ_central;
        phi_T = apply_limiter(r_T, limiter_type);
    }

    // Reconstruct bottom and top states at face
    Q_bottom[face_idx] = Q[cell_B] + 0.5 * phi_B * dQ_B;
    Q_top[face_idx] = Q[cell_T] - 0.5 * phi_T * dQ_T;
}

/**
 * @brief Compute MUSCL-Hancock predictor step
 *
 * MUSCL-Hancock method (2nd order in space and time):
 * 1. Reconstruct Q at cell interfaces
 * 2. Evolve for half time step: Q* = Q + 0.5*dt*L(Q)
 * 3. Use Q* for flux computation
 *
 * This kernel computes step 2: half-step evolution.
 *
 * @param Q_left Left reconstructed state
 * @param Q_right Right reconstructed state
 * @param dQdt Time derivative (from flux divergence)
 * @param Q_evolved Evolved state output
 * @param dt Time step
 * @param n Number of elements
 */
__global__ void musclHancockPredictor(
    const double* __restrict__ Q_left,
    const double* __restrict__ Q_right,
    const double* __restrict__ dQdt,
    double* __restrict__ Q_evolved,
    double dt,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    // Half-step evolution
    Q_evolved[idx] = 0.5 * (Q_left[idx] + Q_right[idx]) + 0.5 * dt * dQdt[idx];

    // Positivity preserving for water depth
    if (Q_evolved[idx] < 0.0) {
        Q_evolved[idx] = 0.0;
    }
}

/**
 * @brief Compute 2nd-order fluxes using MUSCL reconstruction
 *
 * Combines MUSCL reconstruction with Riemann solver for 2nd-order fluxes.
 * Uses reconstructed left and right states at each interface.
 *
 * This is the main kernel for 2nd-order flux computation.
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param flux_x X-fluxes output [3*(nx+1)*ny]
 * @param flux_y Y-fluxes output [3*nx*(ny+1)]
 * @param nx, ny Grid dimensions
 * @param dx, dy Cell spacing
 * @param g Gravitational acceleration
 * @param riemann_type Riemann solver type (0=HLL, 1=HLLC)
 * @param limiter_type Slope limiter type
 * @param dry_threshold Dry cell threshold
 */
__global__ void computeFluxesMUSCL(
    const double* __restrict__ h,
    const double* __restrict__ u,
    const double* __restrict__ v,
    double* __restrict__ flux_x,
    double* __restrict__ flux_y,
    int nx, int ny,
    double dx, double dy,
    double g,
    int riemann_type,
    int limiter_type,
    double dry_threshold
) {
    // This is a simplified version - full implementation would:
    // 1. Call musclReconstructX/Y for h, u, v
    // 2. Apply Riemann solver to reconstructed states
    // 3. Store fluxes
    //
    // For now, this serves as a placeholder showing the structure.
    // The full implementation requires shared memory and multiple kernel calls.

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    // Implementation note:
    // In practice, MUSCL reconstruction is done in separate kernel calls:
    // 1. musclReconstructX(h, ...), musclReconstructX(u, ...), musclReconstructX(v, ...)
    // 2. Apply Riemann solver with reconstructed states
    // 3. Store fluxes
    //
    // This allows better memory access patterns and kernel specialization.
}

/**
 * @brief Check and enforce monotonicity after reconstruction
 *
 * Ensures reconstructed values don't create new extrema:
 *   Q_min = min(Q_L, Q_R)
 *   Q_max = max(Q_L, Q_R)
 *   Q_reconstructed = clamp(Q_reconstructed, Q_min, Q_max)
 *
 * @param Q_left Left reconstructed state
 * @param Q_right Right reconstructed state
 * @param Q_cell_left Cell value on left
 * @param Q_cell_right Cell value on right
 * @param n Number of faces
 */
__global__ void enforceMonotonicity(
    double* __restrict__ Q_left,
    double* __restrict__ Q_right,
    const double* __restrict__ Q_cell_left,
    const double* __restrict__ Q_cell_right,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= n) return;

    double Q_min = fmin(Q_cell_left[idx], Q_cell_right[idx]);
    double Q_max = fmax(Q_cell_left[idx], Q_cell_right[idx]);

    // Clamp reconstructed values to avoid new extrema
    Q_left[idx] = fmax(Q_min, fmin(Q_max, Q_left[idx]));
    Q_right[idx] = fmax(Q_min, fmin(Q_max, Q_right[idx]));
}

} // namespace cuda
} // namespace hydrosis2d
