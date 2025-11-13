/**
 * @file bc_kernels.cu
 * @brief CUDA kernels for boundary condition application
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "../ShallowWaterSolver.cuh"

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Boundary condition types
 */
enum class BCType {
    WALL = 0,        // Reflective wall (zero normal velocity)
    INFLOW = 1,      // Prescribed inflow (h, u, v specified)
    OUTFLOW = 2,     // Zero-gradient outflow
    PERIODIC = 3,    // Periodic boundary
    CRITICAL = 4     // Critical flow (transmissive)
};

/**
 * @brief Apply wall boundary conditions (reflective)
 *
 * For wall boundaries:
 *   - Normal velocity component is reflected (negated)
 *   - Tangential velocity component is preserved
 *   - Water depth is copied from interior
 *
 * Left/Right walls: reflect u, preserve v
 * Top/Bottom walls: reflect v, preserve u
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param nx, ny Grid dimensions
 * @param boundary Which boundary (0=left, 1=right, 2=bottom, 3=top)
 */
__global__ void applyWallBC(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    int nx, int ny,
    int boundary
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (boundary == 0) {
        // Left wall (i = 0)
        if (j >= ny) return;
        int idx_ghost = 0 + j * nx;
        int idx_interior = 1 + j * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = -u[idx_interior];  // Reflect normal component
        v[idx_ghost] = v[idx_interior];   // Preserve tangential

    } else if (boundary == 1) {
        // Right wall (i = nx-1)
        if (j >= ny) return;
        int idx_ghost = (nx - 1) + j * nx;
        int idx_interior = (nx - 2) + j * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = -u[idx_interior];
        v[idx_ghost] = v[idx_interior];

    } else if (boundary == 2) {
        // Bottom wall (j = 0)
        if (i >= nx) return;
        int idx_ghost = i + 0 * nx;
        int idx_interior = i + 1 * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = -v[idx_interior];  // Reflect normal component

    } else if (boundary == 3) {
        // Top wall (j = ny-1)
        if (i >= nx) return;
        int idx_ghost = i + (ny - 1) * nx;
        int idx_interior = i + (ny - 2) * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = -v[idx_interior];
    }
}

/**
 * @brief Apply inflow boundary conditions
 *
 * Prescribes water depth and velocity at boundary cells.
 * Values are provided in bc_values array.
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param bc_values Boundary values [3*N]: h, u, v for each boundary cell
 * @param nx, ny Grid dimensions
 * @param boundary Which boundary (0=left, 1=right, 2=bottom, 3=top)
 */
__global__ void applyInflowBC(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    const double* __restrict__ bc_values,
    int nx, int ny,
    int boundary
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (boundary == 0) {
        // Left boundary
        if (j >= ny) return;
        int idx = 0 + j * nx;
        h[idx] = bc_values[3 * j + 0];
        u[idx] = bc_values[3 * j + 1];
        v[idx] = bc_values[3 * j + 2];

    } else if (boundary == 1) {
        // Right boundary
        if (j >= ny) return;
        int idx = (nx - 1) + j * nx;
        h[idx] = bc_values[3 * j + 0];
        u[idx] = bc_values[3 * j + 1];
        v[idx] = bc_values[3 * j + 2];

    } else if (boundary == 2) {
        // Bottom boundary
        if (i >= nx) return;
        int idx = i + 0 * nx;
        h[idx] = bc_values[3 * i + 0];
        u[idx] = bc_values[3 * i + 1];
        v[idx] = bc_values[3 * i + 2];

    } else if (boundary == 3) {
        // Top boundary
        if (i >= nx) return;
        int idx = i + (ny - 1) * nx;
        h[idx] = bc_values[3 * i + 0];
        u[idx] = bc_values[3 * i + 1];
        v[idx] = bc_values[3 * i + 2];
    }
}

/**
 * @brief Apply outflow boundary conditions (zero gradient)
 *
 * Extrapolates values from interior to boundary:
 *   Q_boundary = Q_interior
 *
 * This is a simple transmissive boundary condition.
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param nx, ny Grid dimensions
 * @param boundary Which boundary (0=left, 1=right, 2=bottom, 3=top)
 */
__global__ void applyOutflowBC(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    int nx, int ny,
    int boundary
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (boundary == 0) {
        // Left boundary
        if (j >= ny) return;
        int idx_ghost = 0 + j * nx;
        int idx_interior = 1 + j * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = v[idx_interior];

    } else if (boundary == 1) {
        // Right boundary
        if (j >= ny) return;
        int idx_ghost = (nx - 1) + j * nx;
        int idx_interior = (nx - 2) + j * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = v[idx_interior];

    } else if (boundary == 2) {
        // Bottom boundary
        if (i >= nx) return;
        int idx_ghost = i + 0 * nx;
        int idx_interior = i + 1 * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = v[idx_interior];

    } else if (boundary == 3) {
        // Top boundary
        if (i >= nx) return;
        int idx_ghost = i + (ny - 1) * nx;
        int idx_interior = i + (ny - 2) * nx;

        h[idx_ghost] = h[idx_interior];
        u[idx_ghost] = u[idx_interior];
        v[idx_ghost] = v[idx_interior];
    }
}

/**
 * @brief Apply periodic boundary conditions
 *
 * Copies values from opposite boundary:
 *   Left boundary = Right interior
 *   Right boundary = Left interior
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param nx, ny Grid dimensions
 * @param boundary Which boundary pair (0=left-right, 1=bottom-top)
 */
__global__ void applyPeriodicBC(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    int nx, int ny,
    int boundary
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (boundary == 0) {
        // X-direction periodicity
        if (j >= ny) return;

        int idx_left = 0 + j * nx;
        int idx_right = (nx - 1) + j * nx;
        int idx_left_interior = 1 + j * nx;
        int idx_right_interior = (nx - 2) + j * nx;

        // Left ghost = right interior
        h[idx_left] = h[idx_right_interior];
        u[idx_left] = u[idx_right_interior];
        v[idx_left] = v[idx_right_interior];

        // Right ghost = left interior
        h[idx_right] = h[idx_left_interior];
        u[idx_right] = u[idx_left_interior];
        v[idx_right] = v[idx_left_interior];

    } else if (boundary == 1) {
        // Y-direction periodicity
        if (i >= nx) return;

        int idx_bottom = i + 0 * nx;
        int idx_top = i + (ny - 1) * nx;
        int idx_bottom_interior = i + 1 * nx;
        int idx_top_interior = i + (ny - 2) * nx;

        // Bottom ghost = top interior
        h[idx_bottom] = h[idx_top_interior];
        u[idx_bottom] = u[idx_top_interior];
        v[idx_bottom] = v[idx_top_interior];

        // Top ghost = bottom interior
        h[idx_top] = h[idx_bottom_interior];
        u[idx_top] = u[idx_bottom_interior];
        v[idx_top] = v[idx_bottom_interior];
    }
}

/**
 * @brief Apply critical flow boundary condition (transmissive)
 *
 * For supercritical outflow:
 *   - Extrapolate all variables from interior
 *
 * For subcritical outflow:
 *   - Extrapolate u, v from interior
 *   - Set h based on critical flow condition: h = (u²/g)^(1/3)
 *
 * Froude number: Fr = |u| / sqrt(g*h)
 *   Fr > 1: supercritical
 *   Fr < 1: subcritical
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param g Gravitational acceleration
 * @param nx, ny Grid dimensions
 * @param boundary Which boundary (0=left, 1=right, 2=bottom, 3=top)
 */
__global__ void applyCriticalBC(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    double g,
    int nx, int ny,
    int boundary
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    const double dry_tol = 1e-10;

    if (boundary == 0 || boundary == 1) {
        // Left or right boundary
        if (j >= ny) return;

        int idx_ghost = (boundary == 0) ? (0 + j * nx) : ((nx - 1) + j * nx);
        int idx_interior = (boundary == 0) ? (1 + j * nx) : ((nx - 2) + j * nx);

        double h_int = h[idx_interior];
        double u_int = u[idx_interior];
        double v_int = v[idx_interior];

        if (h_int < dry_tol) {
            h[idx_ghost] = 0.0;
            u[idx_ghost] = 0.0;
            v[idx_ghost] = 0.0;
            return;
        }

        // Froude number
        double Fr = fabs(u_int) / sqrt(g * h_int);

        if (Fr > 1.0) {
            // Supercritical: extrapolate all
            h[idx_ghost] = h_int;
            u[idx_ghost] = u_int;
            v[idx_ghost] = v_int;
        } else {
            // Subcritical: critical depth
            double u_mag = sqrt(u_int * u_int + v_int * v_int);
            double h_crit = cbrt(u_mag * u_mag / g);

            h[idx_ghost] = h_crit;
            u[idx_ghost] = u_int;
            v[idx_ghost] = v_int;
        }

    } else {
        // Bottom or top boundary
        if (i >= nx) return;

        int idx_ghost = (boundary == 2) ? (i + 0 * nx) : (i + (ny - 1) * nx);
        int idx_interior = (boundary == 2) ? (i + 1 * nx) : (i + (ny - 2) * nx);

        double h_int = h[idx_interior];
        double u_int = u[idx_interior];
        double v_int = v[idx_interior];

        if (h_int < dry_tol) {
            h[idx_ghost] = 0.0;
            u[idx_ghost] = 0.0;
            v[idx_ghost] = 0.0;
            return;
        }

        // Froude number (based on v for vertical boundaries)
        double Fr = fabs(v_int) / sqrt(g * h_int);

        if (Fr > 1.0) {
            // Supercritical: extrapolate all
            h[idx_ghost] = h_int;
            u[idx_ghost] = u_int;
            v[idx_ghost] = v_int;
        } else {
            // Subcritical: critical depth
            double u_mag = sqrt(u_int * u_int + v_int * v_int);
            double h_crit = cbrt(u_mag * u_mag / g);

            h[idx_ghost] = h_crit;
            u[idx_ghost] = u_int;
            v[idx_ghost] = v_int;
        }
    }
}

/**
 * @brief Apply all boundary conditions
 *
 * Unified kernel that applies boundary conditions to all four boundaries
 * based on bc_types array.
 *
 * @param h Water depth [nx*ny]
 * @param u X-velocity [nx*ny]
 * @param v Y-velocity [nx*ny]
 * @param bc_types Boundary type for each boundary [4]: left, right, bottom, top
 * @param bc_values Boundary values (for inflow BC)
 * @param g Gravitational acceleration
 * @param nx, ny Grid dimensions
 */
__global__ void applyAllBoundaryConditions(
    double* __restrict__ h,
    double* __restrict__ u,
    double* __restrict__ v,
    const int* __restrict__ bc_types,
    const double* __restrict__ bc_values,
    double g,
    int nx, int ny
) {
    // This kernel is called with 2D grid covering entire domain
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    // Only process boundary cells
    bool is_left = (i == 0);
    bool is_right = (i == nx - 1);
    bool is_bottom = (j == 0);
    bool is_top = (j == ny - 1);

    if (!is_left && !is_right && !is_bottom && !is_top) {
        return;  // Interior cell, skip
    }

    int idx = i + j * nx;

    // Apply BC for each boundary
    if (is_left) {
        int bc_type = bc_types[0];
        int idx_interior = 1 + j * nx;

        switch (bc_type) {
            case static_cast<int>(BCType::WALL):
                h[idx] = h[idx_interior];
                u[idx] = -u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            case static_cast<int>(BCType::OUTFLOW):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            case static_cast<int>(BCType::INFLOW):
                h[idx] = bc_values[3 * j + 0];
                u[idx] = bc_values[3 * j + 1];
                v[idx] = bc_values[3 * j + 2];
                break;

            // PERIODIC and CRITICAL require separate handling
        }
    }

    if (is_right) {
        int bc_type = bc_types[1];
        int idx_interior = (nx - 2) + j * nx;

        switch (bc_type) {
            case static_cast<int>(BCType::WALL):
                h[idx] = h[idx_interior];
                u[idx] = -u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            case static_cast<int>(BCType::OUTFLOW):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            // Other types handled separately
        }
    }

    if (is_bottom) {
        int bc_type = bc_types[2];
        int idx_interior = i + 1 * nx;

        switch (bc_type) {
            case static_cast<int>(BCType::WALL):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = -v[idx_interior];
                break;

            case static_cast<int>(BCType::OUTFLOW):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            // Other types handled separately
        }
    }

    if (is_top) {
        int bc_type = bc_types[3];
        int idx_interior = i + (ny - 2) * nx;

        switch (bc_type) {
            case static_cast<int>(BCType::WALL):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = -v[idx_interior];
                break;

            case static_cast<int>(BCType::OUTFLOW):
                h[idx] = h[idx_interior];
                u[idx] = u[idx_interior];
                v[idx] = v[idx_interior];
                break;

            // Other types handled separately
        }
    }
}

/**
 * @brief Convert conservative variables to primitive at boundaries
 *
 * Helper kernel to compute h, u, v from h, hu, hv at boundary cells only.
 * Useful after flux updates before applying boundary conditions.
 *
 * @param h Water depth [nx*ny]
 * @param hu X-momentum [nx*ny]
 * @param hv Y-momentum [nx*ny]
 * @param u X-velocity output [nx*ny]
 * @param v Y-velocity output [nx*ny]
 * @param nx, ny Grid dimensions
 * @param dry_threshold Dry/wet threshold
 */
__global__ void boundaryConservativeToPrimitive(
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

    // Only process boundary cells
    if (i != 0 && i != nx - 1 && j != 0 && j != ny - 1) {
        return;
    }

    int idx = i + j * nx;

    if (h[idx] > dry_threshold) {
        u[idx] = hu[idx] / h[idx];
        v[idx] = hv[idx] / h[idx];
    } else {
        u[idx] = 0.0;
        v[idx] = 0.0;
    }
}

} // namespace cuda
} // namespace hydrosis2d
