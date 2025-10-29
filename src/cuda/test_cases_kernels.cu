#include "cuda_kernels.cuh"
#include "test_cases.h"
#include <cmath>

namespace CudaKernels {

// ============================================================================
// Test Case Initialization Kernels
// ============================================================================

/**
 * @brief Initialize test case on host, then copy to device
 * This approach is simpler and avoids complex GPU code
 */
__global__ void initialize_from_host_kernel(
    CellData* cells,
    const CellData* host_data,
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        cells[idx] = host_data[idx];
    }
}

/**
 * @brief Generic initialization kernel (supports all test cases)
 */
__global__ void initialize_test_case_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin, real_t ymin,
    int test_case) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    real_t x = xmin + i * dx;
    real_t y = ymin + j * dy;
    real_t xc = xmin + 0.5 * nx * dx;
    real_t yc = ymin + 0.5 * ny * dy;

    CellData& cell = cells[idx];
    cell.n = 0.03; // Default Manning's n

    switch (test_case) {
        case 0: { // 1D Dam Break
            cell.z = 0.0;
            cell.n = 0.0;

            if (x < xc) {
                cell.U.h = 10.0;
            } else {
                cell.U.h = 1.0;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 1: { // 2D Circular Dam
            real_t r = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));
            real_t R_dam = 0.15 * nx * dx;

            cell.z = 0.0;
            cell.n = 0.0;

            if (r < R_dam) {
                cell.U.h = 10.0;
            } else {
                cell.U.h = 1.0;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 2: { // Partial Dam Break
            real_t gap_width = 0.2 * ny * dy;
            bool in_gap = (fabs(y - yc) < gap_width / 2.0);

            cell.z = 0.0;
            cell.n = 0.0;

            if (x < xc) {
                cell.U.h = 5.0;
            } else {
                cell.U.h = 0.5;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 3: { // Thacker's Planar Beach
            real_t a = 1.0;
            real_t h0 = 0.1;
            real_t x_rel = x - xc;

            cell.z = h0 * (x_rel * x_rel / (a * a) - 1.0);
            cell.n = 0.0;

            real_t eta = 0.0;
            cell.U.h = fmax(eta - cell.z, (real_t)0.0);
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 4: { // MacDonald Test
            // Sloping bed with hump
            if (x < 8.0) {
                cell.z = 0.0;
            } else if (x < 12.0) {
                cell.z = 0.2 - 0.05 * (x - 8.0);
            } else {
                cell.z = 0.0;
            }

            cell.n = 0.03;

            // Initial water
            if (x < 16.0) {
                real_t eta = 0.4;
                cell.U.h = fmax(eta - cell.z, (real_t)0.0);
            } else {
                cell.U.h = 0.0;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 5: { // Lake at Rest
            real_t eta = 1.0;

            // Sinusoidal bed topography
            cell.z = 0.2 * sin(2.0 * M_PI * x / (nx * dx)) *
                    cos(2.0 * M_PI * y / (ny * dy));
            cell.n = 0.03;

            cell.U.h = eta - cell.z;
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 6: { // Small Perturbation
            real_t h0 = 1.0;
            real_t A = 0.01;
            real_t r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
            real_t sigma = 0.1 * nx * dx;

            cell.z = 0.0;
            cell.n = 0.0;

            cell.U.h = h0 + A * exp(-r2 / (2.0 * sigma * sigma));
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 7: { // Flow Over Bump
            real_t q = 0.5;
            real_t h_in = 1.0;
            real_t L = 0.1 * nx * dx;

            cell.z = 0.2 * exp(-((x - xc) * (x - xc)) / (L * L));
            cell.n = 0.0;

            cell.U.h = h_in;
            cell.U.qx = q * h_in;
            cell.U.qy = 0.0;
            break;
        }

        case 8: { // Oblique Hydraulic Jump
            cell.z = 0.0;
            cell.n = 0.0;

            if (x - xc > y - yc) {
                cell.U.h = 0.5;
                cell.U.qx = 1.0;
                cell.U.qy = 1.0;
            } else {
                cell.U.h = 1.0;
                cell.U.qx = 0.5;
                cell.U.qy = 0.5;
            }
            break;
        }

        default:
            // Default: simple dam break
            cell.z = 0.0;
            cell.n = 0.0;
            cell.U.h = (x < xc) ? 2.0 : 1.0;
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
    }

    cell.is_wet = (cell.U.h > Constants::DRY_TOLERANCE);
}

} // namespace CudaKernels
