#ifndef TEST_CASES_H
#define TEST_CASES_H

#include "hydrosis_types.h"
#include <cmath>

/**
 * @brief Standard test cases for validation
 */
namespace TestCases {

/**
 * @brief 1D Dam Break (Ritter's solution)
 *
 * Analytical solution available for validation
 * Tests shock capturing and rarefaction wave
 */
struct DamBreak1D {
    static constexpr int id = 0;
    static constexpr const char* name = "1D Dam Break (Ritter)";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t x_dam = xmin + 0.5 * nx * dx;

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;

                cells[idx].z = 0.0;  // Flat bed
                cells[idx].n = 0.0;  // No friction

                // Left: high water, Right: low water
                if (x < x_dam) {
                    cells[idx].U.h = 10.0;
                } else {
                    cells[idx].U.h = 1.0;
                }

                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = (cells[idx].U.h > Constants::DRY_TOLERANCE);
            }
        }
    }

    // Analytical solution at time t
    static real_t analytical_solution(real_t x, real_t t, real_t x0 = 0.5,
                                     real_t h_L = 10.0, real_t h_R = 1.0) {
        real_t g = Constants::GRAVITY;
        real_t c_L = sqrt(g * h_L);

        // Position relative to dam
        real_t xi = (x - x0) / t;

        // Rarefaction wave region
        if (xi <= -c_L) {
            return h_L;  // Undisturbed left state
        } else if (xi >= 2.0 * c_L) {
            return h_R;  // Undisturbed right state
        } else {
            // Rarefaction fan
            real_t h = (4.0 / (9.0 * g)) * pow(c_L - 0.5 * xi, 2.0);
            return h;
        }
    }
};

/**
 * @brief 2D Circular Dam Break
 *
 * Tests radial symmetry and 2D propagation
 * No analytical solution but should maintain symmetry
 */
struct CircularDam {
    static constexpr int id = 1;
    static constexpr const char* name = "2D Circular Dam Break";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t xc = xmin + 0.5 * nx * dx;
        real_t yc = ymin + 0.5 * ny * dy;
        real_t R_dam = 0.15 * nx * dx;  // Dam radius

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;
                real_t y = ymin + j * dy;
                real_t r = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));

                cells[idx].z = 0.0;
                cells[idx].n = 0.0;

                // Inside dam: high water, Outside: low water
                if (r < R_dam) {
                    cells[idx].U.h = 10.0;
                } else {
                    cells[idx].U.h = 1.0;
                }

                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Partial Dam Break
 *
 * Tests 2D flow patterns with asymmetric initial conditions
 */
struct PartialDamBreak {
    static constexpr int id = 2;
    static constexpr const char* name = "Partial Dam Break";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t x_dam = xmin + 0.5 * nx * dx;
        real_t y_mid = ymin + 0.5 * ny * dy;
        real_t gap_width = 0.2 * ny * dy;  // Gap in dam

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;
                real_t y = ymin + j * dy;

                cells[idx].z = 0.0;
                cells[idx].n = 0.0;

                // Left side: high water
                // Right side: low water except in gap region
                bool in_gap = (fabs(y - y_mid) < gap_width / 2.0);

                if (x < x_dam) {
                    cells[idx].U.h = 5.0;
                } else if (in_gap) {
                    cells[idx].U.h = 0.5;  // Lower level in gap
                } else {
                    cells[idx].U.h = 0.5;
                }

                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Thacker's Planar Beach Test
 *
 * Analytical solution for wetting and drying
 * Oscillating flow on a planar beach
 */
struct ThackerPlanarBeach {
    static constexpr int id = 3;
    static constexpr const char* name = "Thacker's Planar Beach";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t a = 1.0;   // Basin width parameter
        real_t h0 = 0.1;  // Depth parameter
        real_t omega = sqrt(2.0 * Constants::GRAVITY * h0) / a;

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + (i - nx/2) * dx;

                // Parabolic bed: z = h0(x²/a² - 1)
                cells[idx].z = h0 * (x * x / (a * a) - 1.0);
                cells[idx].n = 0.0;

                // Initial water surface elevation
                real_t eta = 0.0;
                real_t h = std::max(eta - cells[idx].z, (real_t)0.0);

                cells[idx].U.h = h;
                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = (h > Constants::DRY_TOLERANCE);
            }
        }
    }
};

/**
 * @brief MacDonald Test Case
 *
 * Standard benchmark for wetting/drying with friction
 * Includes bed elevation and Manning's n
 */
struct MacDonald {
    static constexpr int id = 4;
    static constexpr const char* name = "MacDonald Wetting/Drying";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;

                // Sloping bed with hump
                if (x < 8.0) {
                    cells[idx].z = 0.0;
                } else if (x < 12.0) {
                    cells[idx].z = 0.2 - 0.05 * (x - 8.0);
                } else {
                    cells[idx].z = 0.0;
                }

                cells[idx].n = 0.03;  // Manning's n

                // Initial condition: water on left
                if (x < 16.0) {
                    real_t eta = 0.4;  // Surface elevation
                    cells[idx].U.h = std::max(eta - cells[idx].z, (real_t)0.0);
                } else {
                    cells[idx].U.h = 0.0;
                }

                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = (cells[idx].U.h > Constants::DRY_TOLERANCE);
            }
        }
    }
};

/**
 * @brief Lake at Rest Test
 *
 * Tests well-balanced property and source term treatment
 * Water surface should remain exactly flat
 */
struct LakeAtRest {
    static constexpr int id = 5;
    static constexpr const char* name = "Lake at Rest";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t eta = 1.0;  // Constant water surface elevation

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;
                real_t y = ymin + j * dy;

                // Random bed topography
                cells[idx].z = 0.2 * sin(2.0 * M_PI * x / (nx * dx)) *
                              cos(2.0 * M_PI * y / (ny * dy));
                cells[idx].n = 0.03;

                // Depth such that eta = h + z
                cells[idx].U.h = eta - cells[idx].z;
                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Small Perturbation Test
 *
 * Tests stability and accuracy for small amplitude waves
 */
struct SmallPerturbation {
    static constexpr int id = 6;
    static constexpr const char* name = "Small Perturbation";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t h0 = 1.0;  // Background depth
        real_t A = 0.01;  // Perturbation amplitude

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;
                real_t y = ymin + j * dy;

                cells[idx].z = 0.0;
                cells[idx].n = 0.0;

                // Small Gaussian perturbation at center
                real_t xc = xmin + 0.5 * nx * dx;
                real_t yc = ymin + 0.5 * ny * dy;
                real_t r2 = (x - xc) * (x - xc) + (y - yc) * (y - yc);
                real_t sigma = 0.1 * nx * dx;

                cells[idx].U.h = h0 + A * exp(-r2 / (2.0 * sigma * sigma));
                cells[idx].U.qx = 0.0;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Steady Flow Over Bump
 *
 * Tests subcritical/supercritical flow transitions
 */
struct FlowOverBump {
    static constexpr int id = 7;
    static constexpr const char* name = "Flow Over Bump";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        real_t q = 0.5;   // Unit discharge (steady flow)
        real_t h_in = 1.0; // Inlet depth

        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;

                // Gaussian bump
                real_t xc = xmin + 0.5 * nx * dx;
                real_t L = 0.1 * nx * dx;
                cells[idx].z = 0.2 * exp(-((x - xc) * (x - xc)) / (L * L));
                cells[idx].n = 0.0;

                // Initialize with steady flow
                cells[idx].U.h = h_in;
                cells[idx].U.qx = q * h_in;
                cells[idx].U.qy = 0.0;
                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Oblique Hydraulic Jump
 *
 * Tests shock capturing in 2D
 */
struct ObliqueJump {
    static constexpr int id = 8;
    static constexpr const char* name = "Oblique Hydraulic Jump";

    static void initialize(CellData* cells, int nx, int ny,
                          real_t dx, real_t dy, real_t xmin, real_t ymin) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                int idx = j * nx + i;
                real_t x = xmin + i * dx;
                real_t y = ymin + j * dy;

                cells[idx].z = 0.0;
                cells[idx].n = 0.0;

                // Diagonal discontinuity
                real_t xc = xmin + 0.5 * nx * dx;
                real_t yc = ymin + 0.5 * ny * dy;

                if (x - xc > y - yc) {
                    // Supercritical flow
                    cells[idx].U.h = 0.5;
                    cells[idx].U.qx = 1.0;
                    cells[idx].U.qy = 1.0;
                } else {
                    // Subcritical flow
                    cells[idx].U.h = 1.0;
                    cells[idx].U.qx = 0.5;
                    cells[idx].U.qy = 0.5;
                }

                cells[idx].is_wet = true;
            }
        }
    }
};

/**
 * @brief Get test case by ID
 */
inline const char* get_test_name(int test_id) {
    switch(test_id) {
        case 0: return DamBreak1D::name;
        case 1: return CircularDam::name;
        case 2: return PartialDamBreak::name;
        case 3: return ThackerPlanarBeach::name;
        case 4: return MacDonald::name;
        case 5: return LakeAtRest::name;
        case 6: return SmallPerturbation::name;
        case 7: return FlowOverBump::name;
        case 8: return ObliqueJump::name;
        default: return "Unknown Test";
    }
}

} // namespace TestCases

#endif // TEST_CASES_H
