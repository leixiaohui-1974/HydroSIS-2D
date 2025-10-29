#include "validation.h"
#include "test_cases.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

real_t Validation::compute_L1_error(
    const real_t* numerical,
    const real_t* analytical,
    int n) {

    real_t sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += fabs(numerical[i] - analytical[i]);
    }
    return sum / n;
}

real_t Validation::compute_L2_error(
    const real_t* numerical,
    const real_t* analytical,
    int n) {

    real_t sum = 0.0;
    for (int i = 0; i < n; i++) {
        real_t diff = numerical[i] - analytical[i];
        sum += diff * diff;
    }
    return sqrt(sum / n);
}

real_t Validation::compute_Linf_error(
    const real_t* numerical,
    const real_t* analytical,
    int n) {

    real_t max_error = 0.0;
    for (int i = 0; i < n; i++) {
        real_t error = fabs(numerical[i] - analytical[i]);
        max_error = std::max(max_error, error);
    }
    return max_error;
}

real_t Validation::compute_total_mass(
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    int halo_width) {

    real_t total_mass = 0.0;

    for (int j = halo_width; j < ny - halo_width; j++) {
        for (int i = halo_width; i < nx - halo_width; i++) {
            int idx = j * nx + i;
            total_mass += cells[idx].U.h * dx * dy;
        }
    }

    return total_mass;
}

real_t Validation::check_mass_conservation(
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t initial_mass) {

    real_t current_mass = compute_total_mass(cells, nx, ny, dx, dy);
    real_t relative_error = fabs(current_mass - initial_mass) / (initial_mass + 1e-10);

    return relative_error;
}

bool Validation::check_physical_validity(
    const CellData* cells,
    int nx, int ny) {

    bool is_valid = true;

    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            int idx = j * nx + i;

            // Check for negative depth
            if (cells[idx].U.h < -1e-6) {
                std::cerr << "Warning: Negative depth at (" << i << "," << j
                         << "): h = " << cells[idx].U.h << std::endl;
                is_valid = false;
            }

            // Check for unreasonable velocities (>100 m/s)
            if (cells[idx].U.h > Constants::MIN_DEPTH) {
                real_t u = cells[idx].U.qx / cells[idx].U.h;
                real_t v = cells[idx].U.qy / cells[idx].U.h;
                real_t vel_mag = sqrt(u * u + v * v);

                if (vel_mag > 100.0) {
                    std::cerr << "Warning: Unreasonable velocity at (" << i << "," << j
                             << "): |V| = " << vel_mag << " m/s" << std::endl;
                    is_valid = false;
                }

                // Check for NaN or Inf
                if (std::isnan(cells[idx].U.h) || std::isinf(cells[idx].U.h) ||
                    std::isnan(u) || std::isinf(u) ||
                    std::isnan(v) || std::isinf(v)) {
                    std::cerr << "Warning: NaN or Inf detected at (" << i << "," << j << ")" << std::endl;
                    is_valid = false;
                }
            }
        }
    }

    return is_valid;
}

Validation::Statistics Validation::compute_statistics(
    const real_t* data,
    int n) {

    Statistics stats;

    if (n == 0) {
        stats.min = stats.max = stats.mean = stats.std_dev = 0.0;
        return stats;
    }

    stats.min = data[0];
    stats.max = data[0];
    real_t sum = 0.0;

    for (int i = 0; i < n; i++) {
        stats.min = std::min(stats.min, data[i]);
        stats.max = std::max(stats.max, data[i]);
        sum += data[i];
    }

    stats.mean = sum / n;

    // Compute standard deviation
    real_t sum_sq = 0.0;
    for (int i = 0; i < n; i++) {
        real_t diff = data[i] - stats.mean;
        sum_sq += diff * diff;
    }
    stats.std_dev = sqrt(sum_sq / n);

    return stats;
}

std::vector<real_t> Validation::extract_centerline_x(
    const CellData* cells,
    int nx, int ny,
    int halo_width) {

    int j_center = ny / 2;
    std::vector<real_t> profile;

    for (int i = halo_width; i < nx - halo_width; i++) {
        int idx = j_center * nx + i;
        profile.push_back(cells[idx].U.h);
    }

    return profile;
}

void Validation::compare_with_analytical(
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin,
    real_t time,
    int test_case) {

    if (test_case == 0) {
        // 1D Dam Break - compare with Ritter's solution
        std::cout << "\n=== Comparison with Analytical Solution ===" << std::endl;
        std::cout << "Test: 1D Dam Break (Ritter's solution)" << std::endl;
        std::cout << "Time: " << time << " s" << std::endl;

        // Extract centerline
        int j_center = ny / 2;
        int n_points = nx - 4; // Exclude halos

        std::vector<real_t> numerical(n_points);
        std::vector<real_t> analytical(n_points);

        for (int i = 2; i < nx - 2; i++) {
            int idx = j_center * nx + i;
            real_t x = xmin + i * dx;

            numerical[i - 2] = cells[idx].U.h;
            analytical[i - 2] = TestCases::DamBreak1D::analytical_solution(
                x, time, xmin + 0.5 * nx * dx, 10.0, 1.0);
        }

        real_t L1 = compute_L1_error(numerical.data(), analytical.data(), n_points);
        real_t L2 = compute_L2_error(numerical.data(), analytical.data(), n_points);
        real_t Linf = compute_Linf_error(numerical.data(), analytical.data(), n_points);

        std::cout << "L1 error:   " << L1 << std::endl;
        std::cout << "L2 error:   " << L2 << std::endl;
        std::cout << "Linf error: " << Linf << std::endl;
        std::cout << "==========================================" << std::endl;

    } else if (test_case == 5) {
        // Lake at Rest - should have zero velocity
        std::cout << "\n=== Lake at Rest Validation ===" << std::endl;

        real_t max_vel = 0.0;
        real_t max_depth_change = 0.0;
        real_t initial_h = 1.0; // From test case

        for (int j = 2; j < ny - 2; j++) {
            for (int i = 2; i < nx - 2; i++) {
                int idx = j * nx + i;

                if (cells[idx].U.h > Constants::MIN_DEPTH) {
                    real_t u = cells[idx].U.qx / cells[idx].U.h;
                    real_t v = cells[idx].U.qy / cells[idx].U.h;
                    real_t vel_mag = sqrt(u * u + v * v);
                    max_vel = std::max(max_vel, vel_mag);

                    real_t eta = cells[idx].z + cells[idx].U.h;
                    max_depth_change = std::max(max_depth_change, fabs(eta - initial_h));
                }
            }
        }

        std::cout << "Maximum velocity:        " << max_vel << " m/s" << std::endl;
        std::cout << "Max surface change:      " << max_depth_change << " m" << std::endl;
        std::cout << "Well-balanced property:  "
                  << (max_vel < 1e-4 ? "PASS" : "FAIL") << std::endl;
        std::cout << "==========================================" << std::endl;
    }
}
