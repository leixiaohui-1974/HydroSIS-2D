#ifndef VALIDATION_H
#define VALIDATION_H

#include "hydrosis_types.h"
#include <cmath>
#include <vector>

/**
 * @brief Validation and error analysis utilities
 */
class Validation {
public:
    /**
     * @brief Compute L1 error norm
     * @param numerical Numerical solution
     * @param analytical Analytical solution
     * @param n Number of points
     * @return L1 error
     */
    static real_t compute_L1_error(
        const real_t* numerical,
        const real_t* analytical,
        int n);

    /**
     * @brief Compute L2 error norm
     */
    static real_t compute_L2_error(
        const real_t* numerical,
        const real_t* analytical,
        int n);

    /**
     * @brief Compute L-infinity error norm
     */
    static real_t compute_Linf_error(
        const real_t* numerical,
        const real_t* analytical,
        int n);

    /**
     * @brief Check mass conservation
     * @param cells Cell data
     * @param nx, ny Grid dimensions
     * @param dx, dy Grid spacing
     * @param initial_mass Initial total mass
     * @return Relative mass error
     */
    static real_t check_mass_conservation(
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        real_t initial_mass);

    /**
     * @brief Compute total mass
     */
    static real_t compute_total_mass(
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        int halo_width = 2);

    /**
     * @brief Check if solution is physically valid
     * @return true if valid (no negative depths, reasonable velocities)
     */
    static bool check_physical_validity(
        const CellData* cells,
        int nx, int ny);

    /**
     * @brief Compute statistics (min, max, mean) of a field
     */
    struct Statistics {
        real_t min;
        real_t max;
        real_t mean;
        real_t std_dev;
    };

    static Statistics compute_statistics(
        const real_t* data,
        int n);

    /**
     * @brief Extract 1D profile along centerline
     */
    static std::vector<real_t> extract_centerline_x(
        const CellData* cells,
        int nx, int ny,
        int halo_width = 2);

    /**
     * @brief Compare with analytical solution (if available)
     */
    static void compare_with_analytical(
        const CellData* cells,
        int nx, int ny,
        real_t dx, real_t dy,
        real_t xmin,
        real_t time,
        int test_case);
};

#endif // VALIDATION_H
