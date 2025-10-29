#ifndef HYDROSIS_SOLVER_H
#define HYDROSIS_SOLVER_H

#include "hydrosis_types.h"
#include "cuda_kernels.cuh"
#include "multi_gpu.h"
#include <string>
#include <vector>
#include <cuda_runtime.h>

/**
 * @brief Main GPU-accelerated 2D hydrodynamic solver
 *
 * This class implements a high-performance finite volume solver for
 * the 2D shallow water equations with multi-GPU support.
 */
class HydroSisSolver {
public:
    HydroSisSolver();
    ~HydroSisSolver();

    /**
     * @brief Initialize solver with parameters
     * @param params Simulation parameters
     * @param use_multi_gpu Enable multi-GPU parallelization
     */
    void initialize(const SimParams& params, bool use_multi_gpu = false);

    /**
     * @brief Set initial conditions from test case
     * @param test_case Test case ID (0: dam break, 1: circular dam, etc.)
     */
    void set_initial_conditions(int test_case);

    /**
     * @brief Set bed elevation from file or function
     * @param filename Path to elevation data (empty for flat bed)
     */
    void set_bed_elevation(const std::string& filename = "");

    /**
     * @brief Run simulation
     */
    void run();

    /**
     * @brief Perform single time step
     * @return Current time step size
     */
    real_t step();

    /**
     * @brief Save results to file
     * @param filename Output filename
     * @param time Current simulation time
     */
    void save_results(const std::string& filename, real_t time);

    /**
     * @brief Get current simulation time
     */
    real_t get_time() const { return current_time_; }

    /**
     * @brief Get number of time steps completed
     */
    int get_step_count() const { return step_count_; }

    /**
     * @brief Get performance metrics
     */
    void print_performance_stats() const;

    /**
     * @brief Enable VTK output for visualization
     */
    void enable_vtk_output(bool enable) { output_vtk_ = enable; }

    /**
     * @brief Enable validation and error analysis
     */
    void enable_validation(bool enable, int test_case_id) {
        enable_validation_ = enable;
        test_case_id_ = test_case_id;
    }

private:
    SimParams params_;
    bool initialized_;
    bool use_multi_gpu_;

    // Grid data (device)
    CellData* d_cells_;
    CellData* d_cells_new_;

    // Grid data (host) - for I/O
    CellData* h_cells_;

    // Time stepping
    real_t current_time_;
    int step_count_;

    // CUDA streams for overlapping computation and communication
    cudaStream_t compute_stream_;
    cudaStream_t comm_stream_;

    // Multi-GPU manager
    MultiGPU* multi_gpu_;

    // Device memory for timestep reduction
    real_t* d_dt_global_;

    // Performance tracking
    float total_compute_time_;
    float total_comm_time_;

    // Output and validation
    bool output_vtk_;
    bool enable_validation_;
    int test_case_id_;
    real_t initial_mass_;

    /**
     * @brief Allocate device and host memory
     */
    void allocate_memory();

    /**
     * @brief Free all allocated memory
     */
    void free_memory();

    /**
     * @brief Compute time step based on CFL condition
     * @return Time step size
     */
    real_t compute_timestep();

    /**
     * @brief Update cells for one time step
     * @param dt Time step size
     */
    void update_cells(real_t dt);

    /**
     * @brief Apply source terms (bed slope, friction)
     * @param dt Time step size
     */
    void apply_sources(real_t dt);

    /**
     * @brief Apply boundary conditions
     */
    void apply_boundaries();

    /**
     * @brief Apply wet/dry treatment
     */
    void apply_wet_dry();

    /**
     * @brief Exchange ghost cells (multi-GPU)
     */
    void exchange_ghost_cells();

    /**
     * @brief Copy data from device to host
     */
    void copy_to_host();

    /**
     * @brief Copy data from host to device
     */
    void copy_to_device();

    /**
     * @brief Get total number of cells (including halos)
     */
    int get_total_cells() const;

    /**
     * @brief Get grid dimensions for kernels
     */
    void get_kernel_dims(dim3& block, dim3& grid) const;
};

#endif // HYDROSIS_SOLVER_H
