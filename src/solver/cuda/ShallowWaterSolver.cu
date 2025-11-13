/**
 * @file ShallowWaterSolver.cu
 * @brief Main CUDA shallow water solver implementation
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include "ShallowWaterSolver.cuh"
#include "RiemannSolver.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>

// Include kernel implementations
#include "kernels/flux_kernels.cu"
#include "kernels/update_kernels.cu"
#include "kernels/source_kernels.cu"
#include "kernels/bc_kernels.cu"
#include "kernels/muscl_kernels.cu"

namespace hydrosis2d {
namespace cuda {

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + \
                cudaGetErrorString(error)); \
        } \
    } while(0)

/**
 * @brief ShallowWaterSolver constructor
 */
ShallowWaterSolver::ShallowWaterSolver()
    : initialized_(false)
    , current_time_(0.0)
    , current_step_(0)
    , callback_(nullptr)
    , callback_interval_(100)
{
}

/**
 * @brief ShallowWaterSolver destructor
 */
ShallowWaterSolver::~ShallowWaterSolver() {
    cleanup();
}

/**
 * @brief Initialize solver from configuration file
 */
void ShallowWaterSolver::initialize(const std::string& config_file) {
    // Load configuration (JSON parsing would go here)
    // For now, use default configuration
    config_.nx = 200;
    config_.ny = 100;
    config_.dx = 1.0;
    config_.dy = 1.0;
    config_.g = 9.81;
    config_.cfl = 0.8;
    config_.riemann_solver = RiemannSolver::HLLC;
    config_.spatial_order = 1;  // Start with 1st order
    config_.time_integrator = TimeIntegrator::EULER;
    config_.dry_threshold = 1e-6;

    allocateMemory();
    initialized_ = true;
}

/**
 * @brief Allocate GPU memory
 */
void ShallowWaterSolver::allocateMemory() {
    size_t ncells = config_.nx * config_.ny;
    size_t nfaces_x = (config_.nx + 1) * config_.ny;
    size_t nfaces_y = config_.nx * (config_.ny + 1);

    // Conservative variables
    CUDA_CHECK(cudaMalloc(&dev_mem_.h, ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_mem_.hu, ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_mem_.hv, ncells * sizeof(double)));

    // Primitive variables
    CUDA_CHECK(cudaMalloc(&dev_mem_.u, ncells * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_mem_.v, ncells * sizeof(double)));

    // Terrain
    CUDA_CHECK(cudaMalloc(&dev_mem_.z, ncells * sizeof(double)));

    // Fluxes
    CUDA_CHECK(cudaMalloc(&dev_mem_.flux_x, 3 * nfaces_x * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dev_mem_.flux_y, 3 * nfaces_y * sizeof(double)));

    // Temporary storage for RK2
    if (config_.time_integrator == TimeIntegrator::RK2 ||
        config_.time_integrator == TimeIntegrator::RK3) {
        CUDA_CHECK(cudaMalloc(&dev_mem_.h_temp, ncells * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev_mem_.hu_temp, ncells * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev_mem_.hv_temp, ncells * sizeof(double)));

        CUDA_CHECK(cudaMalloc(&dev_mem_.flux_x_temp, 3 * nfaces_x * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dev_mem_.flux_y_temp, 3 * nfaces_y * sizeof(double)));
    }

    // Manning roughness
    CUDA_CHECK(cudaMalloc(&dev_mem_.manning, ncells * sizeof(double)));

    // Boundary condition storage
    CUDA_CHECK(cudaMalloc(&dev_mem_.bc_types, 4 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&dev_mem_.bc_values, 4 * std::max(config_.nx, config_.ny) * 3 * sizeof(double)));

    // CFL computation
    CUDA_CHECK(cudaMalloc(&dev_mem_.max_wave_speed, sizeof(double)));
}

/**
 * @brief Clean up GPU memory
 */
void ShallowWaterSolver::cleanup() {
    if (!initialized_) return;

    cudaFree(dev_mem_.h);
    cudaFree(dev_mem_.hu);
    cudaFree(dev_mem_.hv);
    cudaFree(dev_mem_.u);
    cudaFree(dev_mem_.v);
    cudaFree(dev_mem_.z);
    cudaFree(dev_mem_.flux_x);
    cudaFree(dev_mem_.flux_y);
    cudaFree(dev_mem_.manning);
    cudaFree(dev_mem_.bc_types);
    cudaFree(dev_mem_.bc_values);
    cudaFree(dev_mem_.max_wave_speed);

    if (dev_mem_.h_temp) {
        cudaFree(dev_mem_.h_temp);
        cudaFree(dev_mem_.hu_temp);
        cudaFree(dev_mem_.hv_temp);
        cudaFree(dev_mem_.flux_x_temp);
        cudaFree(dev_mem_.flux_y_temp);
    }

    initialized_ = false;
}

/**
 * @brief Set initial conditions from host arrays
 */
void ShallowWaterSolver::setInitialConditions(
    const double* h_host,
    const double* u_host,
    const double* v_host,
    const double* z_host
) {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    size_t ncells = config_.nx * config_.ny;

    // Copy to device
    CUDA_CHECK(cudaMemcpy(dev_mem_.h, h_host, ncells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mem_.u, u_host, ncells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mem_.v, v_host, ncells * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_mem_.z, z_host, ncells * sizeof(double), cudaMemcpyHostToDevice));

    // Compute conservative variables
    dim3 blockDim(16, 16);
    dim3 gridDim((config_.nx + 15) / 16, (config_.ny + 15) / 16);

    primitiveToConservative<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        dev_mem_.hu, dev_mem_.hv,
        config_.nx, config_.ny
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    current_time_ = 0.0;
    current_step_ = 0;
}

/**
 * @brief Compute adaptive time step based on CFL condition
 */
double ShallowWaterSolver::computeTimeStep() {
    // Reset max wave speed
    double zero = 0.0;
    CUDA_CHECK(cudaMemcpy(dev_mem_.max_wave_speed, &zero, sizeof(double), cudaMemcpyHostToDevice));

    // Compute max wave speed
    dim3 blockDim(16, 16);
    dim3 gridDim((config_.nx + 15) / 16, (config_.ny + 15) / 16);

    computeCFLTimeStep_kernel<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        config_.dx, config_.dy,
        config_.g,
        config_.nx, config_.ny,
        dev_mem_.max_wave_speed
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Get max wave speed from device
    double max_speed;
    CUDA_CHECK(cudaMemcpy(&max_speed, dev_mem_.max_wave_speed, sizeof(double), cudaMemcpyDeviceToHost));

    // Compute dt
    if (max_speed > 1e-12) {
        return config_.cfl / max_speed;
    } else {
        return 1e-3;  // Default small time step
    }
}

/**
 * @brief Apply boundary conditions
 */
void ShallowWaterSolver::applyBoundaryConditions() {
    dim3 blockDim(16, 16);
    dim3 gridDim((config_.nx + 15) / 16, (config_.ny + 15) / 16);

    // For now, apply wall BC on all boundaries
    applyWallBC<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        0  // Left wall
    );

    applyWallBC<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        1  // Right wall
    );

    applyWallBC<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        2  // Bottom wall
    );

    applyWallBC<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        3  // Top wall
    );

    CUDA_CHECK(cudaGetLastError());
}

/**
 * @brief Perform single time step (Euler method)
 */
void ShallowWaterSolver::stepEuler(double dt) {
    dim3 blockDim(16, 16);
    dim3 gridDim_cells((config_.nx + 15) / 16, (config_.ny + 15) / 16);

    // 1. Convert conservative to primitive
    conservativeToPrimitive<<<gridDim_cells, blockDim>>>(
        dev_mem_.h, dev_mem_.hu, dev_mem_.hv,
        dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        config_.dry_threshold
    );
    CUDA_CHECK(cudaGetLastError());

    // 2. Apply boundary conditions
    applyBoundaryConditions();

    // 3. Compute fluxes
    dim3 gridDim_faces_x((config_.nx + 1 + 15) / 16, (config_.ny + 15) / 16);
    computeFluxesX_FirstOrder<<<gridDim_faces_x, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        dev_mem_.flux_x,
        config_.nx, config_.ny,
        config_.dx, config_.dy,
        config_.g,
        static_cast<int>(config_.riemann_solver)
    );

    dim3 gridDim_faces_y((config_.nx + 15) / 16, (config_.ny + 1 + 15) / 16);
    computeFluxesY_FirstOrder<<<gridDim_faces_y, blockDim>>>(
        dev_mem_.h, dev_mem_.u, dev_mem_.v,
        dev_mem_.flux_y,
        config_.nx, config_.ny,
        config_.dx, config_.dy,
        config_.g,
        static_cast<int>(config_.riemann_solver)
    );
    CUDA_CHECK(cudaGetLastError());

    // 4. Update conservative variables
    updateConservativeVariables_Euler<<<gridDim_cells, blockDim>>>(
        dev_mem_.h, dev_mem_.hu, dev_mem_.hv,
        dev_mem_.flux_x, dev_mem_.flux_y,
        config_.nx, config_.ny,
        config_.dx, config_.dy,
        dt
    );
    CUDA_CHECK(cudaGetLastError());

    // 5. Apply source terms if terrain is not flat
    // TODO: Add source term application

    CUDA_CHECK(cudaDeviceSynchronize());
}

/**
 * @brief Perform single time step (RK2 method)
 */
void ShallowWaterSolver::stepRK2(double dt) {
    // Implementation of RK2 time stepping
    // TODO: Implement full RK2 using predictor-corrector kernels
    stepEuler(dt);  // Fallback to Euler for now
}

/**
 * @brief Perform single time step with adaptive dt
 */
void ShallowWaterSolver::step(double dt) {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    switch (config_.time_integrator) {
        case TimeIntegrator::EULER:
            stepEuler(dt);
            break;
        case TimeIntegrator::RK2:
            stepRK2(dt);
            break;
        default:
            stepEuler(dt);
    }

    current_time_ += dt;
    current_step_++;

    // Callback
    if (callback_ && current_step_ % callback_interval_ == 0) {
        callback_(current_time_, current_step_, dt);
    }
}

/**
 * @brief Run simulation until end time
 */
void ShallowWaterSolver::run(double t_end) {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    while (current_time_ < t_end) {
        double dt = computeTimeStep();

        // Don't overshoot end time
        if (current_time_ + dt > t_end) {
            dt = t_end - current_time_;
        }

        step(dt);

        // Progress output
        if (current_step_ % 100 == 0) {
            printf("Step %d, t = %.3f / %.3f s, dt = %.6f s\n",
                   current_step_, current_time_, t_end, dt);
        }
    }

    printf("Simulation complete: %d steps, final time = %.3f s\n",
           current_step_, current_time_);
}

/**
 * @brief Get solution on host
 */
void ShallowWaterSolver::getSolution(double* h_out, double* u_out, double* v_out) {
    if (!initialized_) {
        throw std::runtime_error("Solver not initialized");
    }

    size_t ncells = config_.nx * config_.ny;

    // Convert to primitive first
    dim3 blockDim(16, 16);
    dim3 gridDim((config_.nx + 15) / 16, (config_.ny + 15) / 16);

    conservativeToPrimitive<<<gridDim, blockDim>>>(
        dev_mem_.h, dev_mem_.hu, dev_mem_.hv,
        dev_mem_.u, dev_mem_.v,
        config_.nx, config_.ny,
        config_.dry_threshold
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy to host
    CUDA_CHECK(cudaMemcpy(h_out, dev_mem_.h, ncells * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(u_out, dev_mem_.u, ncells * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_out, dev_mem_.v, ncells * sizeof(double), cudaMemcpyDeviceToHost));
}

/**
 * @brief Set progress callback
 */
void ShallowWaterSolver::setCallback(ProgressCallback callback, int interval) {
    callback_ = callback;
    callback_interval_ = interval;
}

/**
 * @brief Get current simulation time
 */
double ShallowWaterSolver::getCurrentTime() const {
    return current_time_;
}

/**
 * @brief Get current step number
 */
int ShallowWaterSolver::getCurrentStep() const {
    return current_step_;
}

/**
 * @brief Get configuration
 */
const SolverConfig& ShallowWaterSolver::getConfig() const {
    return config_;
}

} // namespace cuda
} // namespace hydrosis2d
