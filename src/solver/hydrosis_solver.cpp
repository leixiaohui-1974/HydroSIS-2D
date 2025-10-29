#include "hydrosis_solver.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <chrono>

HydroSisSolver::HydroSisSolver()
    : initialized_(false)
    , use_multi_gpu_(false)
    , d_cells_(nullptr)
    , d_cells_new_(nullptr)
    , h_cells_(nullptr)
    , current_time_(0.0)
    , step_count_(0)
    , multi_gpu_(nullptr)
    , d_dt_global_(nullptr)
    , total_compute_time_(0.0)
    , total_comm_time_(0.0) {
}

HydroSisSolver::~HydroSisSolver() {
    if (initialized_) {
        free_memory();
        CUDA_CHECK(cudaStreamDestroy(compute_stream_));
        CUDA_CHECK(cudaStreamDestroy(comm_stream_));
    }

    if (multi_gpu_) {
        delete multi_gpu_;
    }
}

void HydroSisSolver::initialize(const SimParams& params, bool use_multi_gpu) {
    params_ = params;
    use_multi_gpu_ = use_multi_gpu;

    if (use_multi_gpu_) {
        multi_gpu_ = new MultiGPU();
        // MPI already initialized externally
        multi_gpu_->decompose_domain(params_.nx, params_.ny);

        // Update local grid dimensions
        const DomainInfo& domain = multi_gpu_->get_domain_info();
        params_.nx = domain.nx_local + 2 * domain.halo_width;
        params_.ny = domain.ny_local + 2 * domain.halo_width;
    }

    // Create CUDA streams
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUDA_CHECK(cudaStreamCreate(&comm_stream_));

    // Allocate memory
    allocate_memory();

    current_time_ = params_.t_start;
    step_count_ = 0;

    initialized_ = true;

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "\n=== HydroSIS-2D Solver Initialized ===" << std::endl;
        std::cout << "Grid size: " << params_.nx << " x " << params_.ny << std::endl;
        std::cout << "Grid spacing: dx=" << params_.dx << " m, dy=" << params_.dy << " m" << std::endl;
        std::cout << "CFL number: " << params_.cfl << std::endl;
        std::cout << "Simulation time: " << params_.t_start << " to " << params_.t_end << " s" << std::endl;
        std::cout << "Multi-GPU: " << (use_multi_gpu_ ? "Enabled" : "Disabled") << std::endl;
        std::cout << "======================================\n" << std::endl;
    }
}

void HydroSisSolver::allocate_memory() {
    int n_cells = get_total_cells();

    // Device memory
    CUDA_CHECK(cudaMalloc(&d_cells_, n_cells * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&d_cells_new_, n_cells * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&d_dt_global_, sizeof(real_t)));

    // Host memory
    h_cells_ = new CellData[n_cells];

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "Memory allocated: "
                  << (2 * n_cells * sizeof(CellData) / (1024.0 * 1024.0))
                  << " MB on GPU" << std::endl;
    }
}

void HydroSisSolver::free_memory() {
    if (d_cells_) CUDA_CHECK(cudaFree(d_cells_));
    if (d_cells_new_) CUDA_CHECK(cudaFree(d_cells_new_));
    if (d_dt_global_) CUDA_CHECK(cudaFree(d_dt_global_));

    if (h_cells_) delete[] h_cells_;

    d_cells_ = nullptr;
    d_cells_new_ = nullptr;
    d_dt_global_ = nullptr;
    h_cells_ = nullptr;
}

int HydroSisSolver::get_total_cells() const {
    return params_.nx * params_.ny;
}

void HydroSisSolver::get_kernel_dims(dim3& block, dim3& grid) const {
    block = dim3(16, 16);
    grid = dim3(
        (params_.nx + block.x - 1) / block.x,
        (params_.ny + block.y - 1) / block.y
    );
}

void HydroSisSolver::set_initial_conditions(int test_case) {
    dim3 block, grid;
    get_kernel_dims(block, grid);

    CudaKernels::initialize_grid_kernel<<<grid, block>>>(
        d_cells_,
        params_.nx, params_.ny,
        params_.dx, params_.dy,
        params_.xmin, params_.ymin,
        test_case
    );

    CUDA_CHECK(cudaDeviceSynchronize());

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "Initial conditions set (test case " << test_case << ")" << std::endl;
    }
}

void HydroSisSolver::set_bed_elevation(const std::string& filename) {
    if (filename.empty()) {
        // Flat bed - already initialized to 0
        return;
    }

    // TODO: Load from file
    std::cout << "Loading bed elevation from: " << filename << std::endl;
}

real_t HydroSisSolver::compute_timestep() {
    // Set initial large value
    real_t h_dt_max = params_.dt_max;
    CUDA_CHECK(cudaMemcpy(d_dt_global_, &h_dt_max, sizeof(real_t), cudaMemcpyHostToDevice));

    // Compute timestep on GPU
    dim3 block, grid;
    get_kernel_dims(block, grid);

    CudaKernels::compute_timestep_kernel<<<grid, block, 0, compute_stream_>>>(
        d_cells_,
        params_.nx, params_.ny,
        params_.dx, params_.dy,
        params_.cfl, params_.g,
        d_dt_global_
    );

    // Copy result back
    real_t dt_local;
    CUDA_CHECK(cudaMemcpy(&dt_local, d_dt_global_, sizeof(real_t), cudaMemcpyDeviceToHost));

    // Global reduction for multi-GPU
    if (use_multi_gpu_) {
        return multi_gpu_->global_min(dt_local);
    }

    return std::min(dt_local, params_.dt_max);
}

void HydroSisSolver::update_cells(real_t dt) {
    dim3 block, grid;
    get_kernel_dims(block, grid);

    CudaKernels::update_cells_kernel<<<grid, block, 0, compute_stream_>>>(
        d_cells_,
        d_cells_new_,
        params_.nx, params_.ny,
        params_.dx, params_.dy,
        dt,
        params_
    );
}

void HydroSisSolver::apply_sources(real_t dt) {
    dim3 block, grid;
    get_kernel_dims(block, grid);

    CudaKernels::apply_source_terms_kernel<<<grid, block, 0, compute_stream_>>>(
        d_cells_new_,
        params_.nx, params_.ny,
        params_.dx, params_.dy,
        dt,
        params_.g
    );
}

void HydroSisSolver::apply_boundaries() {
    if (use_multi_gpu_) {
        // Handled by halo exchange
        return;
    }

    int max_dim = std::max(params_.nx, params_.ny);
    dim3 block(256);
    dim3 grid((max_dim + block.x - 1) / block.x);

    CudaKernels::apply_boundary_conditions_kernel<<<grid, block, 0, compute_stream_>>>(
        d_cells_new_,
        params_.nx, params_.ny,
        params_.bc_type
    );
}

void HydroSisSolver::apply_wet_dry() {
    dim3 block, grid;
    get_kernel_dims(block, grid);

    CudaKernels::wet_dry_treatment_kernel<<<grid, block, 0, compute_stream_>>>(
        d_cells_new_,
        params_.nx, params_.ny,
        params_.h_dry
    );
}

void HydroSisSolver::exchange_ghost_cells() {
    if (use_multi_gpu_) {
        auto start = std::chrono::high_resolution_clock::now();

        multi_gpu_->exchange_halos(d_cells_new_, comm_stream_);
        CUDA_CHECK(cudaStreamSynchronize(comm_stream_));

        auto end = std::chrono::high_resolution_clock::now();
        total_comm_time_ += std::chrono::duration<float>(end - start).count();
    }
}

real_t HydroSisSolver::step() {
    auto step_start = std::chrono::high_resolution_clock::now();

    // Compute time step
    real_t dt = compute_timestep();

    // Don't overshoot end time
    if (current_time_ + dt > params_.t_end) {
        dt = params_.t_end - current_time_;
    }

    // Update cells
    update_cells(dt);

    // Apply source terms
    apply_sources(dt);

    // Synchronize compute stream
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

    // Apply wet/dry treatment
    apply_wet_dry();

    // Apply boundary conditions
    apply_boundaries();

    // Exchange ghost cells (multi-GPU)
    exchange_ghost_cells();

    // Swap buffers
    std::swap(d_cells_, d_cells_new_);

    // Update time
    current_time_ += dt;
    step_count_++;

    auto step_end = std::chrono::high_resolution_clock::now();
    total_compute_time_ += std::chrono::duration<float>(step_end - step_start).count();

    return dt;
}

void HydroSisSolver::run() {
    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "Starting simulation..." << std::endl;
    }

    real_t next_output_time = current_time_ + params_.output_interval;
    int output_count = 0;

    auto start_time = std::chrono::high_resolution_clock::now();

    while (current_time_ < params_.t_end) {
        real_t dt = step();

        // Output results
        if (current_time_ >= next_output_time) {
            char filename[256];
            sprintf(filename, "output_%04d.dat", output_count);
            save_results(filename, current_time_);
            next_output_time += params_.output_interval;
            output_count++;

            if (!use_multi_gpu_ || multi_gpu_->is_root()) {
                std::cout << "Step " << step_count_
                          << ", Time = " << current_time_
                          << " s, dt = " << dt
                          << " s" << std::endl;
            }
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float>(end_time - start_time).count();

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "\nSimulation completed!" << std::endl;
        std::cout << "Total steps: " << step_count_ << std::endl;
        std::cout << "Total time: " << total_time << " s" << std::endl;
        std::cout << "Average time per step: " << (total_time / step_count_ * 1000.0) << " ms" << std::endl;

        int n_cells = params_.nx * params_.ny;
        float cell_updates_per_sec = (n_cells * step_count_) / total_time;
        std::cout << "Performance: " << (cell_updates_per_sec / 1e9) << " gigacells/s" << std::endl;
    }
}

void HydroSisSolver::copy_to_host() {
    int n_cells = get_total_cells();
    CUDA_CHECK(cudaMemcpy(h_cells_, d_cells_, n_cells * sizeof(CellData), cudaMemcpyDeviceToHost));
}

void HydroSisSolver::copy_to_device() {
    int n_cells = get_total_cells();
    CUDA_CHECK(cudaMemcpy(d_cells_, h_cells_, n_cells * sizeof(CellData), cudaMemcpyHostToDevice));
}

void HydroSisSolver::save_results(const std::string& filename, real_t time) {
    copy_to_host();

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::ofstream file(filename);
        file << "# Time: " << time << " s\n";
        file << "# x y h u v z\n";

        for (int j = Constants::HALO_WIDTH; j < params_.ny - Constants::HALO_WIDTH; j++) {
            for (int i = Constants::HALO_WIDTH; i < params_.nx - Constants::HALO_WIDTH; i++) {
                int idx = j * params_.nx + i;
                const CellData& cell = h_cells_[idx];

                real_t x = params_.xmin + i * params_.dx;
                real_t y = params_.ymin + j * params_.dy;

                PrimitiveVars W;
                W.h = cell.U.h;
                if (cell.U.h > Constants::MIN_DEPTH) {
                    W.u = cell.U.qx / cell.U.h;
                    W.v = cell.U.qy / cell.U.h;
                } else {
                    W.u = 0.0;
                    W.v = 0.0;
                }

                file << x << " " << y << " "
                     << W.h << " " << W.u << " " << W.v << " "
                     << cell.z << "\n";
            }
        }

        file.close();
    }
}

void HydroSisSolver::print_performance_stats() const {
    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "\n=== Performance Statistics ===" << std::endl;
        std::cout << "Total compute time: " << total_compute_time_ << " s" << std::endl;
        std::cout << "Total communication time: " << total_comm_time_ << " s" << std::endl;
        std::cout << "Compute/Comm ratio: " << (total_compute_time_ / (total_comm_time_ + 1e-10)) << std::endl;
        std::cout << "===============================" << std::endl;
    }
}
