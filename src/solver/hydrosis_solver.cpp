#include "hydrosis_solver.h"
#include "vtk_writer.h"
#include "validation.h"
#include "terrain_reader.h"
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
    , total_comm_time_(0.0)
    , output_vtk_(false)
    , enable_validation_(false)
    , test_case_id_(0)
    , initial_mass_(0.0) {
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

    // Compute initial mass for validation
    if (enable_validation_) {
        copy_to_host();
        initial_mass_ = Validation::compute_total_mass(
            h_cells_, params_.nx, params_.ny, params_.dx, params_.dy);

        if (!use_multi_gpu_ || multi_gpu_->is_root()) {
            std::cout << "Initial mass: " << initial_mass_ << " m^3" << std::endl;
        }
    }
}

void HydroSisSolver::set_bed_elevation(const std::string& filename) {
    if (filename.empty()) {
        // Flat bed - already initialized to 0
        if (!use_multi_gpu_ || multi_gpu_->is_root()) {
            std::cout << "Using flat bed (z = 0)" << std::endl;
        }
        return;
    }

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "Loading bed elevation from: " << filename << std::endl;
    }

    // Allocate temporary array for elevation data
    int n_cells = params_.nx * params_.ny;
    real_t* z_elevation = new real_t[n_cells];

    // Initialize to zero
    for (int i = 0; i < n_cells; i++) {
        z_elevation[i] = 0.0;
    }

    // Check file extension
    std::string ext = filename.substr(filename.find_last_of(".") + 1);

    if (ext == "asc" || ext == "ASC") {
        // Load ASCII Grid format
        TerrainReader::TerrainData terrain;
        if (TerrainReader::load_ascii_grid(filename, terrain)) {
            TerrainReader::print_statistics(terrain);

            // Interpolate to simulation grid
            TerrainReader::interpolate_to_grid(
                terrain, z_elevation,
                params_.nx, params_.ny,
                params_.xmin, params_.ymin,
                params_.dx, params_.dy
            );
        } else {
            std::cerr << "Warning: Failed to load terrain file, using flat bed" << std::endl;
        }
    } else if (ext == "bin" || ext == "BIN") {
        // Binary format - need grid dimensions
        TerrainReader::TerrainData terrain;
        if (TerrainReader::load_binary(filename, terrain, params_.nx, params_.ny)) {
            TerrainReader::print_statistics(terrain);

            // Copy directly (assuming same grid)
            for (int i = 0; i < n_cells; i++) {
                z_elevation[i] = terrain.elevation[i];
            }
        } else {
            std::cerr << "Warning: Failed to load terrain file, using flat bed" << std::endl;
        }
    } else {
        std::cerr << "Warning: Unknown terrain file format: " << ext << std::endl;
        std::cerr << "Supported formats: .asc (ASCII Grid), .bin (Binary)" << std::endl;
    }

    // Copy elevation data to host cells
    copy_to_host();
    for (int i = 0; i < n_cells; i++) {
        h_cells_[i].z = z_elevation[i];
    }
    copy_to_device();

    // Cleanup
    delete[] z_elevation;

    if (!use_multi_gpu_ || multi_gpu_->is_root()) {
        std::cout << "Bed elevation loaded successfully" << std::endl;
    }
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
            // Save results
            if (output_vtk_) {
                copy_to_host();
                std::string vtk_file = VTKWriter::generate_filename("output", output_count, ".vtk");
                VTKWriter::write_structured_grid(
                    vtk_file, h_cells_,
                    params_.nx, params_.ny,
                    params_.dx, params_.dy,
                    params_.xmin, params_.ymin,
                    current_time_, Constants::HALO_WIDTH);
            } else {
                char filename[256];
                sprintf(filename, "output_%04d.dat", output_count);
                save_results(filename, current_time_);
            }

            // Validation checks
            if (enable_validation_) {
                copy_to_host();

                // Check mass conservation
                real_t mass_error = Validation::check_mass_conservation(
                    h_cells_, params_.nx, params_.ny,
                    params_.dx, params_.dy, initial_mass_);

                // Check physical validity
                bool is_valid = Validation::check_physical_validity(
                    h_cells_, params_.nx, params_.ny);

                if (!use_multi_gpu_ || multi_gpu_->is_root()) {
                    std::cout << "Step " << step_count_
                              << ", Time = " << current_time_
                              << " s, dt = " << dt << " s"
                              << ", Mass error = " << (mass_error * 100.0) << "%"
                              << ", Valid = " << (is_valid ? "Yes" : "No")
                              << std::endl;
                }
            } else {
                if (!use_multi_gpu_ || multi_gpu_->is_root()) {
                    std::cout << "Step " << step_count_
                              << ", Time = " << current_time_
                              << " s, dt = " << dt
                              << " s" << std::endl;
                }
            }

            next_output_time += params_.output_interval;
            output_count++;
        }
    }

    // Final validation report
    if (enable_validation_) {
        copy_to_host();

        if (!use_multi_gpu_ || multi_gpu_->is_root()) {
            Validation::compare_with_analytical(
                h_cells_, params_.nx, params_.ny,
                params_.dx, params_.dy, params_.xmin,
                current_time_, test_case_id_);
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
