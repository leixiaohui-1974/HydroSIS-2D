#ifndef MULTI_GPU_H
#define MULTI_GPU_H

#include "hydrosis_types.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>

/**
 * @brief Multi-GPU parallel computing manager
 *
 * This class handles domain decomposition, GPU assignment, and
 * inter-GPU communication using CUDA-Aware MPI for maximum performance.
 */
class MultiGPU {
public:
    MultiGPU();
    ~MultiGPU();

    /**
     * @brief Initialize MPI and assign GPUs to processes
     * @param argc Command line argument count
     * @param argv Command line arguments
     */
    void initialize(int* argc, char*** argv);

    /**
     * @brief Decompose domain across multiple GPUs
     * @param nx_global Global grid size in x
     * @param ny_global Global grid size in y
     */
    void decompose_domain(int nx_global, int ny_global);

    /**
     * @brief Exchange halo/ghost cells between neighboring GPUs
     * @param cells Device pointer to cell data
     * @param stream CUDA stream for asynchronous operations
     */
    void exchange_halos(CellData* cells, cudaStream_t stream = 0);

    /**
     * @brief Gather results from all GPUs to root process
     * @param local_cells Local cell data on this GPU
     * @param global_cells Global cell data (only valid on root)
     */
    void gather_results(
        const CellData* local_cells,
        CellData* global_cells);

    /**
     * @brief Compute global maximum (e.g., for timestep)
     * @param local_value Local value on this GPU
     * @return Global maximum across all GPUs
     */
    real_t global_max(real_t local_value);

    /**
     * @brief Compute global minimum (e.g., for timestep)
     * @param local_value Local value on this GPU
     * @return Global minimum across all GPUs
     */
    real_t global_min(real_t local_value);

    /**
     * @brief Synchronize all GPUs
     */
    void barrier();

    /**
     * @brief Finalize MPI
     */
    void finalize();

    // Getters
    const DomainInfo& get_domain_info() const { return domain_info_; }
    int get_rank() const { return domain_info_.rank; }
    int get_num_procs() const { return domain_info_.num_procs; }
    bool is_root() const { return domain_info_.rank == 0; }

private:
    DomainInfo domain_info_;
    bool initialized_;

    // MPI datatypes for efficient communication
    MPI_Datatype mpi_cell_type_;
    MPI_Datatype mpi_halo_x_type_;
    MPI_Datatype mpi_halo_y_type_;

    // Device buffers for halo exchange
    CellData* halo_send_left_;
    CellData* halo_send_right_;
    CellData* halo_send_bottom_;
    CellData* halo_send_top_;
    CellData* halo_recv_left_;
    CellData* halo_recv_right_;
    CellData* halo_recv_bottom_;
    CellData* halo_recv_top_;

    int halo_size_x_;
    int halo_size_y_;

    /**
     * @brief Create MPI datatype for CellData structure
     */
    void create_mpi_datatypes();

    /**
     * @brief Allocate device buffers for halo exchange
     */
    void allocate_halo_buffers();

    /**
     * @brief Free device halo buffers
     */
    void free_halo_buffers();

    /**
     * @brief Pack halo cells into send buffers
     */
    void pack_halos(const CellData* cells, cudaStream_t stream);

    /**
     * @brief Unpack received halo cells
     */
    void unpack_halos(CellData* cells, cudaStream_t stream);

    /**
     * @brief Determine optimal domain decomposition
     */
    void compute_decomposition(
        int nx_global, int ny_global,
        int& npx, int& npy);
};

// CUDA kernels for halo packing/unpacking
namespace HaloKernels {

/**
 * @brief Pack left halo cells
 */
__global__ void pack_left_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Pack right halo cells
 */
__global__ void pack_right_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Pack bottom halo cells
 */
__global__ void pack_bottom_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Pack top halo cells
 */
__global__ void pack_top_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Unpack left halo cells
 */
__global__ void unpack_left_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Unpack right halo cells
 */
__global__ void unpack_right_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Unpack bottom halo cells
 */
__global__ void unpack_bottom_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

/**
 * @brief Unpack top halo cells
 */
__global__ void unpack_top_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width);

} // namespace HaloKernels

#endif // MULTI_GPU_H
