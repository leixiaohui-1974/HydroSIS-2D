#include "multi_gpu.h"
#include <iostream>
#include <cmath>
#include <algorithm>

MultiGPU::MultiGPU()
    : initialized_(false)
    , halo_send_left_(nullptr)
    , halo_send_right_(nullptr)
    , halo_send_bottom_(nullptr)
    , halo_send_top_(nullptr)
    , halo_recv_left_(nullptr)
    , halo_recv_right_(nullptr)
    , halo_recv_bottom_(nullptr)
    , halo_recv_top_(nullptr)
    , halo_size_x_(0)
    , halo_size_y_(0) {
}

MultiGPU::~MultiGPU() {
    if (initialized_) {
        free_halo_buffers();
        MPI_Type_free(&mpi_cell_type_);
    }
}

void MultiGPU::initialize(int* argc, char*** argv) {
    MPI_Init(argc, argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &domain_info_.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &domain_info_.num_procs);

    // Assign GPU to this MPI rank
    int num_gpus;
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

    if (num_gpus == 0) {
        if (domain_info_.rank == 0) {
            std::cerr << "Error: No CUDA-capable GPUs found!" << std::endl;
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Round-robin GPU assignment
    domain_info_.gpu_id = domain_info_.rank % num_gpus;
    CUDA_CHECK(cudaSetDevice(domain_info_.gpu_id));

    if (domain_info_.rank == 0) {
        std::cout << "Multi-GPU initialization:" << std::endl;
        std::cout << "  Number of MPI processes: " << domain_info_.num_procs << std::endl;
        std::cout << "  Number of GPUs: " << num_gpus << std::endl;
    }

    // Print GPU assignment
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, domain_info_.gpu_id));

    std::cout << "Rank " << domain_info_.rank
              << " on " << processor_name
              << " using GPU " << domain_info_.gpu_id
              << " (" << prop.name << ")" << std::endl;

    barrier();

    create_mpi_datatypes();

    domain_info_.halo_width = Constants::HALO_WIDTH;
    initialized_ = true;
}

void MultiGPU::compute_decomposition(
    int nx_global, int ny_global,
    int& npx, int& npy) {

    // Find optimal 2D decomposition
    // Try to make domains as square as possible
    npx = (int)sqrt((double)domain_info_.num_procs);
    while (domain_info_.num_procs % npx != 0) npx--;
    npy = domain_info_.num_procs / npx;

    // Prefer more decomposition in the larger dimension
    if (nx_global < ny_global && npx > npy) {
        std::swap(npx, npy);
    }

    if (domain_info_.rank == 0) {
        std::cout << "Domain decomposition: " << npx << " x " << npy << std::endl;
    }
}

void MultiGPU::decompose_domain(int nx_global, int ny_global) {
    int npx, npy;
    compute_decomposition(nx_global, ny_global, npx, npy);

    // Determine this rank's position in 2D processor grid
    int px = domain_info_.rank % npx;
    int py = domain_info_.rank / npx;

    // Compute local domain size
    domain_info_.nx_local = nx_global / npx;
    domain_info_.ny_local = ny_global / npy;

    // Handle remainder cells
    if (px < nx_global % npx) domain_info_.nx_local++;
    if (py < ny_global % npy) domain_info_.ny_local++;

    // Compute starting indices
    domain_info_.i_start = px * (nx_global / npx) + std::min(px, nx_global % npx);
    domain_info_.j_start = py * (ny_global / npy) + std::min(py, ny_global % npy);

    domain_info_.i_end = domain_info_.i_start + domain_info_.nx_local;
    domain_info_.j_end = domain_info_.j_start + domain_info_.ny_local;

    // Determine neighbor ranks
    domain_info_.left_rank = (px > 0) ? (py * npx + px - 1) : MPI_PROC_NULL;
    domain_info_.right_rank = (px < npx - 1) ? (py * npx + px + 1) : MPI_PROC_NULL;
    domain_info_.bottom_rank = (py > 0) ? ((py - 1) * npx + px) : MPI_PROC_NULL;
    domain_info_.top_rank = (py < npy - 1) ? ((py + 1) * npx + px) : MPI_PROC_NULL;

    if (domain_info_.rank == 0) {
        std::cout << "\nDomain decomposition details:" << std::endl;
    }

    barrier();

    std::cout << "Rank " << domain_info_.rank
              << ": local size = " << domain_info_.nx_local << " x " << domain_info_.ny_local
              << ", indices [" << domain_info_.i_start << ":" << domain_info_.i_end
              << ", " << domain_info_.j_start << ":" << domain_info_.j_end << "]"
              << ", neighbors: L=" << domain_info_.left_rank
              << " R=" << domain_info_.right_rank
              << " B=" << domain_info_.bottom_rank
              << " T=" << domain_info_.top_rank << std::endl;

    barrier();

    // Allocate halo buffers
    allocate_halo_buffers();
}

void MultiGPU::create_mpi_datatypes() {
    // Create MPI datatype for ConservativeVars
    MPI_Datatype cons_type;
    int blocklengths[3] = {1, 1, 1};
    MPI_Aint displacements[3];
    MPI_Datatype types[3] = {
        (sizeof(real_t) == 8) ? MPI_DOUBLE : MPI_FLOAT,
        (sizeof(real_t) == 8) ? MPI_DOUBLE : MPI_FLOAT,
        (sizeof(real_t) == 8) ? MPI_DOUBLE : MPI_FLOAT
    };

    displacements[0] = offsetof(ConservativeVars, h);
    displacements[1] = offsetof(ConservativeVars, qx);
    displacements[2] = offsetof(ConservativeVars, qy);

    MPI_Type_create_struct(3, blocklengths, displacements, types, &cons_type);
    MPI_Type_commit(&cons_type);

    // Create MPI datatype for CellData (simplified - just send conservative vars)
    mpi_cell_type_ = cons_type;
}

void MultiGPU::allocate_halo_buffers() {
    int halo_width = domain_info_.halo_width;

    // X-direction halos (left/right)
    halo_size_x_ = halo_width * domain_info_.ny_local;
    CUDA_CHECK(cudaMalloc(&halo_send_left_, halo_size_x_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_send_right_, halo_size_x_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_recv_left_, halo_size_x_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_recv_right_, halo_size_x_ * sizeof(CellData)));

    // Y-direction halos (bottom/top)
    halo_size_y_ = halo_width * domain_info_.nx_local;
    CUDA_CHECK(cudaMalloc(&halo_send_bottom_, halo_size_y_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_send_top_, halo_size_y_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_recv_bottom_, halo_size_y_ * sizeof(CellData)));
    CUDA_CHECK(cudaMalloc(&halo_recv_top_, halo_size_y_ * sizeof(CellData)));
}

void MultiGPU::free_halo_buffers() {
    if (halo_send_left_) CUDA_CHECK(cudaFree(halo_send_left_));
    if (halo_send_right_) CUDA_CHECK(cudaFree(halo_send_right_));
    if (halo_send_bottom_) CUDA_CHECK(cudaFree(halo_send_bottom_));
    if (halo_send_top_) CUDA_CHECK(cudaFree(halo_send_top_));
    if (halo_recv_left_) CUDA_CHECK(cudaFree(halo_recv_left_));
    if (halo_recv_right_) CUDA_CHECK(cudaFree(halo_recv_right_));
    if (halo_recv_bottom_) CUDA_CHECK(cudaFree(halo_recv_bottom_));
    if (halo_recv_top_) CUDA_CHECK(cudaFree(halo_recv_top_));
}

void MultiGPU::exchange_halos(CellData* cells, cudaStream_t stream) {
    // Pack halos
    pack_halos(cells, stream);

    // Synchronize stream before MPI communication
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // X-direction exchange (left/right)
    MPI_Request requests[8];
    MPI_Status statuses[8];
    int req_count = 0;

    // Non-blocking send/recv for X-direction
    if (domain_info_.left_rank != MPI_PROC_NULL) {
        MPI_Isend(halo_send_left_, halo_size_x_, mpi_cell_type_,
                  domain_info_.left_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(halo_recv_left_, halo_size_x_, mpi_cell_type_,
                  domain_info_.left_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
    }

    if (domain_info_.right_rank != MPI_PROC_NULL) {
        MPI_Isend(halo_send_right_, halo_size_x_, mpi_cell_type_,
                  domain_info_.right_rank, 1, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(halo_recv_right_, halo_size_x_, mpi_cell_type_,
                  domain_info_.right_rank, 0, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Y-direction exchange (bottom/top)
    if (domain_info_.bottom_rank != MPI_PROC_NULL) {
        MPI_Isend(halo_send_bottom_, halo_size_y_, mpi_cell_type_,
                  domain_info_.bottom_rank, 2, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(halo_recv_bottom_, halo_size_y_, mpi_cell_type_,
                  domain_info_.bottom_rank, 3, MPI_COMM_WORLD, &requests[req_count++]);
    }

    if (domain_info_.top_rank != MPI_PROC_NULL) {
        MPI_Isend(halo_send_top_, halo_size_y_, mpi_cell_type_,
                  domain_info_.top_rank, 3, MPI_COMM_WORLD, &requests[req_count++]);
        MPI_Irecv(halo_recv_top_, halo_size_y_, mpi_cell_type_,
                  domain_info_.top_rank, 2, MPI_COMM_WORLD, &requests[req_count++]);
    }

    // Wait for all communication to complete
    MPI_Waitall(req_count, requests, statuses);

    // Unpack received halos
    unpack_halos(cells, stream);
}

void MultiGPU::pack_halos(const CellData* cells, cudaStream_t stream) {
    int nx = domain_info_.nx_local;
    int ny = domain_info_.ny_local;
    int halo_width = domain_info_.halo_width;

    dim3 block(16, 16);
    dim3 grid_x((halo_width + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y);
    dim3 grid_y((nx + block.x - 1) / block.x,
                (halo_width + block.y - 1) / block.y);

    // Pack left and right halos
    if (domain_info_.left_rank != MPI_PROC_NULL) {
        HaloKernels::pack_left_halo<<<grid_x, block, 0, stream>>>(
            cells, halo_send_left_, nx, ny, halo_width);
    }

    if (domain_info_.right_rank != MPI_PROC_NULL) {
        HaloKernels::pack_right_halo<<<grid_x, block, 0, stream>>>(
            cells, halo_send_right_, nx, ny, halo_width);
    }

    // Pack bottom and top halos
    if (domain_info_.bottom_rank != MPI_PROC_NULL) {
        HaloKernels::pack_bottom_halo<<<grid_y, block, 0, stream>>>(
            cells, halo_send_bottom_, nx, ny, halo_width);
    }

    if (domain_info_.top_rank != MPI_PROC_NULL) {
        HaloKernels::pack_top_halo<<<grid_y, block, 0, stream>>>(
            cells, halo_send_top_, nx, ny, halo_width);
    }
}

void MultiGPU::unpack_halos(CellData* cells, cudaStream_t stream) {
    int nx = domain_info_.nx_local;
    int ny = domain_info_.ny_local;
    int halo_width = domain_info_.halo_width;

    dim3 block(16, 16);
    dim3 grid_x((halo_width + block.x - 1) / block.x,
                (ny + block.y - 1) / block.y);
    dim3 grid_y((nx + block.x - 1) / block.x,
                (halo_width + block.y - 1) / block.y);

    if (domain_info_.left_rank != MPI_PROC_NULL) {
        HaloKernels::unpack_left_halo<<<grid_x, block, 0, stream>>>(
            cells, halo_recv_left_, nx, ny, halo_width);
    }

    if (domain_info_.right_rank != MPI_PROC_NULL) {
        HaloKernels::unpack_right_halo<<<grid_x, block, 0, stream>>>(
            cells, halo_recv_right_, nx, ny, halo_width);
    }

    if (domain_info_.bottom_rank != MPI_PROC_NULL) {
        HaloKernels::unpack_bottom_halo<<<grid_y, block, 0, stream>>>(
            cells, halo_recv_bottom_, nx, ny, halo_width);
    }

    if (domain_info_.top_rank != MPI_PROC_NULL) {
        HaloKernels::unpack_top_halo<<<grid_y, block, 0, stream>>>(
            cells, halo_recv_top_, nx, ny, halo_width);
    }
}

real_t MultiGPU::global_max(real_t local_value) {
    real_t global_value;
    if (sizeof(real_t) == 8) {
        MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    } else {
        MPI_Allreduce(&local_value, &global_value, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    }
    return global_value;
}

real_t MultiGPU::global_min(real_t local_value) {
    real_t global_value;
    if (sizeof(real_t) == 8) {
        MPI_Allreduce(&local_value, &global_value, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    } else {
        MPI_Allreduce(&local_value, &global_value, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    }
    return global_value;
}

void MultiGPU::barrier() {
    MPI_Barrier(MPI_COMM_WORLD);
}

void MultiGPU::finalize() {
    if (initialized_) {
        MPI_Finalize();
        initialized_ = false;
    }
}
