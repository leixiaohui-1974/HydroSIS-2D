#include "multi_gpu.h"

namespace HaloKernels {

// ============================================================================
// Pack Kernels
// ============================================================================

__global__ void pack_left_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int hx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (hx >= halo_width || j >= ny) return;

    // Pack from i = halo_width + hx
    int i = halo_width + hx;
    int idx_src = j * nx + i;
    int idx_dst = j * halo_width + hx;

    halo_buffer[idx_dst] = cells[idx_src];
}

__global__ void pack_right_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int hx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (hx >= halo_width || j >= ny) return;

    // Pack from i = nx - 2*halo_width + hx
    int i = nx - 2 * halo_width + hx;
    int idx_src = j * nx + i;
    int idx_dst = j * halo_width + hx;

    halo_buffer[idx_dst] = cells[idx_src];
}

__global__ void pack_bottom_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hy = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || hy >= halo_width) return;

    // Pack from j = halo_width + hy
    int j = halo_width + hy;
    int idx_src = j * nx + i;
    int idx_dst = hy * nx + i;

    halo_buffer[idx_dst] = cells[idx_src];
}

__global__ void pack_top_halo(
    const CellData* cells,
    CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hy = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || hy >= halo_width) return;

    // Pack from j = ny - 2*halo_width + hy
    int j = ny - 2 * halo_width + hy;
    int idx_src = j * nx + i;
    int idx_dst = hy * nx + i;

    halo_buffer[idx_dst] = cells[idx_src];
}

// ============================================================================
// Unpack Kernels
// ============================================================================

__global__ void unpack_left_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int hx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (hx >= halo_width || j >= ny) return;

    // Unpack to i = hx (ghost cells on the left)
    int i = hx;
    int idx_dst = j * nx + i;
    int idx_src = j * halo_width + hx;

    cells[idx_dst] = halo_buffer[idx_src];
}

__global__ void unpack_right_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int hx = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (hx >= halo_width || j >= ny) return;

    // Unpack to i = nx - halo_width + hx (ghost cells on the right)
    int i = nx - halo_width + hx;
    int idx_dst = j * nx + i;
    int idx_src = j * halo_width + hx;

    cells[idx_dst] = halo_buffer[idx_src];
}

__global__ void unpack_bottom_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hy = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || hy >= halo_width) return;

    // Unpack to j = hy (ghost cells on the bottom)
    int j = hy;
    int idx_dst = j * nx + i;
    int idx_src = hy * nx + i;

    cells[idx_dst] = halo_buffer[idx_src];
}

__global__ void unpack_top_halo(
    CellData* cells,
    const CellData* halo_buffer,
    int nx, int ny,
    int halo_width) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int hy = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || hy >= halo_width) return;

    // Unpack to j = ny - halo_width + hy (ghost cells on the top)
    int j = ny - halo_width + hy;
    int idx_dst = j * nx + i;
    int idx_src = hy * nx + i;

    cells[idx_dst] = halo_buffer[idx_src];
}

} // namespace HaloKernels
