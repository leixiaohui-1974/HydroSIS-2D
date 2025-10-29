#ifndef HYDROSIS_TYPES_H
#define HYDROSIS_TYPES_H

#include <cuda_runtime.h>

// Precision type - can switch between float and double
#ifdef USE_DOUBLE_PRECISION
typedef double real_t;
#define REAL_EPSILON 1e-12
#else
typedef float real_t;
#define REAL_EPSILON 1e-6f
#endif

// Conservative variables for 2D Shallow Water Equations
struct ConservativeVars {
    real_t h;      // Water depth
    real_t qx;     // Discharge in x-direction (h*u)
    real_t qy;     // Discharge in y-direction (h*v)

    __host__ __device__ ConservativeVars() : h(0), qx(0), qy(0) {}
    __host__ __device__ ConservativeVars(real_t h_, real_t qx_, real_t qy_)
        : h(h_), qx(qx_), qy(qy_) {}
};

// Primitive variables
struct PrimitiveVars {
    real_t h;      // Water depth
    real_t u;      // Velocity in x-direction
    real_t v;      // Velocity in y-direction

    __host__ __device__ PrimitiveVars() : h(0), u(0), v(0) {}
    __host__ __device__ PrimitiveVars(real_t h_, real_t u_, real_t v_)
        : h(h_), u(u_), v(v_) {}
};

// Flux vector
struct FluxVector {
    real_t f1, f2, f3;

    __host__ __device__ FluxVector() : f1(0), f2(0), f3(0) {}
    __host__ __device__ FluxVector(real_t f1_, real_t f2_, real_t f3_)
        : f1(f1_), f2(f2_), f3(f3_) {}
};

// Grid cell properties
struct CellData {
    ConservativeVars U;     // Conservative variables
    real_t z;               // Bed elevation
    real_t n;               // Manning's roughness coefficient
    real_t dt_local;        // Local time step for LTS
    bool is_wet;            // Wet/dry flag

    __host__ __device__ CellData()
        : U(), z(0), n(0.03f), dt_local(0), is_wet(false) {}
};

// Simulation parameters
struct SimParams {
    // Grid parameters
    int nx, ny;             // Grid dimensions
    real_t dx, dy;          // Grid spacing
    real_t xmin, ymin;      // Domain origin
    real_t xmax, ymax;      // Domain extent

    // Physical parameters
    real_t g;               // Gravitational acceleration (9.81 m/s^2)
    real_t cfl;             // CFL number (0.5-0.9)
    real_t h_dry;           // Dry threshold (1e-4 m)
    real_t friction_type;   // 0: Manning, 1: Chezy

    // Time parameters
    real_t t_start;         // Start time
    real_t t_end;           // End time
    real_t dt_max;          // Maximum time step
    real_t output_interval; // Output interval

    // Boundary conditions
    int bc_type[4];         // 0: wall, 1: open, 2: inflow, 3: outflow
                            // Order: [left, right, bottom, top]

    // Solver options
    bool use_lts;           // Enable Local Time Stepping
    int riemann_solver;     // 0: HLL, 1: HLLC, 2: Roe
    int slope_limiter;      // 0: minmod, 1: superbee, 2: MC
    int order;              // 1: first-order, 2: second-order (MUSCL)
};

// Domain decomposition info for multi-GPU
struct DomainInfo {
    int rank;               // MPI rank
    int num_procs;          // Total number of processes
    int gpu_id;             // GPU device ID

    // Local domain
    int nx_local, ny_local; // Local grid dimensions
    int i_start, j_start;   // Global starting indices
    int i_end, j_end;       // Global ending indices

    // Neighbor ranks
    int left_rank, right_rank;
    int bottom_rank, top_rank;

    // Halo/ghost cells
    int halo_width;         // Typically 2 for second-order
};

// Constants
namespace Constants {
    constexpr real_t GRAVITY = 9.81;
    constexpr real_t MIN_DEPTH = 1e-6;
    constexpr real_t DRY_TOLERANCE = 1e-4;
    constexpr int HALO_WIDTH = 2;
}

#endif // HYDROSIS_TYPES_H
