#include "cuda_kernels.cuh"
#include <cmath>
#include <cfloat>

namespace CudaKernels {

// ============================================================================
// Device Helper Functions
// ============================================================================

__device__ inline real_t minmod(real_t a, real_t b) {
    if (a * b <= 0.0) return 0.0;
    return (fabs(a) < fabs(b)) ? a : b;
}

__device__ inline real_t superbee(real_t a, real_t b) {
    real_t s1 = minmod(2.0 * a, b);
    real_t s2 = minmod(a, 2.0 * b);
    return (fabs(s1) > fabs(s2)) ? s1 : s2;
}

__device__ inline real_t mc_limiter(real_t a, real_t b) {
    real_t c = 0.5 * (a + b);
    if (a * b <= 0.0) return 0.0;
    return minmod(c, minmod(2.0 * a, 2.0 * b));
}

__device__ inline PrimitiveVars conservative_to_primitive(
    const ConservativeVars& U) {
    PrimitiveVars W;
    W.h = U.h;
    if (U.h > Constants::MIN_DEPTH) {
        W.u = U.qx / U.h;
        W.v = U.qy / U.h;
    } else {
        W.u = 0.0;
        W.v = 0.0;
    }
    return W;
}

__device__ inline ConservativeVars primitive_to_conservative(
    const PrimitiveVars& W) {
    ConservativeVars U;
    U.h = W.h;
    U.qx = W.h * W.u;
    U.qy = W.h * W.v;
    return U;
}

// ============================================================================
// Flux Computation
// ============================================================================

__device__ FluxVector compute_flux_x(
    const ConservativeVars& U, real_t g) {
    FluxVector F;

    if (U.h < Constants::MIN_DEPTH) {
        F.f1 = 0.0;
        F.f2 = 0.0;
        F.f3 = 0.0;
        return F;
    }

    real_t u = U.qx / U.h;
    real_t v = U.qy / U.h;

    F.f1 = U.qx;                                    // h*u
    F.f2 = U.qx * u + 0.5 * g * U.h * U.h;         // h*u^2 + 0.5*g*h^2
    F.f3 = U.qx * v;                                // h*u*v

    return F;
}

__device__ FluxVector compute_flux_y(
    const ConservativeVars& U, real_t g) {
    FluxVector G;

    if (U.h < Constants::MIN_DEPTH) {
        G.f1 = 0.0;
        G.f2 = 0.0;
        G.f3 = 0.0;
        return G;
    }

    real_t u = U.qx / U.h;
    real_t v = U.qy / U.h;

    G.f1 = U.qy;                                    // h*v
    G.f2 = U.qy * u;                                // h*v*u
    G.f3 = U.qy * v + 0.5 * g * U.h * U.h;         // h*v^2 + 0.5*g*h^2

    return G;
}

// ============================================================================
// HLLC Riemann Solver (Most Accurate)
// ============================================================================

__device__ FluxVector hllc_riemann_solver(
    const ConservativeVars& U_L,
    const ConservativeVars& U_R,
    real_t z_L, real_t z_R,
    real_t g, int dir) {

    FluxVector F;
    F.f1 = 0.0; F.f2 = 0.0; F.f3 = 0.0;

    // Dry cell treatment
    bool dry_L = (U_L.h < Constants::MIN_DEPTH);
    bool dry_R = (U_R.h < Constants::MIN_DEPTH);

    if (dry_L && dry_R) return F;

    if (dry_L) {
        return (dir == 0) ? compute_flux_x(U_R, g) : compute_flux_y(U_R, g);
    }
    if (dry_R) {
        return (dir == 0) ? compute_flux_x(U_L, g) : compute_flux_y(U_L, g);
    }

    // Primitive variables
    PrimitiveVars W_L = conservative_to_primitive(U_L);
    PrimitiveVars W_R = conservative_to_primitive(U_R);

    // Wave speeds
    real_t c_L = sqrt(g * W_L.h);  // Celerity left
    real_t c_R = sqrt(g * W_R.h);  // Celerity right

    real_t u_L = (dir == 0) ? W_L.u : W_L.v;
    real_t u_R = (dir == 0) ? W_R.u : W_R.v;

    // Roe averages for wave speed estimates
    real_t h_roe = 0.5 * (W_L.h + W_R.h);
    real_t u_roe = (sqrt(W_L.h) * u_L + sqrt(W_R.h) * u_R) /
                   (sqrt(W_L.h) + sqrt(W_R.h));
    real_t c_roe = sqrt(g * h_roe);

    // Wave speeds (Einfeldt)
    real_t S_L = min(u_L - c_L, u_roe - c_roe);
    real_t S_R = max(u_R + c_R, u_roe + c_roe);

    // Middle wave speed (HLLC)
    real_t S_star = (S_R * u_R - S_L * u_L +
                     0.5 * g * (W_L.h * W_L.h - W_R.h * W_R.h)) /
                    (S_R - S_L + REAL_EPSILON);

    // Compute fluxes
    FluxVector F_L = (dir == 0) ? compute_flux_x(U_L, g) : compute_flux_y(U_L, g);
    FluxVector F_R = (dir == 0) ? compute_flux_x(U_R, g) : compute_flux_y(U_R, g);

    // HLLC flux selection
    if (S_L >= 0.0) {
        F = F_L;
    } else if (S_R <= 0.0) {
        F = F_R;
    } else if (S_star >= 0.0) {
        // Star region, left side
        real_t factor = (S_L - u_L) / (S_L - S_star);
        ConservativeVars U_star_L;
        U_star_L.h = U_L.h * factor;

        if (dir == 0) {
            U_star_L.qx = U_star_L.h * S_star;
            U_star_L.qy = U_L.qy * factor;
        } else {
            U_star_L.qx = U_L.qx * factor;
            U_star_L.qy = U_star_L.h * S_star;
        }

        F.f1 = F_L.f1 + S_L * (U_star_L.h - U_L.h);
        F.f2 = F_L.f2 + S_L * (U_star_L.qx - U_L.qx);
        F.f3 = F_L.f3 + S_L * (U_star_L.qy - U_L.qy);
    } else {
        // Star region, right side
        real_t factor = (S_R - u_R) / (S_R - S_star);
        ConservativeVars U_star_R;
        U_star_R.h = U_R.h * factor;

        if (dir == 0) {
            U_star_R.qx = U_star_R.h * S_star;
            U_star_R.qy = U_R.qy * factor;
        } else {
            U_star_R.qx = U_R.qx * factor;
            U_star_R.qy = U_star_R.h * S_star;
        }

        F.f1 = F_R.f1 + S_R * (U_star_R.h - U_R.h);
        F.f2 = F_R.f2 + S_R * (U_star_R.qx - U_R.qx);
        F.f3 = F_R.f3 + S_R * (U_star_R.qy - U_R.qy);
    }

    return F;
}

// ============================================================================
// MUSCL Reconstruction
// ============================================================================

__device__ void muscl_reconstruction(
    const ConservativeVars& U_m,
    const ConservativeVars& U_c,
    const ConservativeVars& U_p,
    ConservativeVars& U_L,
    ConservativeVars& U_R,
    int limiter) {

    // Compute slopes
    real_t slope_h_L, slope_h_R;
    real_t slope_qx_L, slope_qx_R;
    real_t slope_qy_L, slope_qy_R;

    if (limiter == 0) { // Minmod
        slope_h_L = minmod(U_c.h - U_m.h, U_p.h - U_c.h);
        slope_qx_L = minmod(U_c.qx - U_m.qx, U_p.qx - U_c.qx);
        slope_qy_L = minmod(U_c.qy - U_m.qy, U_p.qy - U_c.qy);
    } else if (limiter == 1) { // Superbee
        slope_h_L = superbee(U_c.h - U_m.h, U_p.h - U_c.h);
        slope_qx_L = superbee(U_c.qx - U_m.qx, U_p.qx - U_c.qx);
        slope_qy_L = superbee(U_c.qy - U_m.qy, U_p.qy - U_c.qy);
    } else { // MC limiter
        slope_h_L = mc_limiter(U_c.h - U_m.h, U_p.h - U_c.h);
        slope_qx_L = mc_limiter(U_c.qx - U_m.qx, U_p.qx - U_c.qx);
        slope_qy_L = mc_limiter(U_c.qy - U_m.qy, U_p.qy - U_c.qy);
    }

    // Reconstruct left and right states
    U_L.h = U_c.h - 0.5 * slope_h_L;
    U_L.qx = U_c.qx - 0.5 * slope_qx_L;
    U_L.qy = U_c.qy - 0.5 * slope_qy_L;

    U_R.h = U_c.h + 0.5 * slope_h_L;
    U_R.qx = U_c.qx + 0.5 * slope_qx_L;
    U_R.qy = U_c.qy + 0.5 * slope_qy_L;

    // Ensure non-negative depth
    U_L.h = max(U_L.h, (real_t)0.0);
    U_R.h = max(U_R.h, (real_t)0.0);
}

// ============================================================================
// Main Update Kernel
// ============================================================================

__global__ void update_cells_kernel(
    const CellData* cells,
    CellData* cells_new,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t dt,
    const SimParams params) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + Constants::HALO_WIDTH;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Constants::HALO_WIDTH;

    if (i >= nx - Constants::HALO_WIDTH || j >= ny - Constants::HALO_WIDTH) return;

    int idx = j * nx + i;

    // Get current cell
    const CellData& cell = cells[idx];
    ConservativeVars U_new = cell.U;

    // X-direction fluxes
    if (params.order == 2) {
        // Second-order MUSCL
        ConservativeVars U_L, U_R;

        // Left interface (i-1/2)
        muscl_reconstruction(
            cells[idx - 2].U, cells[idx - 1].U, cells[idx].U,
            U_L, U_R, params.slope_limiter);
        FluxVector F_left = hllc_riemann_solver(
            U_R, U_L, cells[idx - 1].z, cells[idx].z, params.g, 0);

        // Right interface (i+1/2)
        muscl_reconstruction(
            cells[idx - 1].U, cells[idx].U, cells[idx + 1].U,
            U_L, U_R, params.slope_limiter);
        FluxVector F_right = hllc_riemann_solver(
            U_R, U_L, cells[idx].z, cells[idx + 1].z, params.g, 0);

        // Update with x-fluxes
        U_new.h -= (dt / dx) * (F_right.f1 - F_left.f1);
        U_new.qx -= (dt / dx) * (F_right.f2 - F_left.f2);
        U_new.qy -= (dt / dx) * (F_right.f3 - F_left.f3);
    } else {
        // First-order
        FluxVector F_left = hllc_riemann_solver(
            cells[idx - 1].U, cells[idx].U,
            cells[idx - 1].z, cells[idx].z, params.g, 0);
        FluxVector F_right = hllc_riemann_solver(
            cells[idx].U, cells[idx + 1].U,
            cells[idx].z, cells[idx + 1].z, params.g, 0);

        U_new.h -= (dt / dx) * (F_right.f1 - F_left.f1);
        U_new.qx -= (dt / dx) * (F_right.f2 - F_left.f2);
        U_new.qy -= (dt / dx) * (F_right.f3 - F_left.f3);
    }

    // Y-direction fluxes
    if (params.order == 2) {
        ConservativeVars U_L, U_R;

        // Bottom interface (j-1/2)
        muscl_reconstruction(
            cells[(j - 2) * nx + i].U,
            cells[(j - 1) * nx + i].U,
            cells[idx].U,
            U_L, U_R, params.slope_limiter);
        FluxVector G_bottom = hllc_riemann_solver(
            U_R, U_L,
            cells[(j - 1) * nx + i].z, cells[idx].z, params.g, 1);

        // Top interface (j+1/2)
        muscl_reconstruction(
            cells[(j - 1) * nx + i].U,
            cells[idx].U,
            cells[(j + 1) * nx + i].U,
            U_L, U_R, params.slope_limiter);
        FluxVector G_top = hllc_riemann_solver(
            U_R, U_L,
            cells[idx].z, cells[(j + 1) * nx + i].z, params.g, 1);

        U_new.h -= (dt / dy) * (G_top.f1 - G_bottom.f1);
        U_new.qx -= (dt / dy) * (G_top.f2 - G_bottom.f2);
        U_new.qy -= (dt / dy) * (G_top.f3 - G_bottom.f3);
    } else {
        FluxVector G_bottom = hllc_riemann_solver(
            cells[(j - 1) * nx + i].U, cells[idx].U,
            cells[(j - 1) * nx + i].z, cells[idx].z, params.g, 1);
        FluxVector G_top = hllc_riemann_solver(
            cells[idx].U, cells[(j + 1) * nx + i].U,
            cells[idx].z, cells[(j + 1) * nx + i].z, params.g, 1);

        U_new.h -= (dt / dy) * (G_top.f1 - G_bottom.f1);
        U_new.qx -= (dt / dy) * (G_top.f2 - G_bottom.f2);
        U_new.qy -= (dt / dy) * (G_top.f3 - G_bottom.f3);
    }

    // Store updated state
    cells_new[idx].U = U_new;
    cells_new[idx].z = cell.z;
    cells_new[idx].n = cell.n;
}

// ============================================================================
// Source Terms (Bed Slope + Friction)
// ============================================================================

__device__ void apply_manning_friction(
    ConservativeVars& U,
    real_t n,
    real_t dt,
    real_t g) {

    if (U.h < Constants::MIN_DEPTH) {
        U.qx = 0.0;
        U.qy = 0.0;
        return;
    }

    real_t u = U.qx / U.h;
    real_t v = U.qy / U.h;
    real_t vel_mag = sqrt(u * u + v * v);

    if (vel_mag < REAL_EPSILON) return;

    // Manning's friction: Sf = n^2 * |V| * V / h^(4/3)
    real_t h_43 = pow(U.h, 4.0 / 3.0);
    real_t cf = g * n * n / h_43;
    real_t factor = 1.0 / (1.0 + dt * cf * vel_mag);

    U.qx *= factor;
    U.qy *= factor;
}

__global__ void apply_source_terms_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t dt,
    real_t g) {

    int i = blockIdx.x * blockDim.x + threadIdx.x + Constants::HALO_WIDTH;
    int j = blockIdx.y * blockDim.y + threadIdx.y + Constants::HALO_WIDTH;

    if (i >= nx - Constants::HALO_WIDTH || j >= ny - Constants::HALO_WIDTH) return;

    int idx = j * nx + i;
    CellData& cell = cells[idx];

    // Bed slope source term
    real_t dz_dx = (cells[idx + 1].z - cells[idx - 1].z) / (2.0 * dx);
    real_t dz_dy = (cells[(j + 1) * nx + i].z - cells[(j - 1) * nx + i].z) / (2.0 * dy);

    cell.U.qx -= dt * g * cell.U.h * dz_dx;
    cell.U.qy -= dt * g * cell.U.h * dz_dy;

    // Manning's friction
    apply_manning_friction(cell.U, cell.n, dt, g);

    // Ensure non-negative depth
    cell.U.h = max(cell.U.h, (real_t)0.0);
}

// ============================================================================
// Time Step Computation
// ============================================================================

__global__ void compute_max_wavespeed_kernel(
    const CellData* cells,
    int nx, int ny,
    real_t g,
    real_t* max_speed) {

    __shared__ real_t sdata[256];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    real_t local_max = 0.0;

    if (i < nx && j < ny) {
        int idx = j * nx + i;
        const CellData& cell = cells[idx];

        if (cell.U.h > Constants::MIN_DEPTH) {
            real_t u = cell.U.qx / cell.U.h;
            real_t v = cell.U.qy / cell.U.h;
            real_t c = sqrt(g * cell.U.h);
            local_max = sqrt(u * u + v * v) + c;
        }
    }

    sdata[tid] = local_max;
    __syncthreads();

    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = max(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicMax((int*)max_speed, __float_as_int(sdata[0]));
    }
}

__global__ void compute_timestep_kernel(
    const CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t cfl, real_t g,
    real_t* dt_global) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    const CellData& cell = cells[idx];

    if (cell.U.h > Constants::MIN_DEPTH) {
        real_t u = cell.U.qx / cell.U.h;
        real_t v = cell.U.qy / cell.U.h;
        real_t c = sqrt(g * cell.U.h);

        real_t dt_x = dx / (fabs(u) + c + REAL_EPSILON);
        real_t dt_y = dy / (fabs(v) + c + REAL_EPSILON);
        real_t dt_local = cfl * min(dt_x, dt_y);

        atomicMin((int*)dt_global, __float_as_int(dt_local));
    }
}

// ============================================================================
// Boundary Conditions
// ============================================================================

__global__ void apply_boundary_conditions_kernel(
    CellData* cells,
    int nx, int ny,
    const int* bc_types) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ========================================================================
    // Left boundary (i = 0, 1)
    // ========================================================================
    if (idx < ny) {
        int bc_type = bc_types[0];

        if (bc_type == 0) {
            // Wall boundary (reflective)
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = halo;
                int i_ghost = 2 * Constants::HALO_WIDTH - 1 - i;
                cells[idx * nx + i].U.h = cells[idx * nx + i_ghost].U.h;
                cells[idx * nx + i].U.qx = -cells[idx * nx + i_ghost].U.qx; // Reflect x-velocity
                cells[idx * nx + i].U.qy = cells[idx * nx + i_ghost].U.qy;
            }
        } else if (bc_type == 1) {
            // Open boundary (zero-gradient extrapolation)
            int i_interior = Constants::HALO_WIDTH;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = halo;
                cells[idx * nx + i].U.h = cells[idx * nx + i_interior].U.h;
                cells[idx * nx + i].U.qx = cells[idx * nx + i_interior].U.qx;
                cells[idx * nx + i].U.qy = cells[idx * nx + i_interior].U.qy;
            }
        } else if (bc_type == 2) {
            // Inflow boundary (fixed state)
            // TODO: Get inflow values from params (for now, use simple approach)
            real_t h_inflow = 5.0;  // Will be configurable
            real_t u_inflow = 1.0;
            real_t v_inflow = 0.0;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = halo;
                cells[idx * nx + i].U.h = h_inflow;
                cells[idx * nx + i].U.qx = h_inflow * u_inflow;
                cells[idx * nx + i].U.qy = h_inflow * v_inflow;
            }
        } else if (bc_type == 3) {
            // Outflow boundary (radiation/advective)
            int i_interior = Constants::HALO_WIDTH;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = halo;
                // Simple zero-gradient (can be improved with wave speed)
                cells[idx * nx + i].U.h = cells[idx * nx + i_interior].U.h;
                cells[idx * nx + i].U.qx = cells[idx * nx + i_interior].U.qx;
                cells[idx * nx + i].U.qy = cells[idx * nx + i_interior].U.qy;
            }
        }
    }

    // ========================================================================
    // Right boundary
    // ========================================================================
    if (idx < ny) {
        int bc_type = bc_types[1];

        if (bc_type == 0) {
            // Wall boundary
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = nx - 1 - halo;
                int i_ghost = 2 * (nx - Constants::HALO_WIDTH) - 1 - i;
                cells[idx * nx + i].U.h = cells[idx * nx + i_ghost].U.h;
                cells[idx * nx + i].U.qx = -cells[idx * nx + i_ghost].U.qx;
                cells[idx * nx + i].U.qy = cells[idx * nx + i_ghost].U.qy;
            }
        } else if (bc_type == 1) {
            // Open boundary
            int i_interior = nx - Constants::HALO_WIDTH - 1;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = nx - 1 - halo;
                cells[idx * nx + i].U.h = cells[idx * nx + i_interior].U.h;
                cells[idx * nx + i].U.qx = cells[idx * nx + i_interior].U.qx;
                cells[idx * nx + i].U.qy = cells[idx * nx + i_interior].U.qy;
            }
        } else if (bc_type == 2) {
            // Inflow boundary
            real_t h_inflow = 5.0;
            real_t u_inflow = -1.0;  // Negative for right boundary
            real_t v_inflow = 0.0;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = nx - 1 - halo;
                cells[idx * nx + i].U.h = h_inflow;
                cells[idx * nx + i].U.qx = h_inflow * u_inflow;
                cells[idx * nx + i].U.qy = h_inflow * v_inflow;
            }
        } else if (bc_type == 3) {
            // Outflow boundary
            int i_interior = nx - Constants::HALO_WIDTH - 1;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int i = nx - 1 - halo;
                cells[idx * nx + i].U.h = cells[idx * nx + i_interior].U.h;
                cells[idx * nx + i].U.qx = cells[idx * nx + i_interior].U.qx;
                cells[idx * nx + i].U.qy = cells[idx * nx + i_interior].U.qy;
            }
        }
    }

    // ========================================================================
    // Bottom boundary
    // ========================================================================
    if (idx < nx) {
        int bc_type = bc_types[2];

        if (bc_type == 0) {
            // Wall boundary
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = halo;
                int j_ghost = 2 * Constants::HALO_WIDTH - 1 - j;
                cells[j * nx + idx].U.h = cells[j_ghost * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_ghost * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = -cells[j_ghost * nx + idx].U.qy;
            }
        } else if (bc_type == 1) {
            // Open boundary
            int j_interior = Constants::HALO_WIDTH;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = halo;
                cells[j * nx + idx].U.h = cells[j_interior * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_interior * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = cells[j_interior * nx + idx].U.qy;
            }
        } else if (bc_type == 2) {
            // Inflow boundary
            real_t h_inflow = 5.0;
            real_t u_inflow = 0.0;
            real_t v_inflow = 1.0;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = halo;
                cells[j * nx + idx].U.h = h_inflow;
                cells[j * nx + idx].U.qx = h_inflow * u_inflow;
                cells[j * nx + idx].U.qy = h_inflow * v_inflow;
            }
        } else if (bc_type == 3) {
            // Outflow boundary
            int j_interior = Constants::HALO_WIDTH;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = halo;
                cells[j * nx + idx].U.h = cells[j_interior * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_interior * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = cells[j_interior * nx + idx].U.qy;
            }
        }
    }

    // ========================================================================
    // Top boundary
    // ========================================================================
    if (idx < nx) {
        int bc_type = bc_types[3];

        if (bc_type == 0) {
            // Wall boundary
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = ny - 1 - halo;
                int j_ghost = 2 * (ny - Constants::HALO_WIDTH) - 1 - j;
                cells[j * nx + idx].U.h = cells[j_ghost * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_ghost * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = -cells[j_ghost * nx + idx].U.qy;
            }
        } else if (bc_type == 1) {
            // Open boundary
            int j_interior = ny - Constants::HALO_WIDTH - 1;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = ny - 1 - halo;
                cells[j * nx + idx].U.h = cells[j_interior * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_interior * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = cells[j_interior * nx + idx].U.qy;
            }
        } else if (bc_type == 2) {
            // Inflow boundary
            real_t h_inflow = 5.0;
            real_t u_inflow = 0.0;
            real_t v_inflow = -1.0;  // Negative for top boundary
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = ny - 1 - halo;
                cells[j * nx + idx].U.h = h_inflow;
                cells[j * nx + idx].U.qx = h_inflow * u_inflow;
                cells[j * nx + idx].U.qy = h_inflow * v_inflow;
            }
        } else if (bc_type == 3) {
            // Outflow boundary
            int j_interior = ny - Constants::HALO_WIDTH - 1;
            for (int halo = 0; halo < Constants::HALO_WIDTH; halo++) {
                int j = ny - 1 - halo;
                cells[j * nx + idx].U.h = cells[j_interior * nx + idx].U.h;
                cells[j * nx + idx].U.qx = cells[j_interior * nx + idx].U.qx;
                cells[j * nx + idx].U.qy = cells[j_interior * nx + idx].U.qy;
            }
        }
    }
}

// ============================================================================
// Wet/Dry Treatment
// ============================================================================

__global__ void wet_dry_treatment_kernel(
    CellData* cells,
    int nx, int ny,
    real_t h_dry) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    CellData& cell = cells[idx];

    if (cell.U.h < h_dry) {
        cell.U.h = 0.0;
        cell.U.qx = 0.0;
        cell.U.qy = 0.0;
        cell.is_wet = false;
    } else {
        cell.is_wet = true;
    }
}

// ============================================================================
// Initialization
// ============================================================================

__global__ void initialize_grid_kernel(
    CellData* cells,
    int nx, int ny,
    real_t dx, real_t dy,
    real_t xmin, real_t ymin,
    int test_case) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= nx || j >= ny) return;

    int idx = j * nx + i;
    real_t x = xmin + i * dx;
    real_t y = ymin + j * dy;
    real_t xc = xmin + 0.5 * nx * dx;
    real_t yc = ymin + 0.5 * ny * dy;

    CellData& cell = cells[idx];
    cell.n = 0.03; // Default Manning's n

    switch (test_case) {
        case 0: { // 1D Dam Break
            cell.z = 0.0;
            cell.n = 0.0;
            if (x < xc) {
                cell.U.h = 10.0;
            } else {
                cell.U.h = 1.0;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 1: { // 2D Circular Dam
            real_t r = sqrt((x - xc) * (x - xc) + (y - yc) * (y - yc));
            real_t R_dam = 0.15 * nx * dx;
            cell.z = 0.0;
            cell.n = 0.0;
            if (r < R_dam) {
                cell.U.h = 10.0;
            } else {
                cell.U.h = 1.0;
            }
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        case 5: { // Lake at Rest
            real_t eta = 1.0;
            cell.z = 0.2 * sin(2.0 * M_PI * x / (nx * dx)) *
                    cos(2.0 * M_PI * y / (ny * dy));
            cell.n = 0.03;
            cell.U.h = eta - cell.z;
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
        }

        default:
            // Default: simple dam break
            cell.z = 0.0;
            cell.n = 0.0;
            cell.U.h = (x < xc) ? 2.0 : 1.0;
            cell.U.qx = 0.0;
            cell.U.qy = 0.0;
            break;
    }

    cell.is_wet = (cell.U.h > Constants::DRY_TOLERANCE);
}

__global__ void copy_cells_kernel(
    const CellData* src,
    CellData* dst,
    int n) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

} // namespace CudaKernels
