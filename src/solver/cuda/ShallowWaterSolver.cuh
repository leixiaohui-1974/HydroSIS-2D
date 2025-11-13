/**
 * @file ShallowWaterSolver.cuh
 * @brief CUDA-accelerated 2D shallow water equation solver
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 *
 * GPU-accelerated finite volume solver for 2D shallow water equations.
 * Supports structured and unstructured meshes with MUSCL reconstruction.
 */

#ifndef HYDROSIS2D_SHALLOW_WATER_SOLVER_CUH
#define HYDROSIS2D_SHALLOW_WATER_SOLVER_CUH

#include <cuda_runtime.h>
#include <string>
#include <vector>
#include <functional>

namespace hydrosis2d {
namespace cuda {

/**
 * @brief Mesh types supported by the solver
 */
enum class MeshType {
    STRUCTURED,      ///< Cartesian structured mesh
    UNSTRUCTURED     ///< Triangular/quad unstructured mesh
};

/**
 * @brief Riemann solver types
 */
enum class RiemannSolver {
    HLL,             ///< Harten-Lax-van Leer
    HLLC,            ///< HLL with Contact wave
    ROE              ///< Roe solver (approximate)
};

/**
 * @brief MUSCL limiter types for second-order reconstruction
 */
enum class Limiter {
    NONE,            ///< First-order (no limiting)
    MINMOD,          ///< Most conservative
    VANLEER,         ///< Smooth limiter
    SUPERBEE,        ///< Most aggressive
    MC               ///< Monotonized Central
};

/**
 * @brief Time integration schemes
 */
enum class TimeIntegrator {
    EULER,           ///< Explicit Euler (1st order)
    RK2,             ///< Runge-Kutta 2nd order
    RK3_TVD          ///< RK3 Total Variation Diminishing
};

/**
 * @brief Solver configuration
 */
struct SolverConfig {
    // Mesh parameters
    MeshType mesh_type = MeshType::STRUCTURED;
    int nx = 100;                    ///< Grid cells in x-direction
    int ny = 50;                     ///< Grid cells in y-direction
    double dx = 1.0;                 ///< Cell size in x (m)
    double dy = 1.0;                 ///< Cell size in y (m)

    // Numerical methods
    RiemannSolver riemann_solver = RiemannSolver::HLLC;
    Limiter limiter = Limiter::MINMOD;
    TimeIntegrator time_integrator = TimeIntegrator::RK2;
    int spatial_order = 2;           ///< 1 or 2 (MUSCL)

    // Physical parameters
    double gravity = 9.81;           ///< Gravitational acceleration (m/s²)
    double manning_n = 0.025;        ///< Manning's roughness coefficient
    double dry_threshold = 1e-6;     ///< Dry/wet threshold (m)

    // Time stepping
    double cfl = 0.8;                ///< CFL number
    double max_dt = 1e10;            ///< Maximum time step (s)
    double min_dt = 1e-10;           ///< Minimum time step (s)

    // Output
    double output_interval = 1.0;    ///< Output frequency (s)
    std::string output_dir = "output";
    std::string output_format = "vtk";
};

/**
 * @brief GPU device memory management
 */
struct DeviceMemory {
    // Conservative variables [nx * ny]
    double *d_h;      ///< Water depth (m)
    double *d_hu;     ///< x-momentum (m²/s)
    double *d_hv;     ///< y-momentum (m²/s)

    // Terrain
    double *d_z;      ///< Bed elevation (m)

    // Velocities (for convenience)
    double *d_u;      ///< x-velocity (m/s)
    double *d_v;      ///< y-velocity (m/s)

    // Fluxes [3 * (nx+1) * ny] for x-direction
    //        [3 * nx * (ny+1)] for y-direction
    double *d_flux_x;
    double *d_flux_y;

    // Source terms [nx * ny]
    double *d_source_h;
    double *d_source_hu;
    double *d_source_hv;

    // Time step (single value on device)
    double *d_dt;

    // Work arrays for RK schemes
    double *d_h_temp;
    double *d_hu_temp;
    double *d_hv_temp;

    size_t n_cells;

    DeviceMemory() : d_h(nullptr), d_hu(nullptr), d_hv(nullptr),
                     d_z(nullptr), d_u(nullptr), d_v(nullptr),
                     d_flux_x(nullptr), d_flux_y(nullptr),
                     d_source_h(nullptr), d_source_hu(nullptr), d_source_hv(nullptr),
                     d_dt(nullptr), d_h_temp(nullptr), d_hu_temp(nullptr), d_hv_temp(nullptr),
                     n_cells(0) {}

    ~DeviceMemory() {
        free();
    }

    void allocate(size_t nx, size_t ny);
    void free();
};

/**
 * @brief Progress callback function type
 *
 * Signature: void callback(double time, int step, double dt)
 */
using ProgressCallback = std::function<void(double, int, double)>;

/**
 * @brief CUDA-accelerated shallow water solver
 */
class ShallowWaterSolver {
public:
    ShallowWaterSolver();
    ~ShallowWaterSolver();

    /**
     * @brief Initialize solver from JSON configuration file
     * @param config_file Path to JSON configuration
     */
    void initialize(const std::string& config_file);

    /**
     * @brief Initialize solver from configuration struct
     * @param config Solver configuration
     */
    void initialize(const SolverConfig& config);

    /**
     * @brief Set initial conditions
     * @param h Water depth array [nx * ny]
     * @param u X-velocity array [nx * ny]
     * @param v Y-velocity array [nx * ny]
     * @param z Bed elevation array [nx * ny]
     */
    void setInitialConditions(const double* h, const double* u,
                             const double* v, const double* z);

    /**
     * @brief Run simulation until t_end
     * @param t_end End time (s)
     */
    void run(double t_end);

    /**
     * @brief Perform single time step with given dt
     * @param dt Time step size (s)
     */
    void step(double dt);

    /**
     * @brief Get current solution
     * @param h Output water depth [nx * ny]
     * @param u Output x-velocity [nx * ny]
     * @param v Output y-velocity [nx * ny]
     */
    void getSolution(double* h, double* u, double* v);

    /**
     * @brief Export current state to VTK file
     * @param filename Output filename
     */
    void exportVTK(const std::string& filename);

    /**
     * @brief Set progress callback function
     * @param callback Callback function
     * @param interval Call every N steps
     */
    void setCallback(ProgressCallback callback, int interval = 100);

    /**
     * @brief Get current simulation time
     */
    double getCurrentTime() const { return current_time_; }

    /**
     * @brief Get current step number
     */
    int getCurrentStep() const { return current_step_; }

    /**
     * @brief Get GPU memory usage (MB)
     */
    double getGPUMemoryUsage() const;

private:
    // Configuration
    SolverConfig config_;

    // Device memory
    DeviceMemory dev_mem_;

    // Host memory (for I/O)
    std::vector<double> h_host_;
    std::vector<double> u_host_;
    std::vector<double> v_host_;
    std::vector<double> z_host_;

    // Simulation state
    double current_time_;
    int current_step_;

    // Callback
    ProgressCallback callback_;
    int callback_interval_;

    // CUDA grid/block dimensions
    dim3 grid_;
    dim3 block_;

    // Private methods
    void setupCUDAGrid();
    void allocateMemory();
    void freeMemory();

    /**
     * @brief Compute adaptive time step based on CFL condition
     * @return Time step size (s)
     */
    double computeTimeStep();

    /**
     * @brief Apply boundary conditions
     */
    void applyBoundaryConditions();

    /**
     * @brief Compute fluxes at cell interfaces
     */
    void computeFluxes();

    /**
     * @brief Update conservative variables
     * @param dt Time step size
     */
    void updateConservativeVariables(double dt);

    /**
     * @brief Compute source terms (bed slope, friction)
     */
    void computeSourceTerms();

    /**
     * @brief Convert conservative to primitive variables
     */
    void conservativeToPrimitive();

    /**
     * @brief Check for NaN/Inf values (debugging)
     */
    bool checkValidity();

    /**
     * @brief Write output file
     */
    void writeOutput();
};

// Utility functions

/**
 * @brief Get number of CUDA devices
 */
int getGPUCount();

/**
 * @brief Set active CUDA device
 * @param device_id Device ID
 */
void setGPU(int device_id);

/**
 * @brief Get GPU memory info
 * @param free_mb Free memory (MB)
 * @param total_mb Total memory (MB)
 */
void getGPUMemory(double& free_mb, double& total_mb);

/**
 * @brief Check CUDA errors
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

} // namespace cuda
} // namespace hydrosis2d

#endif // HYDROSIS2D_SHALLOW_WATER_SOLVER_CUH
