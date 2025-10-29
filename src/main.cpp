#include "hydrosis_solver.h"
#include "multi_gpu.h"
#include <iostream>
#include <cstring>

void print_usage() {
    std::cout << "Usage: hydrosis [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --multi-gpu       Enable multi-GPU mode (requires MPI)\n";
    std::cout << "  --nx <value>      Grid size in x-direction (default: 512)\n";
    std::cout << "  --ny <value>      Grid size in y-direction (default: 512)\n";
    std::cout << "  --cfl <value>     CFL number (default: 0.5)\n";
    std::cout << "  --tend <value>    End time in seconds (default: 10.0)\n";
    std::cout << "  --test <id>       Test case ID (default: 0)\n";
    std::cout << "                    0: 1D dam break\n";
    std::cout << "                    1: Circular dam break\n";
    std::cout << "  --help            Show this help message\n";
}

int main(int argc, char** argv) {
    // Parse command line arguments
    bool use_multi_gpu = false;
    int nx = 512;
    int ny = 512;
    real_t cfl = 0.5;
    real_t t_end = 10.0;
    int test_case = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--multi-gpu") == 0) {
            use_multi_gpu = true;
        } else if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            nx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ny") == 0 && i + 1 < argc) {
            ny = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfl") == 0 && i + 1 < argc) {
            cfl = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tend") == 0 && i + 1 < argc) {
            t_end = atof(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_case = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        }
    }

    // Initialize MPI if multi-GPU mode
    MultiGPU* multi_gpu = nullptr;
    if (use_multi_gpu) {
        multi_gpu = new MultiGPU();
        multi_gpu->initialize(&argc, &argv);

        if (multi_gpu->is_root()) {
            std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
            std::cout << "║         HydroSIS-2D: Multi-GPU 2D Hydrodynamic Model      ║\n";
            std::cout << "║                  CUDA + MPI Accelerated                    ║\n";
            std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
        }
    } else {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║         HydroSIS-2D: GPU-Accelerated 2D Hydrodynamic       ║\n";
        std::cout << "║                     Single GPU Mode                         ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    }

    // Setup simulation parameters
    SimParams params;

    // Grid parameters
    params.nx = nx;
    params.ny = ny;
    params.dx = 1.0;
    params.dy = 1.0;
    params.xmin = 0.0;
    params.ymin = 0.0;
    params.xmax = params.xmin + params.nx * params.dx;
    params.ymax = params.ymin + params.ny * params.dy;

    // Physical parameters
    params.g = Constants::GRAVITY;
    params.cfl = cfl;
    params.h_dry = Constants::DRY_TOLERANCE;
    params.friction_type = 0.0; // Manning

    // Time parameters
    params.t_start = 0.0;
    params.t_end = t_end;
    params.dt_max = 0.1;
    params.output_interval = 1.0;

    // Boundary conditions (all walls)
    params.bc_type[0] = 0; // left: wall
    params.bc_type[1] = 0; // right: wall
    params.bc_type[2] = 0; // bottom: wall
    params.bc_type[3] = 0; // top: wall

    // Solver options
    params.use_lts = false;
    params.riemann_solver = 1; // HLLC
    params.slope_limiter = 0;  // minmod
    params.order = 2;          // second-order MUSCL

    // Create and initialize solver
    HydroSisSolver solver;
    solver.initialize(params, use_multi_gpu);

    // Set initial conditions
    solver.set_initial_conditions(test_case);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    solver.run();
    auto end = std::chrono::high_resolution_clock::now();

    // Print final statistics
    solver.print_performance_stats();

    float total_time = std::chrono::duration<float>(end - start).count();

    if (!use_multi_gpu || multi_gpu->is_root()) {
        std::cout << "\n╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   Simulation Summary                       ║\n";
        std::cout << "╠════════════════════════════════════════════════════════════╣\n";

        printf("║ Total wall time:          %8.2f seconds                 ║\n", total_time);
        printf("║ Grid size:                %4d x %4d                    ║\n", nx, ny);
        printf("║ Total cells:              %10d                      ║\n", nx * ny);
        printf("║ Time steps:               %10d                      ║\n", solver.get_step_count());
        printf("║ Final time:               %8.2f s                      ║\n", solver.get_time());

        float cell_updates = (float)nx * ny * solver.get_step_count();
        float gigacells_per_sec = cell_updates / total_time / 1e9;
        printf("║ Performance:              %8.3f gigacells/s            ║\n", gigacells_per_sec);

        std::cout << "╚════════════════════════════════════════════════════════════╝\n\n";
    }

    // Cleanup
    if (multi_gpu) {
        multi_gpu->finalize();
        delete multi_gpu;
    }

    return 0;
}
