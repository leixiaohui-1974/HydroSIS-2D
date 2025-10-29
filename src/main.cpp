#include "hydrosis_solver.h"
#include "multi_gpu.h"
#include "test_cases.h"
#include "validation.h"
#include "config_reader.h"
#include <iostream>
#include <cstring>

void print_usage() {
    std::cout << "Usage: hydrosis [options]\n";
    std::cout << "Options:\n";
    std::cout << "  --config <file>   Load configuration from file\n";
    std::cout << "  --multi-gpu       Enable multi-GPU mode (requires MPI)\n";
    std::cout << "  --nx <value>      Grid size in x-direction (overrides config)\n";
    std::cout << "  --ny <value>      Grid size in y-direction (overrides config)\n";
    std::cout << "  --cfl <value>     CFL number (overrides config)\n";
    std::cout << "  --tend <value>    End time in seconds (overrides config)\n";
    std::cout << "  --test <id>       Test case ID (default: 0)\n";
    std::cout << "                    0: 1D Dam Break (Ritter)\n";
    std::cout << "                    1: 2D Circular Dam Break\n";
    std::cout << "                    2: Partial Dam Break\n";
    std::cout << "                    3: Thacker's Planar Beach\n";
    std::cout << "                    4: MacDonald Wetting/Drying\n";
    std::cout << "                    5: Lake at Rest\n";
    std::cout << "                    6: Small Perturbation\n";
    std::cout << "                    7: Flow Over Bump\n";
    std::cout << "                    8: Oblique Hydraulic Jump\n";
    std::cout << "  --validate        Enable validation and error analysis\n";
    std::cout << "  --vtk             Enable VTK output for visualization\n";
    std::cout << "  --help            Show this help message\n";
    std::cout << "\nNote: Command line options override config file values.\n";
    std::cout << "Example: ./hydrosis --config examples/config_dam_break.ini\n";
}

int main(int argc, char** argv) {
    // Parse command line arguments (first pass - look for config file)
    std::string config_file;
    bool use_multi_gpu = false;
    bool enable_validation = false;
    bool enable_vtk = false;
    int test_case = 0;

    // Command line overrides
    int nx_override = -1;
    int ny_override = -1;
    real_t cfl_override = -1.0;
    real_t t_end_override = -1.0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--config") == 0 && i + 1 < argc) {
            config_file = argv[++i];
        } else if (strcmp(argv[i], "--multi-gpu") == 0) {
            use_multi_gpu = true;
        } else if (strcmp(argv[i], "--nx") == 0 && i + 1 < argc) {
            nx_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--ny") == 0 && i + 1 < argc) {
            ny_override = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--cfl") == 0 && i + 1 < argc) {
            cfl_override = atof(argv[++i]);
        } else if (strcmp(argv[i], "--tend") == 0 && i + 1 < argc) {
            t_end_override = atof(argv[++i]);
        } else if (strcmp(argv[i], "--test") == 0 && i + 1 < argc) {
            test_case = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--validate") == 0) {
            enable_validation = true;
        } else if (strcmp(argv[i], "--vtk") == 0) {
            enable_vtk = true;
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

    // Load from config file if provided
    if (!config_file.empty()) {
        if (!use_multi_gpu || multi_gpu->is_root()) {
            std::cout << "Loading configuration from: " << config_file << std::endl;
        }

        ConfigReader config;
        if (!config.load(config_file)) {
            std::cerr << "Error: Failed to load configuration file" << std::endl;
            if (multi_gpu) {
                multi_gpu->finalize();
                delete multi_gpu;
            }
            return 1;
        }

        // Parse parameters from config
        config.parse_params(params);

        if (!use_multi_gpu || multi_gpu->is_root()) {
            std::cout << "Configuration loaded successfully.\n" << std::endl;
        }
    } else {
        // Use default parameters if no config file
        params.nx = 512;
        params.ny = 512;
        params.dx = 1.0;
        params.dy = 1.0;
        params.xmin = 0.0;
        params.ymin = 0.0;
        params.xmax = params.xmin + params.nx * params.dx;
        params.ymax = params.ymin + params.ny * params.dy;

        params.g = Constants::GRAVITY;
        params.cfl = 0.5;
        params.h_dry = Constants::DRY_TOLERANCE;
        params.friction_type = 0.0;

        params.t_start = 0.0;
        params.t_end = 10.0;
        params.dt_max = 0.1;
        params.output_interval = 1.0;

        params.bc_type[0] = 0;
        params.bc_type[1] = 0;
        params.bc_type[2] = 0;
        params.bc_type[3] = 0;

        params.use_lts = false;
        params.riemann_solver = 1;
        params.slope_limiter = 0;
        params.order = 2;
    }

    // Apply command line overrides
    if (nx_override > 0) {
        params.nx = nx_override;
        params.xmax = params.xmin + params.nx * params.dx;
    }
    if (ny_override > 0) {
        params.ny = ny_override;
        params.ymax = params.ymin + params.ny * params.dy;
    }
    if (cfl_override > 0.0) {
        params.cfl = cfl_override;
    }
    if (t_end_override > 0.0) {
        params.t_end = t_end_override;
    }

    // Print configuration summary
    if (!use_multi_gpu || multi_gpu->is_root()) {
        std::cout << "=== Simulation Configuration ===" << std::endl;
        std::cout << "Grid:        " << params.nx << " x " << params.ny << std::endl;
        std::cout << "Domain:      [" << params.xmin << ", " << params.xmax << "] x "
                  << "[" << params.ymin << ", " << params.ymax << "]" << std::endl;
        std::cout << "Grid spacing: dx=" << params.dx << " m, dy=" << params.dy << " m" << std::endl;
        std::cout << "CFL:         " << params.cfl << std::endl;
        std::cout << "Time:        " << params.t_start << " to " << params.t_end << " s" << std::endl;
        std::cout << "Test Case:   " << TestCases::get_test_name(test_case) << std::endl;
        std::cout << "Validation:  " << (enable_validation ? "Enabled" : "Disabled") << std::endl;
        std::cout << "VTK Output:  " << (enable_vtk ? "Enabled" : "Disabled") << std::endl;
        std::cout << "================================\n" << std::endl;
    }

    // Create and initialize solver
    HydroSisSolver solver;
    solver.initialize(params, use_multi_gpu);

    // Set initial conditions
    solver.set_initial_conditions(test_case);

    // Enable optional features
    if (enable_vtk) {
        solver.enable_vtk_output(true);
    }
    if (enable_validation) {
        solver.enable_validation(true, test_case);
    }

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
        printf("║ Grid size:                %4d x %4d                    ║\n", params.nx, params.ny);
        printf("║ Total cells:              %10d                      ║\n", params.nx * params.ny);
        printf("║ Time steps:               %10d                      ║\n", solver.get_step_count());
        printf("║ Final time:               %8.2f s                      ║\n", solver.get_time());

        float cell_updates = (float)params.nx * params.ny * solver.get_step_count();
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
