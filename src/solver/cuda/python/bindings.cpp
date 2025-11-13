/**
 * @file bindings.cpp
 * @brief pybind11 Python bindings for CUDA solver
 * @author HydroSIS-2D Team
 * @date 2025-11-13
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "../ShallowWaterSolver.cuh"

namespace py = pybind11;
using namespace hydrosis2d::cuda;

/**
 * @brief Python wrapper for ShallowWaterSolver
 *
 * Provides numpy array interface and exception handling.
 */
class PySolver {
private:
    ShallowWaterSolver solver_;
    int nx_, ny_;

public:
    PySolver() : nx_(0), ny_(0) {}

    void initialize(const std::string& config_file) {
        solver_.initialize(config_file);
        // Get mesh dimensions from config
        // (This would be extracted from JSON config)
    }

    void setInitialConditions(
        py::array_t<double> h,
        py::array_t<double> u,
        py::array_t<double> v,
        py::array_t<double> z
    ) {
        // Check dimensions
        if (h.ndim() != 2) {
            throw std::runtime_error("h must be 2D array");
        }

        ny_ = h.shape(0);
        nx_ = h.shape(1);

        // Get raw pointers
        auto h_buf = h.request();
        auto u_buf = u.request();
        auto v_buf = v.request();
        auto z_buf = z.request();

        double* h_ptr = static_cast<double*>(h_buf.ptr);
        double* u_ptr = static_cast<double*>(u_buf.ptr);
        double* v_ptr = static_cast<double*>(v_buf.ptr);
        double* z_ptr = static_cast<double*>(z_buf.ptr);

        solver_.setInitialConditions(h_ptr, u_ptr, v_ptr, z_ptr);
    }

    void run(double t_end) {
        solver_.run(t_end);
    }

    void step(double dt) {
        solver_.step(dt);
    }

    py::dict getSolution() {
        std::vector<double> h_vec(nx_ * ny_);
        std::vector<double> u_vec(nx_ * ny_);
        std::vector<double> v_vec(nx_ * ny_);

        solver_.getSolution(h_vec.data(), u_vec.data(), v_vec.data());

        // Convert to numpy arrays
        py::array_t<double> h({ny_, nx_}, h_vec.data());
        py::array_t<double> u({ny_, nx_}, u_vec.data());
        py::array_t<double> v({ny_, nx_}, v_vec.data());

        py::dict result;
        result["time"] = solver_.getCurrentTime();
        result["step"] = solver_.getCurrentStep();
        result["h"] = h;
        result["u"] = u;
        result["v"] = v;

        return result;
    }

    void exportVTK(const std::string& filename) {
        solver_.exportVTK(filename);
    }

    void setCallback(py::function callback, int interval = 100) {
        // Wrap Python function in C++ lambda
        auto cpp_callback = [callback](double t, int step, double dt) {
            try {
                callback(t, step, dt);
            } catch (py::error_already_set& e) {
                // Handle Python exceptions
                std::cerr << "Python callback error: " << e.what() << std::endl;
            }
        };

        solver_.setCallback(cpp_callback, interval);
    }

    double getCurrentTime() const {
        return solver_.getCurrentTime();
    }

    int getCurrentStep() const {
        return solver_.getCurrentStep();
    }

    double getGPUMemoryUsage() const {
        return solver_.getGPUMemoryUsage();
    }
};

PYBIND11_MODULE(hydrosis2d_cuda, m) {
    m.doc() = "HydroSIS-2D GPU-accelerated solver module";

    // Enum bindings
    py::enum_<MeshType>(m, "MeshType")
        .value("STRUCTURED", MeshType::STRUCTURED)
        .value("UNSTRUCTURED", MeshType::UNSTRUCTURED)
        .export_values();

    py::enum_<RiemannSolver>(m, "RiemannSolver")
        .value("HLL", RiemannSolver::HLL)
        .value("HLLC", RiemannSolver::HLLC)
        .value("ROE", RiemannSolver::ROE)
        .export_values();

    py::enum_<Limiter>(m, "Limiter")
        .value("NONE", Limiter::NONE)
        .value("MINMOD", Limiter::MINMOD)
        .value("VANLEER", Limiter::VANLEER)
        .value("SUPERBEE", Limiter::SUPERBEE)
        .value("MC", Limiter::MC)
        .export_values();

    py::enum_<TimeIntegrator>(m, "TimeIntegrator")
        .value("EULER", TimeIntegrator::EULER)
        .value("RK2", TimeIntegrator::RK2)
        .value("RK3_TVD", TimeIntegrator::RK3_TVD)
        .export_values();

    // Configuration struct
    py::class_<SolverConfig>(m, "SolverConfig")
        .def(py::init<>())
        .def_readwrite("mesh_type", &SolverConfig::mesh_type)
        .def_readwrite("nx", &SolverConfig::nx)
        .def_readwrite("ny", &SolverConfig::ny)
        .def_readwrite("dx", &SolverConfig::dx)
        .def_readwrite("dy", &SolverConfig::dy)
        .def_readwrite("riemann_solver", &SolverConfig::riemann_solver)
        .def_readwrite("limiter", &SolverConfig::limiter)
        .def_readwrite("time_integrator", &SolverConfig::time_integrator)
        .def_readwrite("spatial_order", &SolverConfig::spatial_order)
        .def_readwrite("gravity", &SolverConfig::gravity)
        .def_readwrite("manning_n", &SolverConfig::manning_n)
        .def_readwrite("dry_threshold", &SolverConfig::dry_threshold)
        .def_readwrite("cfl", &SolverConfig::cfl)
        .def_readwrite("max_dt", &SolverConfig::max_dt)
        .def_readwrite("min_dt", &SolverConfig::min_dt)
        .def_readwrite("output_interval", &SolverConfig::output_interval)
        .def_readwrite("output_dir", &SolverConfig::output_dir)
        .def_readwrite("output_format", &SolverConfig::output_format);

    // Main solver class
    py::class_<PySolver>(m, "Solver")
        .def(py::init<>())
        .def("initialize", &PySolver::initialize,
             py::arg("config_file"),
             "Initialize solver from JSON configuration file")
        .def("set_initial_conditions", &PySolver::setInitialConditions,
             py::arg("h"), py::arg("u"), py::arg("v"), py::arg("z"),
             "Set initial conditions (numpy arrays)")
        .def("run", &PySolver::run,
             py::arg("t_end"),
             "Run simulation until t_end")
        .def("step", &PySolver::step,
             py::arg("dt"),
             "Perform single time step")
        .def("get_solution", &PySolver::getSolution,
             "Get current solution as dictionary with numpy arrays")
        .def("export_vtk", &PySolver::exportVTK,
             py::arg("filename"),
             "Export current state to VTK file")
        .def("set_callback", &PySolver::setCallback,
             py::arg("callback"), py::arg("interval") = 100,
             "Set progress callback function")
        .def("get_current_time", &PySolver::getCurrentTime,
             "Get current simulation time")
        .def("get_current_step", &PySolver::getCurrentStep,
             "Get current step number")
        .def("get_gpu_memory_usage", &PySolver::getGPUMemoryUsage,
             "Get GPU memory usage in MB");

    // Utility functions
    m.def("get_gpu_count", &getGPUCount,
          "Get number of available CUDA devices");

    m.def("set_gpu", &setGPU,
          py::arg("device_id"),
          "Set active CUDA device");

    m.def("get_gpu_memory", [](){
        double free_mb, total_mb;
        getGPUMemory(free_mb, total_mb);
        py::dict mem;
        mem["free"] = free_mb;
        mem["total"] = total_mb;
        mem["used"] = total_mb - free_mb;
        return mem;
    }, "Get GPU memory info (MB)");

    // Version info
    m.attr("__version__") = "1.0.0-dev";
}
