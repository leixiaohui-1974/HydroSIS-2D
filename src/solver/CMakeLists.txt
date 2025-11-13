cmake_minimum_required(VERSION 3.18)
project(HydroSIS2D_CUDA LANGUAGES CXX CUDA)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# CUDA architecture (auto-detect or specify)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80;86;89;90")  # Support common GPUs
endif()

# Find packages
find_package(CUDAToolkit REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
    # Download pybind11 if not found
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# CUDA source files
set(CUDA_SOURCES
    cuda/ShallowWaterSolver.cu
    cuda/kernels/flux_kernels.cu
    cuda/kernels/update_kernels.cu
    cuda/kernels/source_kernels.cu
    cuda/kernels/bc_kernels.cu
    cuda/kernels/muscl_kernels.cu
)

# C++ source files
set(CXX_SOURCES
    # cuda/DeviceMemory.cpp
    # cuda/utils.cpp
)

# Python binding
set(PYTHON_BINDING
    cuda/python/bindings.cpp
)

# Create Python module
pybind11_add_module(hydrosis2d_cuda
    ${PYTHON_BINDING}
    ${CUDA_SOURCES}
    ${CXX_SOURCES}
)

# Include directories
target_include_directories(hydrosis2d_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

# Link libraries
target_link_libraries(hydrosis2d_cuda PRIVATE
    CUDA::cudart
    CUDA::cuda_driver
)

# CUDA compilation flags
target_compile_options(hydrosis2d_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --expt-relaxed-constexpr
        --use_fast_math
        -lineinfo
    >
)

# Set RPATH for Python module
set_target_properties(hydrosis2d_cuda PROPERTIES
    BUILD_RPATH_USE_ORIGIN ON
    INSTALL_RPATH "$ORIGIN"
)

# Installation
install(TARGETS hydrosis2d_cuda
    LIBRARY DESTINATION ${Python3_SITEARCH}/hydrosis2d
)

# Print configuration
message(STATUS "========================================")
message(STATUS "HydroSIS-2D CUDA Solver Configuration")
message(STATUS "========================================")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Python Version: ${Python3_VERSION}")
message(STATUS "Python Include: ${Python3_INCLUDE_DIRS}")
message(STATUS "pybind11 Version: ${pybind11_VERSION}")
message(STATUS "========================================")
