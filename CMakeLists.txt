cmake_minimum_required(VERSION 3.18)

project(HydroSIS-2D LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Options
option(USE_DOUBLE_PRECISION "Use double precision" OFF)
option(ENABLE_MPI "Enable MPI for multi-GPU" ON)

# Find packages
find_package(CUDA REQUIRED)
find_package(MPI)

if(MPI_FOUND AND ENABLE_MPI)
    message(STATUS "MPI found, enabling multi-GPU support")
    add_definitions(-DUSE_MPI)
    include_directories(${MPI_INCLUDE_PATH})
else()
    message(STATUS "MPI not found or disabled, single GPU only")
endif()

# Set CUDA architecture (adjust for your GPU)
# Common options:
#   - sm_70: V100
#   - sm_75: RTX 2080, T4
#   - sm_80: A100
#   - sm_86: RTX 3090
#   - sm_89: RTX 4090
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86)

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xptxas -v")  # Verbose register usage

if(USE_DOUBLE_PRECISION)
    add_definitions(-DUSE_DOUBLE_PRECISION)
    message(STATUS "Using double precision")
else()
    message(STATUS "Using single precision (float)")
endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${CUDA_INCLUDE_DIRS})

# Source files
set(CUDA_SOURCES
    src/cuda/cuda_kernels.cu
    src/cuda/test_cases_kernels.cu
    src/parallel/halo_kernels.cu
)

set(CXX_SOURCES
    src/solver/hydrosis_solver.cpp
    src/utils/vtk_writer.cpp
    src/utils/validation.cpp
    src/utils/config_reader.cpp
    src/utils/performance_monitor.cpp
    src/main.cpp
)

if(MPI_FOUND AND ENABLE_MPI)
    list(APPEND CXX_SOURCES src/parallel/multi_gpu.cpp)
endif()

# Executable
add_executable(hydrosis ${CUDA_SOURCES} ${CXX_SOURCES})

# Link libraries
target_link_libraries(hydrosis ${CUDA_LIBRARIES})

if(MPI_FOUND AND ENABLE_MPI)
    target_link_libraries(hydrosis ${MPI_LIBRARIES})

    if(MPI_COMPILE_FLAGS)
        set_target_properties(hydrosis PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
    endif()

    if(MPI_LINK_FLAGS)
        set_target_properties(hydrosis PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
    endif()
endif()

# Installation
install(TARGETS hydrosis DESTINATION bin)

# Print configuration
message(STATUS "")
message(STATUS "==================================================")
message(STATUS "HydroSIS-2D Configuration Summary")
message(STATUS "==================================================")
message(STATUS "Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler:         ${CMAKE_CXX_COMPILER}")
message(STATUS "CUDA compiler:        ${CMAKE_CUDA_COMPILER}")
message(STATUS "CUDA architectures:   ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Precision:            ${USE_DOUBLE_PRECISION}")
message(STATUS "MPI enabled:          ${ENABLE_MPI}")
if(MPI_FOUND AND ENABLE_MPI)
    message(STATUS "MPI C++ compiler:     ${MPI_CXX_COMPILER}")
endif()
message(STATUS "==================================================")
message(STATUS "")
