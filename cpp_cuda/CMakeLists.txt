cmake_minimum_required(VERSION 3.18)
project(holographic_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# CUDA architecture: Blackwell / Compute 12.0
set(CMAKE_CUDA_ARCHITECTURES 120)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -fopenmp")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")

# ---- Dependencies ----

# CUDA
find_package(CUDAToolkit REQUIRED)

# FFTW3 (single-precision)
find_path(FFTW3F_INCLUDE_DIR fftw3.h HINTS /usr/include)
find_library(FFTW3F_LIB fftw3f HINTS /usr/lib/x86_64-linux-gnu)
find_library(FFTW3F_THREADS_LIB fftw3f_threads HINTS /usr/lib/x86_64-linux-gnu)
find_library(FFTW3F_OMP_LIB fftw3f_omp HINTS /usr/lib/x86_64-linux-gnu)

if(NOT FFTW3F_LIB)
    message(FATAL_ERROR "Could not find libfftw3f. Install with: sudo apt install libfftw3-dev")
endif()

# OpenMP
find_package(OpenMP REQUIRED)

# pybind11
find_package(pybind11 REQUIRED)

# ---- Static Libraries ----

# CUDA holographic reconstruction library
add_library(holographic_cuda_lib STATIC
    src/holographic_reconstruction.cu
)
target_include_directories(holographic_cuda_lib PUBLIC include)
target_link_libraries(holographic_cuda_lib PUBLIC
    CUDA::cufft
    CUDA::cudart
)
set_target_properties(holographic_cuda_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# CPU holographic reconstruction library (OpenMP + FFTW)
add_library(holographic_cpu_lib STATIC
    src/cpu_baseline.cpp
)
target_include_directories(holographic_cpu_lib PUBLIC
    include
    ${FFTW3F_INCLUDE_DIR}
)
target_link_libraries(holographic_cpu_lib PUBLIC
    ${FFTW3F_LIB}
    ${FFTW3F_THREADS_LIB}
    ${FFTW3F_OMP_LIB}
    OpenMP::OpenMP_CXX
    m
)
set_target_properties(holographic_cpu_lib PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# ---- pybind11 Module ----

pybind11_add_module(cuda_holographic
    src/python_wrapper.cu
)
target_link_libraries(cuda_holographic PRIVATE
    holographic_cuda_lib
    holographic_cpu_lib
)
set_target_properties(cuda_holographic PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ---- Standalone Benchmarks (for profiling) ----

add_executable(benchmark_cuda src/benchmark_cuda.cu)
target_link_libraries(benchmark_cuda PRIVATE holographic_cuda_lib)

add_executable(benchmark_cpu src/benchmark_cpu.cpp)
target_link_libraries(benchmark_cpu PRIVATE holographic_cpu_lib)

# ---- Tests ----

add_executable(test_reconstruction tests/test_reconstruction.cpp)
target_link_libraries(test_reconstruction PRIVATE
    holographic_cuda_lib
    holographic_cpu_lib
)

# ---- Pipeline (TensorRT + OpenCV) ----

# OpenCV
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# TensorRT
find_path(TENSORRT_INCLUDE_DIR NvInfer.h HINTS /usr/include/x86_64-linux-gnu)
find_library(NVINFER_LIB nvinfer HINTS /usr/lib/x86_64-linux-gnu)
find_library(NVINFER_PLUGIN_LIB nvinfer_plugin HINTS /usr/lib/x86_64-linux-gnu)

if(NOT NVINFER_LIB)
    message(WARNING "TensorRT not found. Pipeline targets will not be built.")
else()
    message(STATUS "TensorRT found: ${NVINFER_LIB}")
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")

    # Pipeline static library
    add_library(pipeline_lib STATIC
        src/ema_stage.cu
        src/tensorrt_engine.cu
        src/peak_detection.cu
        src/postprocessing.cpp
        src/pipeline.cu
    )
    target_include_directories(pipeline_lib PUBLIC
        include
        ${TENSORRT_INCLUDE_DIR}
    )
    target_link_libraries(pipeline_lib PUBLIC
        holographic_cuda_lib
        ${NVINFER_LIB}
        ${NVINFER_PLUGIN_LIB}
        ${OpenCV_LIBS}
        CUDA::cudart
    )
    set_target_properties(pipeline_lib PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        POSITION_INDEPENDENT_CODE ON
    )

    # Pipeline benchmark executable
    add_executable(benchmark_pipeline src/benchmark_pipeline.cu)
    target_link_libraries(benchmark_pipeline PRIVATE pipeline_lib)
    set_target_properties(benchmark_pipeline PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()
