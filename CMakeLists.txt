cmake_minimum_required(VERSION 3.18)  # CUDA 11+ Support
project(cuda_vgicp LANGUAGES C CXX CUDA)

find_package(Eigen3 CONFIG)
find_package(PCL REQUIRED)
find_package(CUDAToolkit 13 REQUIRED)
find_package(fmt REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CUDA_IMPLEMENTATION
    src/voxel_downsample.cu
)
include_directories(${PCL_INCLUDE_DIRS} /usr/local/cuda-13.0/include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_library(cuda_implementation STATIC ${CUDA_IMPLEMENTATION})
target_compile_options(cuda_implementation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
set_target_properties(cuda_implementation PROPERTIES
    CUDA_ARCHITECTURES 89
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

###############
## Benchmark ##
###############

# Downsampling benchmark
add_executable(downsampling_benchmark
    src/benchmark/downsampling_benchmark.cpp
)
target_include_directories(downsampling_benchmark PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(downsampling_benchmark PRIVATE
    fmt::fmt
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    cuda_implementation
)
