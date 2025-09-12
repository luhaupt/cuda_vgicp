cmake_minimum_required(VERSION 3.18)  # CUDA 11+ Support
project(cuda_vgicp LANGUAGES C CXX CUDA)

find_package(Eigen3 CONFIG)
find_package(PCL REQUIRED)
find_package(CUDAToolkit 13 REQUIRED)
find_package(fmt REQUIRED)
find_package(CUDA REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_library(cuda_vgicp
    src/voxel_downsample.cu
)
target_include_directories(cuda_vgicp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(cuda_vgicp PUBLIC
    fmt::fmt
)

include_directories(${PCL_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}
    ${CUDAToolkit_INCLUDE_DIRS}/cccl
)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

target_compile_options(cuda_vgicp PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
set_target_properties(cuda_vgicp PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "89"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

###############
## Benchmark ##
###############

# Downsampling benchmark
add_executable(downsampling_benchmark
    src/benchmark/downsampling_benchmark.cu
)
target_include_directories(downsampling_benchmark PRIVATE
    ${PCL_INCLUDE_DIRS}
)
target_link_libraries(downsampling_benchmark PRIVATE
    fmt::fmt
    Eigen3::Eigen
    ${PCL_LIBRARIES}
    cuda_vgicp
)

# Hash grid benchmark
# add_executable(hash_grid_benchmark
#     src/benchmark/hash_grid_benchmark.cu
# )
# target_include_directories(hash_grid_benchmark PRIVATE
#     ${PCL_INCLUDE_DIRS}
# )
# target_link_libraries(hash_grid_benchmark PRIVATE
#     cuda_vgicp
#     ${PCL_LIBRARIES}
#     CUDA::cudart
# )

# NNS benchmark
# add_executable(nns_benchmark
#     src/benchmark/nns_benchmark.cu
# )
# target_include_directories(nns_benchmark PRIVATE
#     ${PCL_INCLUDE_DIRS}
# )
# target_link_libraries(nns_benchmark PRIVATE
#     cuda_vgicp
#     ${PCL_LIBRARIES}
#     CUDA::cudart
# )
