#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/reduce.h>
#include <thrust/unique.h>
#include <thrust/tuple.h>
#include <cub/cub.cuh>
#include "../include/voxel_downsample.cuh"
#include "../include/hash_grid.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

void cuda_vgicp::voxelgrid_downsample(
    const float3* __restrict__ d_points,
    size_t N,
    float leaf_size,
    int* __restrict__ d_num_unique_cells,
    int* __restrict__ d_point_indices,
    int* __restrict__ d_point_indices_sorted,
    uint32_t* __restrict__ d_point_cell_hashes,
    uint32_t* __restrict__ d_point_cell_hashes_sorted,
    uint32_t* __restrict__ d_unique_point_cell_hashes,
    int* __restrict__ d_cell_start,
    int* __restrict__ d_cell_end,
    int* __restrict__ d_points_per_cell,
    int* __restrict__ d_neighbor_indices,
    float* __restrict__ d_neighbor_distances,
    float* __restrict__ d_point_covariances,
    float* __restrict__ d_voxel_centroids,
    float* __restrict__ d_voxel_covariances
) {
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    init_indices_and_compute_cell_hashes_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_points,
        N,
        leaf_size,
        d_point_indices,
        d_point_cell_hashes
    );

    // Sort point indices by cell hash
    size_t temp_bytes = 0;
    void* d_temp = nullptr;
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_point_cell_hashes, d_point_cell_hashes_sorted,
        d_point_indices, d_point_indices_sorted,
        N
    );
    cudaMalloc(&d_temp, temp_bytes);
    cub::DeviceRadixSort::SortPairs(
        d_temp, temp_bytes,
        d_point_cell_hashes, d_point_cell_hashes_sorted,
        d_point_indices, d_point_indices_sorted,
        N
    );
    cudaFree(d_temp);

    // Get amount of unique cells
    size_t rle_temp_bytes = 0;
    void* rle_temp = nullptr;
    cub::DeviceRunLengthEncode::Encode(
        rle_temp, rle_temp_bytes,
        d_point_cell_hashes_sorted,
        d_unique_point_cell_hashes,
        d_points_per_cell,
        d_num_unique_cells,
        N
    );
    cudaMalloc(&rle_temp, rle_temp_bytes);
    cub::DeviceRunLengthEncode::Encode(
        rle_temp, rle_temp_bytes,
        d_point_cell_hashes_sorted,
        d_unique_point_cell_hashes,
        d_points_per_cell,
        d_num_unique_cells,
        N
    );
    cudaFree(rle_temp);

    int h_num_unique_cells = 0;
    cudaMemcpy(&h_num_unique_cells, d_num_unique_cells, sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate cell start and end values
    temp_bytes = 0;
    void* d_temp_storage = nullptr;
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_bytes,
        d_points_per_cell,
        d_cell_start,
        h_num_unique_cells
    );
    cudaMalloc(&d_temp_storage, temp_bytes);
    cub::DeviceScan::ExclusiveSum(
        d_temp_storage,
        temp_bytes,
        d_points_per_cell,
        d_cell_start,
        h_num_unique_cells
    );
    cudaFree(d_temp_storage);

    blocks = (h_num_unique_cells + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_cell_ranges_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_cell_start,
        d_points_per_cell,
        d_cell_end,
        d_num_unique_cells
    );

    int shared_mem_size = THREADS_PER_BLOCK * K_NEIGHBORS * (sizeof(float) + sizeof(int));
    find_k_nearest_neighbors_kernel<<<blocks, THREADS_PER_BLOCK, shared_mem_size>>>(
        d_points,
        N,
        d_point_indices_sorted,
        d_point_cell_hashes_sorted,
        d_cell_start,
        d_cell_end,
        d_num_unique_cells,
        d_points_per_cell,
        leaf_size,
        d_neighbor_indices,
        d_neighbor_distances
    );

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error after kernel launch: "
                  << cudaGetErrorString(err) << std::endl;
    }

    blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_point_covariances_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_points,
        N,
        d_neighbor_indices,
        d_point_covariances
    );

    blocks = (h_num_unique_cells + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    compute_voxel_means_and_covariances_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        d_points,
        d_cell_start,
        d_cell_end,
        d_num_unique_cells,
        d_point_covariances,
        d_voxel_centroids,
        d_voxel_covariances
    );
}
