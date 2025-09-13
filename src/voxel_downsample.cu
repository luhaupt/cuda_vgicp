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
    int* __restrict__ d_num_unique_cells
) {
    // Used device pointer
    int* d_point_indices;
    int* d_point_indices_sorted;
    uint32_t* d_point_cell_hashes;
    uint32_t* d_point_cell_hashes_sorted;
    uint32_t* d_unique_point_cell_hashes;
    int* d_cell_start;
    int* d_cell_end;
    int* d_points_per_cell;
    int* d_neighbor_indices;
    float* d_neighbor_distances;
    float* d_point_covariances;
    float* d_voxel_centroids;
    float* d_voxel_covariances;

    cudaMalloc(&d_point_indices, N * sizeof(int));
    cudaMalloc(&d_point_indices_sorted, N * sizeof(int));
    cudaMalloc(&d_point_cell_hashes, N * sizeof(uint32_t));
    cudaMalloc(&d_point_cell_hashes_sorted, N * sizeof(uint32_t));
    cudaMalloc(&d_unique_point_cell_hashes, N * sizeof(uint32_t));
    cudaMalloc(&d_cell_start, N * sizeof(int));
    cudaMalloc(&d_cell_end, N * sizeof(int));
    cudaMalloc(&d_points_per_cell, N * sizeof(int));
    cudaMalloc(&d_neighbor_indices, N * K_NEIGHBORS * sizeof(int));
    cudaMalloc(&d_neighbor_distances, N * K_NEIGHBORS * sizeof(float));
    cudaMalloc(&d_point_covariances, N * 6 * sizeof(float));
    cudaMalloc(&d_voxel_centroids, N * 3 * sizeof(float));
    cudaMalloc(&d_voxel_covariances, N * 6 * sizeof(float));


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

    cudaFree(d_point_indices);
    cudaFree(d_point_indices_sorted);
    cudaFree(d_point_cell_hashes);
    cudaFree(d_point_cell_hashes_sorted);
    cudaFree(d_unique_point_cell_hashes);
    cudaFree(d_cell_start);
    cudaFree(d_cell_end);
    cudaFree(d_points_per_cell);
    cudaFree(d_neighbor_indices);
    cudaFree(d_neighbor_distances);
    cudaFree(d_point_covariances);
    cudaFree(d_voxel_centroids);
    cudaFree(d_voxel_covariances);
}
