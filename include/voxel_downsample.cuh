#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_types.h>

namespace cuda_vgicp {
    void voxelgrid_downsample(
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
    );
}
