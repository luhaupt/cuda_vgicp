#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_types.h>

namespace cuda_vgicp {

struct Hash {
    float cell_size;
    __host__ __device__ Hash(float c) : cell_size(c) {}

    // Calculates the hash of a point based on its cell index
    __device__ uint32_t operator()(const float3& point) const {
        int ix = static_cast<int>(floorf(point.x / this->cell_size));
        int iy = static_cast<int>(floorf(point.y / this->cell_size));
        int iz = static_cast<int>(floorf(point.z / this->cell_size));

        return ix * 73856093u ^ iy * 19349663u ^ iz * 83492791u;
    }
};

struct SpatialHashGrid {
    thrust::device_vector<int> point_indices;
    thrust::device_vector<uint32_t> cell_hashes;
    thrust::device_vector<int> cell_start;
    thrust::device_vector<int> cell_end;
    thrust::device_vector<int> num_runs{1};
    float cell_size;

    // Builds a spatial hash grid for the point cloud and voxel size
    __host__ __device__ void build(const thrust::device_vector<float3>& points, float voxel_size) {
        int num_points = points.size();
        this->point_indices.resize(num_points);
        this->cell_hashes.resize(num_points);
        this->cell_start.resize(num_points);
        this->cell_end.resize(num_points);

        // Set cell size > voxel/leaf size (downsampling)
        this->cell_size = voxel_size * 2.0;

        thrust::sequence(point_indices.begin(), point_indices.end());
        thrust::transform(
            points.begin(), points.end(),
            this->cell_hashes.begin(),
            Hash(this->cell_size)
        );

        thrust::sort_by_key(this->cell_hashes.begin(), this->cell_hashes.end(), this->point_indices.begin());

        size_t temp_storage_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(
            nullptr, temp_storage_bytes,
            thrust::raw_pointer_cast(this->cell_hashes.data()),
            thrust::raw_pointer_cast(this->cell_start.data()),
            thrust::raw_pointer_cast(this->cell_end.data()),
            thrust::raw_pointer_cast(this->num_runs.data()),
            num_points
        );

        thrust::device_vector<char> temp_storage(temp_storage_bytes);
        cub::DeviceRunLengthEncode::Encode(
            thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes,
            thrust::raw_pointer_cast(this->cell_hashes.data()),
            thrust::raw_pointer_cast(this->cell_start.data()),
            thrust::raw_pointer_cast(this->cell_end.data()),
            thrust::raw_pointer_cast(this->num_runs.data()),
            num_points
        );
    }
};

// __device__ void nearest_neighbor_search(
//     const SpatialHashGrid& grid,
//     const float3* points,
//     float3 query_point,
//     float search_radius,
//     int* neighbors,
//     int& num_neighbors,
//     int max_neighbors
// ) {
//     float cell_size = grid.cell_size;
//     int search_cells = std::ceil(search_radius / cell_size);
//     num_neighbors = 0;

//     float3 center_cell = make_float3(
//         __float2int_rd(query_point.x / cell_size),
//         __float2int_rd(query_point.y / cell_size),
//         __float2int_rd(query_point.z / cell_size)
//     );

//     // Search neighbor cells
//     for (std::size_t dx = -search_cells; dx <= search_cells, dx++) {
//         for (std::size_t dy = -search_cells; dy <= search_cells, dy++) {
//             for (std::size_t dz = -search_cells; dz <= search_cells, dz++) {
//                 float3 cell = make_float3(
//                     center_cell.x + dx,
//                     center_cell.y + dy,
//                     center_cell.z + dz
//                 );

//                 uint32_t hash = ComputeHash(cell);
//                 auto it = thrust::lower_bound(
//                     thrust::seq,
//                     grid.cell_hashes.data(),
//                     grid.cell_hashes.data() + grid.cell_hashes.size(),
//                     hash
//                 );

//                 if (it != grid.cell_hashes.end() && *it == hash) {
//                     int start = it - grid.cell_hashes.data();
//                     int end = thrust::upper_bound(thrust::seq,
//                         grid.cell_hashes.data() + start,
//                         grid.cell_hashes.data() + grid.cell_hashes.size(),
//                         hash
//                     ) - grid.cell_hashes.data();

//                     for (int i = start; i < end && num_neighbors < max_neighbors; i++) {
//                         int point_idx = grid.sorted_indices[i];
//                         float3 p = points[point_idx];
//                         float dist = length(p - query_point);

//                         if (dist <= search_radius) {
//                             neighbors[num_neighbors++] = point_idx;
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

} // cuda_vgicp
