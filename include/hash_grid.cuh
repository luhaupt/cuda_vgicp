#pragma once

#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_types.h>

namespace cuda_vgicp {

struct ComputeHash {
    float voxel_size;
    __host__ __device__ ComputeHash(float v) : voxel_size(v) {}

    __device__ uint32_t operator()(const float3& p) const {
        int3 cell = make_int3(
            __float2int_rd(p.x / voxel_size),
            __float2int_rd(p.y / voxel_size),
            __float2int_rd(p.z / voxel_size)
        );
        return cell.x * 73856093u ^ cell.y * 19349663u ^ cell.z * 83492791u;
    }
};

struct SpatialHashGrid {
    thrust::device_vector<int> cell_start;
    thrust::device_vector<int> cell_end;
    thrust::device_vector<int> sorted_indices;
    thrust::device_vector<uint32_t> cell_hashes;

    // Builds a spatial hash grid for the point cloud and voxel size
    void build(const thrust::device_vector<float3>& points, float voxel_size) {
        int num_points = points.size();
        thrust::device_vector<uint32_t> cell_hashes_unsorted(num_points);
        thrust::device_vector<int> point_indices(num_points);

        thrust::transform(
            points.begin(), points.end(),
            cell_hashes_unsorted.begin(),
            ComputeHash(voxel_size)
        );

        thrust::sequence(point_indices.begin(), point_indices.end());

        this->cell_hashes = cell_hashes_unsorted;
        this->sorted_indices = point_indices;
        thrust::sort_by_key(this->cell_hashes.begin(), this->cell_hashes.end(), this->sorted_indices.begin());

        this->cell_start.resize(num_points);
        this->cell_end.resize(num_points);
        thrust::device_vector<int> num_runs(1);

        size_t temp_storage_bytes = 0;
        cub::DeviceRunLengthEncode::Encode(
            nullptr, temp_storage_bytes,
            thrust::raw_pointer_cast(this->cell_hashes.data()),
            thrust::raw_pointer_cast(this->cell_start.data()),
            thrust::raw_pointer_cast(this->cell_end.data()),
            thrust::raw_pointer_cast(num_runs.data()),
            num_points
        );

        thrust::device_vector<char> temp_storage(temp_storage_bytes);
        cub::DeviceRunLengthEncode::Encode(
            thrust::raw_pointer_cast(temp_storage.data()), temp_storage_bytes,
            thrust::raw_pointer_cast(this->cell_hashes.data()),
            thrust::raw_pointer_cast(this->cell_start.data()),
            thrust::raw_pointer_cast(this->cell_end.data()),
            thrust::raw_pointer_cast(num_runs.data()),
            num_points
        );
    }
};

} // cuda_vgicp
