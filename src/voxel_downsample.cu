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
#include "../include/voxel_downsample.h"

#include <cstdint>
#include <iostream>
#include <vector>

// -- pack voxel coordinate (x,y,z) into uint64_t using 21 bits per axis (same as original)
__host__ __device__ inline uint64_t pack_voxel_key(int vx, int vy, int vz) {
    const uint64_t mask = ((1ull << 21) - 1ull);
    return ((uint64_t)(vx & mask)) | (((uint64_t)(vy & mask)) << 21) | (((uint64_t)(vz & mask)) << 42);
}

struct ComputeKeyAndValue {
    float inv_leaf;
    int offset;
    __host__ __device__
    ComputeKeyAndValue(float leaf) : inv_leaf(1.f / leaf), offset(1 << 20) {}

    __host__ __device__
    thrust::tuple<uint64_t, float4> operator()(const pcl::PointXYZ& point) const {
        float x = point.x;
        float y = point.y;
        float z = point.z;

        int vx = static_cast<int>(floorf(x * inv_leaf)) + offset;
        int vy = static_cast<int>(floorf(y * inv_leaf)) + offset;
        int vz = static_cast<int>(floorf(z * inv_leaf)) + offset;

        const int maxcoord = (1 << 21) - 1;

        uint64_t key;
        if (vx < 0 || vy < 0 || vz < 0 || vx > maxcoord || vy > maxcoord || vz > maxcoord) {
            key = ~0ull;
        } else {
            key = pack_voxel_key(vx, vy, vz);
        }
        return thrust::make_tuple(key, make_float4(x, y, z, 1.f));
    }
};

struct SumPoint {
    __host__ __device__ thrust::tuple<float,float,float,float> operator()(const thrust::tuple<float,float,float,float>& a,
                                                                         const thrust::tuple<float,float,float,float>& b) const {
        return thrust::make_tuple(thrust::get<0>(a)+thrust::get<0>(b),
                                  thrust::get<1>(a)+thrust::get<1>(b),
                                  thrust::get<2>(a)+thrust::get<2>(b),
                                  thrust::get<3>(a)+thrust::get<3>(b));
    }
};

struct SumFloat4 {
    __host__ __device__
    float4 operator()(const float4& a, const float4& b) const {
        return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
    }
};

std::vector<float3> cuda_vgicp::voxelgrid_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size) {
    size_t N = cloud->points.size();
    thrust::device_vector<pcl::PointXYZ> d_points = cloud->points;
    thrust::device_vector<uint64_t> keys(N);
    thrust::device_vector<float4> vals(N);

    auto key_val_zip = thrust::make_zip_iterator(thrust::make_tuple(keys.begin(), vals.begin()));
    thrust::transform(d_points.begin(), d_points.end(), key_val_zip, ComputeKeyAndValue(leaf_size));
    thrust::sort_by_key(keys.begin(), keys.end(), vals.begin());

    auto valid_end = thrust::find(keys.begin(), keys.end(), ~0ull);
    size_t valid_n = thrust::distance(keys.begin(), valid_end);
    if (valid_n == 0) return {};

    thrust::device_vector<uint64_t> unique_keys(valid_n);
    thrust::device_vector<float4> sum_vals(valid_n);

    auto reduce_end = thrust::reduce_by_key(
        keys.begin(), keys.begin() + valid_n,
        vals.begin(),
        unique_keys.begin(),
        sum_vals.begin(),
        thrust::equal_to<uint64_t>(),
        SumFloat4()
    );

    size_t M = reduce_end.first - unique_keys.begin();
    thrust::device_vector<float3> centroids(M);
    thrust::transform(
        sum_vals.begin(),
        sum_vals.begin() + M,
        centroids.begin(),
        [] __device__ (const float4& s) {
            float inv_count = 1.f / s.w;
            return make_float3(s.x * inv_count, s.y * inv_count, s.z * inv_count);
        }
    );

    std::vector<float3> downsampled_points(M);
    thrust::copy_n(centroids.begin(), M, downsampled_points.begin());

    return downsampled_points;
}
