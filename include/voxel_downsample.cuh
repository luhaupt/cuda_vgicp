#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_types.h>

struct Voxel {
    float x, y, z;
    int count;
};

__global__ void voxelgrid_kernel(const float3* points, int N, float leaf_size, Voxel* voxels, int max_voxels);
__global__ void compute_centroids(const Voxel* voxels, int max_voxels, float3* centroids, int* amount_centroids);

namespace cuda_vgicp {
    std::vector<float3> voxelgrid_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size);
}
