#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_types.h>

struct Voxel {
    int ix, iy, iz;         // Voxel indices
    float3 mean;            // Mean, but first: sum of all included points
    int count;              // Sum of points
	float covariance[9];    // Covariance, but first: sum of products of points
};

__global__ void build_voxels(const float3* points, int N, float leaf_size, Voxel* voxels, int max_voxels);
__global__ void compute_covariance_and_centroids(Voxel* voxels, int max_voxels, int* amount_centroids);

namespace cuda_vgicp {
    std::vector<float3> voxelgrid_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size);
}
