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
#include "../include/voxel_downsample.cuh"

#include <cstdint>
#include <iostream>
#include <vector>

__global__ void build_voxels(
    const float3* points,
    int N,
    float leaf_size,
    Voxel* voxels,
    int max_voxels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= N) return;

	float3 p = points[idx];
    int ix = floorf(p.x / leaf_size);
    int iy = floorf(p.y / leaf_size);
    int iz = floorf(p.z / leaf_size);
    uint32_t hash = ix * 73856093 ^ iy * 19349663 ^ iz * 83492791;
    uint32_t v_idx = hash % max_voxels;

    // Set voxel indices if not set before
    if (voxels[v_idx].ix != voxels[v_idx].ix) {
        voxels[v_idx].ix = ix;
        voxels[v_idx].iy = iy;
        voxels[v_idx].iz = iz;
    }

    // For mean value
	atomicAdd(&voxels[v_idx].mean.x, p.x);
	atomicAdd(&voxels[v_idx].mean.y, p.y);
	atomicAdd(&voxels[v_idx].mean.z, p.z);

	// For covariances
	atomicAdd(&voxels[v_idx].covariance[0], p.x*p.x);
	atomicAdd(&voxels[v_idx].covariance[1], p.x*p.y);
	atomicAdd(&voxels[v_idx].covariance[2], p.x*p.z);
	atomicAdd(&voxels[v_idx].covariance[3], p.y*p.x);
	atomicAdd(&voxels[v_idx].covariance[4], p.y*p.y);
	atomicAdd(&voxels[v_idx].covariance[5], p.y*p.z);
	atomicAdd(&voxels[v_idx].covariance[6], p.z*p.x);
	atomicAdd(&voxels[v_idx].covariance[7], p.z*p.y);
	atomicAdd(&voxels[v_idx].covariance[8], p.z*p.z);

	// For amount of points
	atomicAdd(&voxels[v_idx].count, 1);
}

__global__ void compute_covariance_and_centroids(
    Voxel* voxels,
    int max_voxels,
    int* amount_centroids
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx >= max_voxels) return;

	Voxel* v = &voxels[idx];

	if(v->count == 0) return;

	// Set real mean value
	v->mean = make_float3(
	    v->mean.x / v->count,
		v->mean.y / v->count,
		v->mean.z / v->count
	);

	// Set real covariances
	v->covariance[0] = v->covariance[0] / v->count - v->mean.x * v->mean.x;
    v->covariance[1] = v->covariance[1] / v->count - v->mean.x * v->mean.y;
    v->covariance[2] = v->covariance[2] / v->count - v->mean.x * v->mean.z;
    v->covariance[3] = v->covariance[3] / v->count - v->mean.y * v->mean.x;
    v->covariance[4] = v->covariance[4] / v->count - v->mean.y * v->mean.y;
    v->covariance[5] = v->covariance[5] / v->count - v->mean.y * v->mean.z;
    v->covariance[6] = v->covariance[6] / v->count - v->mean.z * v->mean.x;
    v->covariance[7] = v->covariance[7] / v->count - v->mean.z * v->mean.y;
    v->covariance[8] = v->covariance[8] / v->count - v->mean.z * v->mean.z;

    // Increment amount of centroids if voxel has points
    atomicAdd(amount_centroids, 1);
}

std::vector<float3> cuda_vgicp::voxelgrid_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size) {
    size_t N = cloud->points.size();
    int threadsPerBlock, minBlocksCent;
    cudaOccupancyMaxPotentialBlockSize(&minBlocksCent, &threadsPerBlock, build_voxels, 0, 0);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    float3* d_points;

    cudaMalloc(&d_points, N * sizeof(float3));
    cudaMemcpy(d_points, cloud->points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    int max_voxels = 2 * N;
    Voxel* d_voxels;
    cudaMalloc(&d_voxels, max_voxels * sizeof(Voxel));
    cudaMemset(d_voxels, 0, max_voxels * sizeof(Voxel));

    // Synchronize device after voxel built
    build_voxels<<<blocks, threadsPerBlock>>>(d_points, N, leaf_size, d_voxels, max_voxels);
    cudaDeviceSynchronize();

    // Synchronize device after centroid calculation
    float3* d_centroids;
    int* d_amount_centroids;
    cudaMalloc(&d_centroids, max_voxels * sizeof(float3));
    cudaMalloc(&d_amount_centroids, sizeof(int));
    cudaMemset(d_amount_centroids, 0, sizeof(int));
    cudaOccupancyMaxPotentialBlockSize(&minBlocksCent, &threadsPerBlock, compute_covariance_and_centroids, 0, 0);
    int blocks_vox = (max_voxels + threadsPerBlock - 1) / threadsPerBlock;
    compute_covariance_and_centroids<<<blocks_vox, threadsPerBlock>>>(d_voxels, max_voxels, d_amount_centroids);
    cudaDeviceSynchronize();

    int amount_centroids;
    cudaMemcpy(&amount_centroids, d_amount_centroids, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<float3> downsampled(amount_centroids);
    cudaMemcpy(downsampled.data(), d_centroids, amount_centroids * sizeof(float3), cudaMemcpyDeviceToHost);
    // std::vector<Voxel> voxels(N);
    // cudaMemcpy(voxels.data(), d_voxels, N * sizeof(Voxel), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_points);
    cudaFree(d_voxels);
    cudaFree(d_centroids);
    cudaFree(d_amount_centroids);

    return downsampled;
}
