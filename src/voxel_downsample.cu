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

__global__ void voxelgrid_kernel(
    const float3* points,
    int N,
    float leaf_size,
    Voxel* voxels,
    int max_voxels
) {
    // Share local voxel array inside blocks
    extern __shared__ Voxel s_voxels[];
    int tid = threadIdx.x;
    int blockOffset = blockIdx.x * blockDim.x;

    for (int i = tid; i < max_voxels; i += blockDim.x) {
        s_voxels[i].x = 0.f;
        s_voxels[i].y = 0.f;
        s_voxels[i].z = 0.f;
        s_voxels[i].count = 0;
    }

    __syncthreads();

    // Local accumulation in shared memory
    for(int idx = blockOffset + tid; idx < N; idx += gridDim.x * blockDim.x) {
        float3 p = points[idx];

        int ix = static_cast<int>(floorf(p.x / leaf_size));
        int iy = static_cast<int>(floorf(p.y / leaf_size));
        int iz = static_cast<int>(floorf(p.z / leaf_size));

        int v_idx = ((ix * 73856093 ^ iy * 19349663 ^ iz * 83492791) & 0x7fffffff) % max_voxels;

        atomicAdd(&s_voxels[v_idx].x, p.x);
        atomicAdd(&s_voxels[v_idx].y, p.y);
        atomicAdd(&s_voxels[v_idx].z, p.z);
        atomicAdd(&s_voxels[v_idx].count, 1);
    }

    __syncthreads();

    // Write shared local voxel array to global voxel array
    for(int i = tid; i < max_voxels; i += blockDim.x) {
        if(s_voxels[i].count > 0) {
            atomicAdd(&voxels[i].x, s_voxels[i].x);
            atomicAdd(&voxels[i].y, s_voxels[i].y);
            atomicAdd(&voxels[i].z, s_voxels[i].z);
            atomicAdd(&voxels[i].count, s_voxels[i].count);
        }
    }
}

__global__ void compute_centroids(
    const Voxel* voxels,
    int max_voxels,
    float3* centroids,
    int* amount_centroids
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= max_voxels) return;

    if(voxels[idx].count > 0) {
        centroids[idx] = make_float3(
            voxels[idx].x / voxels[idx].count,
            voxels[idx].y / voxels[idx].count,
            voxels[idx].z / voxels[idx].count
        );
        atomicAdd(amount_centroids, 1);
    }
}

std::vector<float3> cuda_vgicp::voxelgrid_downsample(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float leaf_size) {
    size_t N = cloud->points.size();
    int threadsPerBlock, minBlocksCent;
    cudaOccupancyMaxPotentialBlockSize(&minBlocksCent, &threadsPerBlock, voxelgrid_kernel, 0, 0);
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    float3* d_points;

    cudaMalloc(&d_points, N * sizeof(float3));
    cudaMemcpy(d_points, cloud->points.data(), N * sizeof(float3), cudaMemcpyHostToDevice);

    int max_voxels = 2 * N;
    Voxel* d_voxels;
    cudaMalloc(&d_voxels, max_voxels * sizeof(Voxel));
    cudaMemset(d_voxels, 0, max_voxels * sizeof(Voxel));

    // Synchronize device after voxelgrid calculation
    voxelgrid_kernel<<<blocks, threadsPerBlock>>>(d_points, N, leaf_size, d_voxels, max_voxels);
    cudaDeviceSynchronize();

    // Synchronize device after centroid calculation
    float3* d_centroids;
    int* d_amount_centroids;
    cudaMalloc(&d_centroids, max_voxels * sizeof(float3));
    cudaMalloc(&d_amount_centroids, sizeof(int));
    cudaMemset(d_amount_centroids, 0, sizeof(int));
    cudaOccupancyMaxPotentialBlockSize(&minBlocksCent, &threadsPerBlock, compute_centroids, 0, 0);
    int blocks_vox = (max_voxels + threadsPerBlock - 1) / threadsPerBlock;
    compute_centroids<<<blocks_vox, threadsPerBlock>>>(d_voxels, max_voxels, d_centroids, d_amount_centroids);
    cudaDeviceSynchronize();

    int amount_centroids;
    cudaMemcpy(&amount_centroids, d_amount_centroids, sizeof(int), cudaMemcpyDeviceToHost);
    std::vector<float3> downsampled(amount_centroids);
    cudaMemcpy(downsampled.data(), d_centroids, amount_centroids * sizeof(float3), cudaMemcpyDeviceToHost);

    // Free allocated memory
    cudaFree(d_points);
    cudaFree(d_voxels);
    cudaFree(d_centroids);
    cudaFree(d_amount_centroids);

    return downsampled;
}
