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

#define K_NEIGHBORS 20

namespace cuda_vgicp {

__device__ __forceinline__ uint32_t hash_coord(int x, int y, int z) {
    return x * 73856093u ^ y * 19349663u ^ z * 83492791u;
}

__global__
void init_indices_and_compute_cell_hashes_kernel(
    const float3* __restrict__ d_points,
    const size_t N,
    const float cell_size,
    int* __restrict__ point_indices,
    uint32_t* __restrict__ point_cell_hashes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    // Get index of cell for point
    float3 p = d_points[idx];

    if (!isfinite(p.x) || !isfinite(p.y) || !isfinite(p.z)) {
        return;
    }

    int cell_x = floorf(p.x / cell_size);
    int cell_y = floorf(p.y / cell_size);
    int cell_z = floorf(p.z / cell_size);

    // Set point index and cell hash (of point)
    point_indices[idx]      = idx;
    uint32_t hash           = hash_coord(cell_x, cell_y, cell_z);
    point_cell_hashes[idx]  = hash;
}

__global__
void compute_cell_ranges_kernel(
    const uint32_t* __restrict__ point_cell_hashes,
    int* __restrict__ cell_start,
    int* __restrict__ cell_end,
    uint32_t* __restrict__ unique_point_cell_hashes,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    uint32_t hash = point_cell_hashes[idx];

    // Is new cell beginning?
    if (idx == 0 || hash != point_cell_hashes[idx - 1]) {
        int cell_id = idx;
        cell_start[cell_id] = idx;
        unique_point_cell_hashes[cell_id] = hash;

        // End der vorherigen Zelle
        if (idx > 0) {
            int prev_cell_id = idx - 1;
            cell_end[prev_cell_id] = idx;
        }
    }

    // Last cell
    if (idx == N - 1) {
        int last_cell_id = idx;
        cell_end[last_cell_id] = N;
    }
}

__device__
float dist2(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    // The calculation including square root is computationally expensive
    // and the relative comparison results in the same outcome without
    return dx * dx + dy * dy + dz * dz;
}

__global__
void find_k_nearest_neighbors_kernel(
    const float3* __restrict__ points,
    size_t N,
    const uint32_t* __restrict__ point_cell_hashes,
    const int* __restrict__ point_indices,
    const uint32_t* __restrict__ unique_point_cell_hashes,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ num_cells,
    float cell_size,
    int* __restrict__ neighbor_indices
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    // Init neighbor array
    float3 p = points[idx];
    float best_dist[K_NEIGHBORS];
    int best_idx[K_NEIGHBORS];
    for(size_t i = 0; i < K_NEIGHBORS; i++) {
        best_dist[i]=1e30f; best_idx[i]=-1;
    }

    int cell_x = floorf(p.x / cell_size);
    int cell_y = floorf(p.y / cell_size);
    int cell_z = floorf(p.z / cell_size);

    bool done = false;
    int iter = 0;

    // Iterative expanding nearest neighbor search
    while (!done && iter < 10) {
        for(size_t dx = -iter; dx <= iter; dx++)
        for(size_t dy = -iter; dy <= iter; dy++)
        for(size_t dz = -iter; dz <= iter; dz++) {
            uint32_t nhash = hash_coord(
                cell_x + dx,
                cell_y + dy,
                cell_z + dz
            );

            // Linear search
            for(int c = 0 ; c < num_cells[0]; c++){
                if (unique_point_cell_hashes[c] != nhash) {
                    continue;
                }

                for (int j = cell_start[c]; j < cell_end[c]; j++) {
                    int pid = point_indices[j];
                    if (pid == idx) {
                        continue;
                    }

                    float distance  = dist2(p, points[pid]);
                    int max_i       = 0;
                    float max_d     = best_dist[0];

                    for (int k = 1; k < K_NEIGHBORS; k++) {
                        if (best_dist[k] > max_d) {
                            max_d = best_dist[k];
                            max_i = k;
                        }
                    }

                    if (distance < max_d) {
                        best_dist[max_i] = distance;
                        best_idx[max_i] = pid;
                    }
                }
            }
        }

        // Check if all neighbor slots have been filled
        done = true;
        for(int k = 0; k < K_NEIGHBORS; k++) {
            if(best_idx[k] < 0) {
                done = false;
            }
        }
        iter++;
    }

    for(int k = 0; k < K_NEIGHBORS; k++) {
        neighbor_indices[idx * K_NEIGHBORS + k] = best_idx[k];
    }
}

__device__ inline void accumulate_cov(float* cov, float3 d) {
	cov[0] += d.x * d.x; // xx
	cov[1] += d.y * d.y; // yy
	cov[2] += d.z * d.z; // zz
	cov[3] += d.x * d.y; // xy
	cov[4] += d.x * d.z; // xz
	cov[5] += d.y * d.z; // yz
}

__global__
void compute_point_covariances_kernel(
	const float3* __restrict__ points,
	size_t N,
	const int* __restrict__ neighbor_indices,
	float* __restrict__ point_covariances
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= N) {
	    return;
	}

	// Compute local mean
	float3 mean = make_float3(0, 0, 0);
	for (size_t i = 0; i < K_NEIGHBORS; i++) {
		int n_idx = neighbor_indices[idx * K_NEIGHBORS + i];
		float3 q = points[n_idx];
		mean.x += q.x;
		mean.y += q.y;
		mean.z += q.z;
	}

	mean.x /= K_NEIGHBORS;
	mean.y /= K_NEIGHBORS;
	mean.z /= K_NEIGHBORS;

	// Compute covariance
	float cov[6] = {0};
	for (size_t i = 0; i < K_NEIGHBORS; i++) {
		int n_idx = neighbor_indices[idx * K_NEIGHBORS + i];
		float3 q = points[n_idx];
		float3 d = make_float3(q.x - mean.x, q.y - mean.y, q.z - mean.z);
		accumulate_cov(cov, d);
	}

	// Normalize by (k-1) for unbiased covariance
	float norm = 1.0f / max(1, K_NEIGHBORS - 1);
	for (size_t i = 0; i < 6; i++) {
		point_covariances[idx * 6 + i] = cov[i] * norm;
	}
}

__global__
void compute_voxel_means_and_covariances_kernel(
    const float3* __restrict__ points,
	const int* __restrict__ cell_start,
	const int* __restrict__ cell_end,
	const int* num_unique_cells,
	const float* __restrict__ point_covariances,
	float* __restrict__ voxel_centroids,
	float* __restrict__ voxel_covariances
) {
	int cell_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (cell_idx >= num_unique_cells[0]) {
	    return;
	}

	int start = cell_start[cell_idx];
	int end   = cell_end[cell_idx];
	int count = end - start;

	// Calculate voxel centroid
	float3 mean = make_float3(0, 0, 0);
	for (int i = start; i < end; i++) {
		mean.x += points[i].x;
		mean.y += points[i].y;
		mean.z += points[i].z;
	}

	mean.x /= count;
	mean.y /= count;
	mean.z /= count;
	voxel_centroids[cell_idx * 3] = mean.x;
	voxel_centroids[cell_idx * 3 + 1] = mean.y;
	voxel_centroids[cell_idx * 3 + 2] = mean.z;

	// Calculate voxel covariance
	float cov[6] = {0};
	for (int i = start; i < end; i++) {
		for (int j = 0; j < 6; j++) {
			cov[j] += point_covariances[i * 6 + j];
		}
	}
	float norm = 1.0f / (float)count;
	for (int j = 0; j < 6; j++) {
		voxel_covariances[cell_idx * 6 + j] = cov[j] * norm;
	}
}

} // cuda_vgicp
