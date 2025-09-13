#pragma once

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <sys/cdefs.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <vector_types.h>

#define THREADS_PER_BLOCK 256
#define K_NEIGHBORS 20

namespace cuda_vgicp {

__device__ __forceinline__ void heapify_down(
    int* neighbor_indices,
    float* neighbor_distances,
    int k
) {
    int largest = k;
	int left    = 2 * k + 1;
	int right   = 2 * k + 2;

	if (left < K_NEIGHBORS && neighbor_distances[left] > neighbor_distances[largest]) {
		largest = left;
	}
	if (right < K_NEIGHBORS && neighbor_distances[right] > neighbor_distances[largest]) {
		largest = right;
	}
	if (largest != k) {
	    float tmp_d = neighbor_distances[k];
		neighbor_distances[k] = neighbor_distances[largest];
		neighbor_distances[largest] = tmp_d;

        int tmp_i = neighbor_indices[k];
        neighbor_indices[k] = neighbor_indices[largest];
        neighbor_indices[largest] = tmp_i;

		heapify_down(neighbor_indices, neighbor_distances, largest);
	}
}

__device__ __forceinline__ void heap_replace_root(
    int* neighbor_indices,
    float* neighbor_distances,
    int idx,
    float dist
) {
    neighbor_distances[0] = dist;
	neighbor_indices[0] = idx;
	heapify_down(neighbor_indices, neighbor_distances, 0);
}

__device__ __forceinline__ uint32_t hash_coord(int3 coord) {
    return coord.x * 73856093u ^ coord.y * 19349663u ^ coord.z * 83492791u;
}

__device__ __forceinline__ int3 get_cell_index(float3 point, float cell_size) {
    return make_int3(
        floorf(point.x / cell_size),
        floorf(point.y / cell_size),
        floorf(point.z / cell_size)
    );
}

__device__ __forceinline__ int find_hash_index(uint32_t hash, const uint32_t* hashes, int size) {
    int left = 0, right = size - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (hashes[mid] == hash) return mid;
        if (hashes[mid] < hash) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

__device__ __forceinline__ float dist2(float3 a, float3 b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    float dz = a.z - b.z;

    // The calculation including square root is computationally expensive
    // and the relative comparison results in the same outcome without
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ void accumulate_cov(float* cov, float3 d) {
	cov[0] += d.x * d.x; // xx
	cov[1] += d.y * d.y; // yy
	cov[2] += d.z * d.z; // zz
	cov[3] += d.x * d.y; // xy
	cov[4] += d.x * d.z; // xz
	cov[5] += d.y * d.z; // yz
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

    int3 cell_index = get_cell_index(p, cell_size);

    // Set point index and cell hash (of point)
    point_indices[idx]      = idx;
    uint32_t hash           = hash_coord(cell_index);
    point_cell_hashes[idx]  = hash;
}

__global__
void compute_cell_ranges_kernel(
    int* __restrict__ cell_start,
    int* __restrict__ points_per_cell,
    int* __restrict__ cell_end,
    int* __restrict__ num_unique_cells
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_unique_cells[0]) {
        return;
    }

    cell_end[idx] = cell_start[idx] + points_per_cell[idx];
}

__global__
void find_k_nearest_neighbors_kernel(
    const float3* __restrict__ points,
    const size_t N,
    const int* __restrict__ point_indices,
    const uint32_t* __restrict__ unique_point_cell_hashes,
    const int* __restrict__ cell_start,
    const int* __restrict__ cell_end,
    const int* __restrict__ num_unique_cells,
    const int* __restrict__ points_per_cell,
    float cell_size,
    int* __restrict__ neighbor_indices,
    float* __restrict__ neighbor_distances
) {
    extern __shared__ __align__(16) unsigned char shared_mem[];
    float* shared_distances = reinterpret_cast<float*>(shared_mem);
    size_t offset = blockDim.x * K_NEIGHBORS * sizeof(float);
    offset = (offset + sizeof(int) - 1) & ~(sizeof(int) - 1);
    int* shared_indices = reinterpret_cast<int*>(shared_mem + offset);

    float* local_distances = &shared_distances[threadIdx.x * K_NEIGHBORS];
    int*   local_indices   = &shared_indices[threadIdx.x * K_NEIGHBORS];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) {
        return;
    }

    float3 query_point              = points[idx];
    int3 query_cell_index           = get_cell_index(query_point, cell_size);
    int existing_query_cell_index   = find_hash_index(
        hash_coord(query_cell_index),
        unique_point_cell_hashes,
        num_unique_cells[0]
    );

    if(existing_query_cell_index < 0 || existing_query_cell_index >= num_unique_cells[0])
        return;

    #pragma unroll
    for (int i = 0; i < K_NEIGHBORS; i++) {
        local_distances[i] = FLT_MAX;
        local_indices[i] = -1;
    }

    if (existing_query_cell_index != -1 && points_per_cell[existing_query_cell_index] >= K_NEIGHBORS + 1) {
        int start = cell_start[existing_query_cell_index];
        int end   = cell_end[existing_query_cell_index];

        for(int i = start; i < end; i++){
            int pid = point_indices[i];
            if(pid == idx) {
                continue;
            }

            float3 p        = points[pid];
            float distance  = dist2(query_point, p);

            if(!isfinite(distance)) continue;

            if(distance < local_distances[0]){
                heap_replace_root(local_indices, local_distances, pid, distance);
            }
        }
    } else {
        bool done = false;
        int iter = 0;

        // Iterative expanding nearest neighbor search
        while (!done && iter < 10) {
            for(int dx = -1; dx <= 1; dx++)
            for(int dy = -1; dy <= 1; dy++)
            for(int dz = -1; dz <= 1; dz++) {
                int3 neighbor_cell_index = make_int3(
                    query_cell_index.x + dx,
                    query_cell_index.y + dy,
                    query_cell_index.z + dz
                );

                uint32_t neighbor_cell_hash = hash_coord(neighbor_cell_index);
                int found_index = find_hash_index(
                    neighbor_cell_hash,
                    unique_point_cell_hashes,
                    num_unique_cells[0]
                );

                if (found_index < 0 || found_index >= num_unique_cells[0]) {
                    continue;
                }

                // if (idx == 50) {
                //     printf("Thread: 50, voxel_index: %d\n", found_index);
                // }

                int start = cell_start[found_index];
                int end   = cell_end[found_index];
                int count = end - start;

                if (count <= 0) {
                    continue;
                }

                // if (idx == 50) {
                //     printf("Start: %d, End: %d\n", start, end);
                // }

                for(int i = start; i < end; i++){
                    int pid = point_indices[i];
                    if(pid == idx) {
                        continue;
                    }

                    float3 p = points[pid];
                    float distance = dist2(query_point, p);

                    // if (idx == 50) {
                    //     printf("Distance: %d\n", distance);
                    // }

                    if(!isfinite(distance)) continue;

                    if(distance < local_distances[0]){
                        heap_replace_root(local_indices, local_distances, pid, distance);
                    }
                }
            }

            done = local_distances[0] < FLT_MAX;
            iter++;
        }
    }

    for (int i = 0; i < K_NEIGHBORS; i++) {
        neighbor_distances[idx * K_NEIGHBORS + i] = local_distances[i];
        neighbor_indices[idx * K_NEIGHBORS + i]   = local_indices[i];
    }
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
