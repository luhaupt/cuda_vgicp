#pragma once

#include <cmath>
#include <cuda_runtime.h>
#include "voxel_downsample.cuh"

struct NearestNeighbor {
    int idx;
    float dist;
};

namespace cuda_vgicp {

__device__ float mahalanobis_distance(
    const float3& p,
    const float3& mean_q,
    const float cov_p[9],
    const float cov_q[9],
    float eps_reg = 1e-6f
) {
    // Σ = Σ_p + Σ_q + eps*I
    float Sigma[9];
    for(int i=0;i<9;i++) Sigma[i] = cov_p[i] + cov_q[i];
    Sigma[0] += eps_reg; Sigma[4] += eps_reg; Sigma[8] += eps_reg;

    // Inverse 3x3
    float inv[9];
    float det = Sigma[0]*(Sigma[4]*Sigma[8]-Sigma[5]*Sigma[7])
              - Sigma[1]*(Sigma[3]*Sigma[8]-Sigma[5]*Sigma[6])
              + Sigma[2]*(Sigma[3]*Sigma[7]-Sigma[4]*Sigma[6]);

    if(fabsf(det) < 1e-12f) return INFINITY;
    float invDet = 1.0f/det;
    inv[0] =  (Sigma[4]*Sigma[8]-Sigma[5]*Sigma[7])*invDet;
    inv[1] = -(Sigma[1]*Sigma[8]-Sigma[2]*Sigma[7])*invDet;
    inv[2] =  (Sigma[1]*Sigma[5]-Sigma[2]*Sigma[4])*invDet;
    inv[3] = -(Sigma[3]*Sigma[8]-Sigma[5]*Sigma[6])*invDet;
    inv[4] =  (Sigma[0]*Sigma[8]-Sigma[2]*Sigma[6])*invDet;
    inv[5] = -(Sigma[0]*Sigma[5]-Sigma[2]*Sigma[3])*invDet;
    inv[6] =  (Sigma[3]*Sigma[7]-Sigma[4]*Sigma[6])*invDet;
    inv[7] = -(Sigma[0]*Sigma[7]-Sigma[1]*Sigma[6])*invDet;
    inv[8] =  (Sigma[0]*Sigma[4]-Sigma[1]*Sigma[3])*invDet;

    // diff = p - q
    float dx = p.x - mean_q.x;
    float dy = p.y - mean_q.y;
    float dz = p.z - mean_q.z;

    // tmp = inv * diff
    float tx = inv[0]*dx + inv[1]*dy + inv[2]*dz;
    float ty = inv[3]*dx + inv[4]*dy + inv[5]*dz;
    float tz = inv[6]*dx + inv[7]*dy + inv[8]*dz;

    return sqrtf(fmaxf(dx*tx + dy*ty + dz*tz,0.0f));
}

__global__ void nns_kernel(
    const Voxel* srcVoxels, int nSrc,
    const Voxel* tgtVoxels,
    const int* cell_start,
    const int* cell_end,
    const int* point_indices,
    const uint32_t* cell_hashes,
    float cell_size,
    NearestNeighbor* out_nn
) {
    int sidx = blockIdx.x*blockDim.x + threadIdx.x;
    if(sidx >= nSrc) return;

    Voxel src = srcVoxels[sidx];
    NearestNeighbor best;
    best.idx = -1;
    best.dist = INFINITY;

    // Hash Zelle des Source-Centroids
    int ix = floorf(src.mean.x / cell_size);
    int iy = floorf(src.mean.y / cell_size);
    int iz = floorf(src.mean.z / cell_size);

    // Prüfe alle Nachbarzellen (3x3x3)
    for(int dx=-1; dx<=1; dx++)
    for(int dy=-1; dy<=1; dy++)
    for(int dz=-1; dz<=1; dz++){
        int nx = ix+dx, ny = iy+dy, nz = iz+dz;
        uint32_t hash = nx*73856093 ^ ny*19349663 ^ nz*83492791;

        int start = cell_start[hash];
        int end   = cell_end[hash];

        for(int t=start;t<end;t++){
            int tgt_idx = point_indices[t];
            Voxel tgt = tgtVoxels[tgt_idx];

            float d = mahalanobis_distance(src.mean, tgt.mean, src.covariance, tgt.covariance);
            if(d < best.dist){
                best.dist = d;
                best.idx  = tgt_idx;
            }
        }
    }

    out_nn[sidx] = best;
}

} // namespace cuda_vgicp
