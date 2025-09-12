#pragma once

#include <vector>
#include <cuda_runtime.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <vector_types.h>

namespace cuda_vgicp {
    void voxelgrid_downsample(
        const float3* __restrict__ points,
        size_t N,
        float leaf_size,
        int* __restrict__ d_num_unique_cells
    );
}
