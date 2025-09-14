#include <fmt/format.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../../include/benchmark.hpp"
#include "../../include/point_cloud.hpp"
#include "../../include/voxel_downsample.cuh"

#define K_NEIGHBORS 20

namespace cuda_vgicp {

template <typename PointCloudPtr>
void benchmark(const std::vector<PointCloudPtr> &raw_points, double leaf_size) {
    Stopwatch sw;
    Summarizer times;
    Summarizer num_points;

    for (const auto &points : raw_points) {
        float3* d_points;
        int* d_point_indices;
        int* d_point_indices_sorted;
        uint32_t* d_point_cell_hashes;
        uint32_t* d_point_cell_hashes_sorted;
        uint32_t* d_unique_point_cell_hashes;
        int* d_cell_start;
        int* d_cell_end;
        int* d_points_per_cell;
        int* d_neighbor_indices;
        float* d_neighbor_distances;
        float* d_point_covariances;
        float* d_voxel_centroids;
        float* d_voxel_covariances;
        int* d_num_unique_cells;
        size_t N = points->size();
        cudaMalloc(&d_points, N * sizeof(float3));
        cudaMemcpy(d_points, points->data(), N * sizeof(float3), cudaMemcpyHostToDevice);
        cudaMalloc(&d_num_unique_cells, sizeof(int));
        cudaMalloc(&d_point_indices, N * sizeof(int));
        cudaMalloc(&d_point_indices_sorted, N * sizeof(int));
        cudaMalloc(&d_point_cell_hashes, N * sizeof(uint32_t));
        cudaMalloc(&d_point_cell_hashes_sorted, N * sizeof(uint32_t));
        cudaMalloc(&d_unique_point_cell_hashes, N * sizeof(uint32_t));
        cudaMalloc(&d_cell_start, N * sizeof(int));
        cudaMalloc(&d_cell_end, N * sizeof(int));
        cudaMalloc(&d_points_per_cell, N * sizeof(int));
        cudaMalloc(&d_neighbor_indices, N * K_NEIGHBORS * sizeof(int));
        cudaMalloc(&d_neighbor_distances, N * K_NEIGHBORS * sizeof(float));
        cudaMalloc(&d_point_covariances, N * 6 * sizeof(float));
        cudaMalloc(&d_voxel_centroids, N * 3 * sizeof(float));
        cudaMalloc(&d_voxel_covariances, N * 6 * sizeof(float));

        sw.start();
        cuda_vgicp::voxelgrid_downsample(
            d_points,
            N,
            leaf_size,
            d_num_unique_cells,
            d_point_indices,
            d_point_indices_sorted,
            d_point_cell_hashes,
            d_point_cell_hashes_sorted,
            d_unique_point_cell_hashes,
            d_cell_start,
            d_cell_end,
            d_points_per_cell,
            d_neighbor_indices,
            d_neighbor_distances,
            d_point_covariances,
            d_voxel_centroids,
            d_voxel_covariances
        );
        sw.lap();
        times.push(sw.msec());
        int h_voxel = 0;
        cudaMemcpy(&h_voxel, d_num_unique_cells, sizeof(int), cudaMemcpyDeviceToHost);
        num_points.push(h_voxel);
        cudaFree(d_num_unique_cells);
        cudaFree(d_points);
        cudaFree(d_point_indices);
        cudaFree(d_point_indices_sorted);
        cudaFree(d_point_cell_hashes);
        cudaFree(d_point_cell_hashes_sorted);
        cudaFree(d_unique_point_cell_hashes);
        cudaFree(d_cell_start);
        cudaFree(d_cell_end);
        cudaFree(d_points_per_cell);
        cudaFree(d_neighbor_indices);
        cudaFree(d_neighbor_distances);
        cudaFree(d_point_covariances);
        cudaFree(d_voxel_centroids);
        cudaFree(d_voxel_covariances);
    }

    std::cout << fmt::format("{} [msec/scan]   {} [points]", times.str(), num_points.str()) << std::endl;
}

} // namespace cuda_vgicp

int main(int argc, char **argv) {
    using namespace cuda_vgicp;

    if (argc < 2) {
        std::cout << "usage: downsampling_benchmark <dataset_path> (--max_num_frames 1000)"
                << std::endl;
        return 0;
    }

    const std::string dataset_path = argv[1];
    size_t max_num_frames = 1000;

    for (int i = 1; i < argc; i++) {
        const std::string arg(argv[i]);
        if (arg == "--max_num_frames") {
        max_num_frames = std::stoul(argv[i + 1]);
        } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
        std::cerr << "unknown option: " << arg << std::endl;
        return 1;
        }
    }

    std::cout << "dataset_path=" << dataset_path << std::endl;
    std::cout << "max_num_frames=" << max_num_frames << std::endl;

    KittiDataset kitti(dataset_path, max_num_frames);
    std::cout << "num_frames=" << kitti.points.size() << std::endl;
    std::cout << fmt::format(
                    "num_points={} [points]",
                    summarize(kitti.points,
                                [](const auto &pts) { return pts.size(); }))
                << std::endl;

        const auto points = kitti.convert<pcl::PointCloud<pcl::PointXYZ>>(true);
        std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> point_vec{ points };

    // Warming up
    std::cout << "---" << std::endl;
    std::cout << "leaf_size=0.5(warmup)" << std::endl;
    std::cout << fmt::format("{:25}: ", "cuda_vgicp") << std::flush;
    benchmark(point_vec, 0.5);

    // Benchmark
    for (double leaf_size = 0.1; leaf_size <= 1.51; leaf_size += 0.1) {
        std::cout << "---" << std::endl;
        std::cout << "leaf_size=" << leaf_size << std::endl;
        std::cout << fmt::format("{:25}: ", "cuda_vgicp") << std::flush;
        benchmark(point_vec, leaf_size);
    }

  return 0;
}
