#include "../../include/benchmark.hpp"
#include "../../include/point_cloud.hpp"
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "../../include/hash_grid.cuh"
#include "../../include/voxel_downsample.h"

int main(int argc, char **argv) {
    // Input parameters
    if (argc < 2) {
        std::cout << "USAGE: ./hash_grid_benchmark <dataset_path>" << std::endl;
        return 0;
    }

    const std::string dataset_path = argv[1];
    int num_trials = 100;

    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--num_trials") {
        num_trials = std::stoi(argv[i + 1]);
        } else if (arg.size() >= 2 && arg.substr(0, 2) == "--") {
        std::cerr << "unknown option: " << arg << std::endl;
        return 1;
        }
    }

    std::cout << "dataset_path=" << dataset_path << std::endl;
    std::cout << "num_trials=" << num_trials << std::endl;
    std::cout << "method=cuda_vgicp" << std::endl;

    // Get dataset and convert to vector
    cuda_vgicp::KittiDataset kitti(dataset_path, 1000);
    const auto raw_points = kitti.convert<pcl::PointCloud<pcl::PointXYZ>>(true);
    std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> points{ raw_points };
    std::cout << "num_raw_points=" << points[0]->size() << std::endl;

    // Search leaf size for min amount of points in point cloud
    const auto search_voxel_resolution = [&](size_t target_num_points) {
        std::pair<double, size_t> left = {
            0.1, cuda_vgicp::voxelgrid_downsample(points[0], 0.1).size()};
        std::pair<double, size_t> right = {
            1.0, cuda_vgicp::voxelgrid_downsample(points[0], 1.0).size()};

        for (int i = 0; i < 20; i++) {
            if (left.second < target_num_points) {
                left.first *= 0.1;
                left.second = cuda_vgicp::voxelgrid_downsample(points[0], left.first).size();
                continue;
            }
            if (right.second > target_num_points) {
                right.first *= 10.0;
                right.second = cuda_vgicp::voxelgrid_downsample(points[0], right.first).size();
                continue;
            }

            const double mid = (left.first + right.first) * 0.5;
            const size_t mid_num_points = cuda_vgicp::voxelgrid_downsample(points[0], mid).size();

            if (std::abs(1.0 - static_cast<double>(mid_num_points) / target_num_points) < 0.001) {
                return mid;
            }

            if (mid_num_points > target_num_points) {
                left = {mid, mid_num_points};
            } else {
                right = {mid, mid_num_points};
            }
        }

        return (left.first + right.first) * 0.5;
    };

    std::cout << "---" << std::endl;

    std::vector<double> downsampling_resolutions;
    std::vector<float3> downsampled;
    for (double target = 1.0; target > 0.05; target -= 0.1) {
        const double downsampling_resolution = search_voxel_resolution(points[0]->size() * target);
        downsampling_resolutions.emplace_back(downsampling_resolution);
        downsampled = cuda_vgicp::voxelgrid_downsample(points[0], downsampling_resolution);
        std::cout << "downsampling_resolution=" << downsampling_resolution << std::endl;
        std::cout << "num_points=" << downsampled.size() << std::endl;
    }

    std::cout << "---" << std::endl;

    // warmup
    auto t1 = std::chrono::high_resolution_clock::now();
    while (std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::high_resolution_clock::now() - t1)
                .count() < 1) {
                    cuda_vgicp::SpatialHashGrid hash_grid;
                    thrust::device_vector<float3> d_points = downsampled;
                    hash_grid.build(d_points, downsampling_resolutions[0]);
    }

    for (size_t i = 0; i < downsampling_resolutions.size(); i++) {
        cudaEvent_t start, stop;
        cuda_vgicp::SpatialHashGrid hash_grid;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        thrust::device_vector<float3> d_points = downsampled;
        hash_grid.build(d_points, downsampling_resolutions[i]);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0.0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "hash_grid_times=" << milliseconds << " [msec]" << std::endl;
    }

    return 0;
}
