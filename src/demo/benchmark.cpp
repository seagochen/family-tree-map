/**
 * @file benchmark.cpp
 * @brief Fire Detection API 性能基准测试程序
 *
 * 对单张图片执行多次推理，计算平均执行时间。
 *
 * 使用方法:
 *   ./fire_detection_benchmark <yolo_engine> <convlstm_engine> <image_path> [iterations]
 *
 * 示例:
 *   ./fire_detection_benchmark segment.engine convlstm.engine test.jpg
 *   ./fire_detection_benchmark segment.engine convlstm.engine test.jpg 500
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <numeric>
#include <cmath>
#include <opencv2/opencv.hpp>
#include "fire_detection_api.h"

int main(int argc, char* argv[]) {
    // 检查参数
    if (argc < 4) {
        std::cout << "Fire Detection Benchmark v" << fire_detection::getVersion() << "\n\n";
        std::cout << "Usage: " << argv[0] << " <yolo_engine> <convlstm_engine> <image_path> [iterations]\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " segment.engine convlstm.engine test.jpg\n";
        std::cout << "  " << argv[0] << " segment.engine convlstm.engine test.jpg 500\n";
        return 1;
    }

    const char* yolo_engine = argv[1];
    const char* convlstm_engine = argv[2];
    const char* image_path = argv[3];
    int iterations = (argc >= 5) ? std::atoi(argv[4]) : 1000;

    if (iterations <= 0) {
        std::cerr << "Error: iterations must be positive\n";
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "  Fire Detection Benchmark v" << fire_detection::getVersion() << "\n";
    std::cout << "========================================\n\n";

    // 读取图片
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        std::cerr << "Error: Failed to load image: " << image_path << "\n";
        return 1;
    }

    std::cout << "Image info:\n";
    std::cout << "  Path: " << image_path << "\n";
    std::cout << "  Size: " << image.cols << "x" << image.rows << "\n";
    std::cout << "  Channels: " << image.channels() << "\n\n";

    // 配置
    fire_detection::Config config;
    config.yolo_engine_path = yolo_engine;
    config.convlstm_engine_path = convlstm_engine;
    config.confidence_threshold = 0.5f;
    config.convlstm_seq_len = 10;

    std::cout << "Configuration:\n";
    std::cout << "  YOLO engine: " << config.yolo_engine_path << "\n";
    std::cout << "  ConvLSTM engine: " << config.convlstm_engine_path << "\n";
    std::cout << "  Confidence: " << config.confidence_threshold << "\n";
    std::cout << "  Iterations: " << iterations << "\n\n";

    // 初始化检测器
    fire_detection::FireDetector detector;
    if (!detector.initialize(config)) {
        std::cerr << "Error: Failed to initialize detector\n";
        return 1;
    }

    std::cout << "Warming up...\n";
    // 预热（前10次不计入统计）
    for (int i = 0; i < 10; i++) {
        detector.processFrame(image);
    }

    std::cout << "Running benchmark (" << iterations << " iterations)...\n\n";

    // 存储每次迭代的时间
    std::vector<double> times;
    times.reserve(iterations);

    auto total_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = detector.processFrame(image);

        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed_ms);

        // 每100次显示进度
        if ((i + 1) % 100 == 0) {
            std::cout << "  Progress: " << (i + 1) << "/" << iterations << "\r" << std::flush;
        }
    }

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();

    std::cout << "\n\n";

    // 计算统计数据
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    // 计算标准差
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / times.size());

    // 找最小和最大值
    auto minmax = std::minmax_element(times.begin(), times.end());
    double min_time = *minmax.first;
    double max_time = *minmax.second;

    // 计算FPS
    double avg_fps = 1000.0 / mean;

    // 输出结果
    std::cout << "========================================\n";
    std::cout << "  Benchmark Results\n";
    std::cout << "========================================\n\n";
    std::cout << "  Total time:     " << std::fixed << std::setprecision(2) << total_time_ms << " ms\n";
    std::cout << "  Iterations:     " << iterations << "\n\n";
    std::cout << "  Average time:   " << std::fixed << std::setprecision(3) << mean << " ms\n";
    std::cout << "  Std deviation:  " << std::fixed << std::setprecision(3) << std_dev << " ms\n";
    std::cout << "  Min time:       " << std::fixed << std::setprecision(3) << min_time << " ms\n";
    std::cout << "  Max time:       " << std::fixed << std::setprecision(3) << max_time << " ms\n\n";
    std::cout << "  Average FPS:    " << std::fixed << std::setprecision(2) << avg_fps << "\n";
    std::cout << "========================================\n";

    return 0;
}
