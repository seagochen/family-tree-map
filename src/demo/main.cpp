/**
 * @file main.cpp
 * @brief Fire Detection API 示例程序
 *
 * 演示如何使用 fire_detection::FireDetector API 进行火灾检测。
 *
 * 使用方法:
 *   ./fire_detection_demo <yolo_engine> <convlstm_engine> <video_path>
 *
 * 示例:
 *   ./fire_detection_demo segment.engine convlstm.engine test.mp4
 */

#include <iostream>
#include <opencv2/opencv.hpp>
#include "fire_detection_api.h"

int main(int argc, char* argv[]) {
    // 检查参数
    if (argc < 4) {
        std::cout << "Fire Detection Demo v" << fire_detection::getVersion() << "\n\n";
        std::cout << "Usage: " << argv[0] << " <yolo_engine> <convlstm_engine> <video_path>\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " segment.engine convlstm.engine test.mp4\n";
        return 1;
    }

    const char* yolo_engine = argv[1];
    const char* convlstm_engine = argv[2];
    const char* video_path = argv[3];

    std::cout << "========================================\n";
    std::cout << "  Fire Detection Demo v" << fire_detection::getVersion() << "\n";
    std::cout << "========================================\n\n";

    // 配置
    fire_detection::Config config;
    config.yolo_engine_path = yolo_engine;
    config.convlstm_engine_path = convlstm_engine;
    config.confidence_threshold = 0.5f;
    config.convlstm_seq_len = 10;

    std::cout << "Configuration:\n";
    std::cout << "  YOLO engine: " << config.yolo_engine_path << "\n";
    std::cout << "  ConvLSTM engine: " << config.convlstm_engine_path << "\n";
    std::cout << "  Video: " << video_path << "\n";
    std::cout << "  Confidence: " << config.confidence_threshold << "\n";
    std::cout << "  Sequence length: " << config.convlstm_seq_len << "\n\n";

    // 初始化检测器
    fire_detection::FireDetector detector;
    if (!detector.initialize(config)) {
        std::cerr << "Error: Failed to initialize detector\n";
        return 1;
    }

    // 打开视频
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open video: " << video_path << "\n";
        return 1;
    }

    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Video info:\n";
    std::cout << "  Frames: " << total_frames << "\n";
    std::cout << "  FPS: " << fps << "\n\n";

    // 创建窗口
    const std::string window_name = "Fire Detection Demo";
    cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);

    std::cout << "Processing... Press 'q' to quit\n\n";

    cv::Mat frame;
    int frame_idx = 0;

    while (cap.read(frame)) {
        // 处理帧
        auto result = detector.processFrame(frame);

        // 在帧上绘制信息
        cv::Mat display = result.frame.roi_frame.clone();

        // 帧信息
        cv::putText(display,
            "Frame: " + std::to_string(frame_idx) + "/" + std::to_string(total_frames),
            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);

        // 检测状态
        std::string status;
        if (result.frame.has_fire && result.frame.has_smoke) {
            status = "FIRE + SMOKE";
        } else if (result.frame.has_fire) {
            status = "FIRE";
        } else if (result.frame.has_smoke) {
            status = "SMOKE";
        } else {
            status = "No detection";
        }

        cv::Scalar status_color = (result.hasFire() || result.hasSmoke())
            ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

        cv::putText(display, "Status: " + status,
            cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);

        // 缓冲区状态
        cv::putText(display,
            "Buffer: " + std::to_string(detector.getBufferSize()) + "/" +
            std::to_string(detector.getBufferCapacity()),
            cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);

        // 绘制检测框（BoundingBox）
        for (const auto& bbox : result.frame.detections) {
            // 计算矩形坐标
            cv::Rect rect(
                static_cast<int>(bbox.x1),
                static_cast<int>(bbox.y1),
                static_cast<int>(bbox.x2 - bbox.x1),
                static_cast<int>(bbox.y2 - bbox.y1)
            );

            // 根据类别选择颜色：火焰=红色，烟雾=黄色
            cv::Scalar color;
            if (bbox.class_id == fire_detection::DetectionClass::FIRE) {
                color = cv::Scalar(0, 0, 255);      // 红色 (BGR)
            } else if (bbox.class_id == fire_detection::DetectionClass::SMOKE) {
                color = cv::Scalar(0, 255, 255);    // 黄色 (BGR)
            } else {
                color = cv::Scalar(0, 255, 0);      // 绿色 (BGR)
            }

            // 绘制矩形框
            cv::rectangle(display, rect, color, 2);

            // 绘制标签（类别 + 置信度）
            std::string label = std::string(fire_detection::detectionClassToString(bbox.class_id)) +
                                " " + std::to_string(static_cast<int>(bbox.confidence * 100)) + "%";
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

            // 标签背景
            cv::rectangle(display,
                cv::Point(rect.x, rect.y - text_size.height - 5),
                cv::Point(rect.x + text_size.width, rect.y),
                color, cv::FILLED);

            // 标签文字
            cv::putText(display, label,
                cv::Point(rect.x, rect.y - 3),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        }

        // 显示检测框数量
        cv::putText(display,
            "Detections: " + std::to_string(result.frame.detections.size()),
            cv::Point(10, 120), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 200, 0), 2);

        // 时序分类结果
        if (result.temporal_valid) {
            cv::Scalar cls_color = fire_detection::temporalClassToColor(result.temporal_class);
            std::string cls_text = std::string("Temporal: ") +
                fire_detection::temporalClassToString(result.temporal_class) +
                " (" + std::to_string(static_cast<int>(result.temporal_confidence * 100)) + "%)";

            cv::putText(display, cls_text,
                cv::Point(10, 150), cv::FONT_HERSHEY_SIMPLEX, 0.7, cls_color, 2);

            // 火灾警报
            if (result.isFireAlert()) {
                cv::putText(display, "!!! FIRE ALERT !!!",
                    cv::Point(display.cols / 2 - 150, 50),
                    cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
            }
        }

        // 显示
        cv::imshow(window_name, display);

        // 键盘控制
        int key = cv::waitKey(1);
        if (key == 'q' || key == 27) {
            std::cout << "Quit requested\n";
            break;
        }

        frame_idx++;
    }

    // 清理
    cap.release();
    cv::destroyAllWindows();

    std::cout << "\n========================================\n";
    std::cout << "  Finished\n";
    std::cout << "  Total frames: " << frame_idx << "\n";
    std::cout << "========================================\n";

    return 0;
}
