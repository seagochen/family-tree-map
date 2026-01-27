/**
 * @file main.cpp
 * @brief 火灾检测流水线，支持实时ROI可视化
 *
 * 本程序功能：
 * 1. 使用 cudatt 的 TrtEngineMultiTs 加载 TensorRT 分割引擎
 * 2. 逐帧读取视频文件
 * 3. 使用 ROI 提取器将非ROI区域遮罩为黑色
 * 4. 实时显示处理后的帧
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// cudatt 头文件
#include <trt_engine/trt_engine.h>
#include <tensors/tensor.hpp>

// 本地头文件
#include "yolov8_postprocess.h"
#include "roi_extractor.h"

// YOLOv8-seg 模型常量
constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int NUM_CLASSES = 3;  // 火焰、人、烟雾
constexpr int MASK_PROTO_H = 160;
constexpr int MASK_PROTO_W = 160;
constexpr int NUM_MASK_COEFFS = 32;

/**
 * @brief 命令行参数解析器（类似 Python argparse）
 */
struct Args {
    std::string video_path;
    std::string engine_path;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.45f;
    bool display_enabled = true;
    bool help_requested = false;
    bool parse_error = false;
    std::string error_message;
};

void printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " --video <path> --engine <path> [options]\n"
              << "\nRequired arguments:\n"
              << "  --video PATH      Path to input video file\n"
              << "  --engine PATH     Path to segment.engine file\n"
              << "\nOptions:\n"
              << "  --confidence FLOAT  Detection confidence threshold (default: 0.5)\n"
              << "  --iou FLOAT         NMS IoU threshold (default: 0.45)\n"
              << "  --no-display        Disable display (for headless mode)\n"
              << "  --help              Show this help message\n"
              << std::endl;
}

/**
 * @brief 解析命令行参数
 * @param argc 参数数量
 * @param argv 参数值数组
 * @return 解析后的参数结构体
 */
Args parseArgs(int argc, char* argv[]) {
    Args args;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            args.help_requested = true;
            return args;
        } else if (arg == "--video") {
            if (i + 1 < argc) {
                args.video_path = argv[++i];
            } else {
                args.parse_error = true;
                args.error_message = "--video requires a path argument";
                return args;
            }
        } else if (arg == "--engine") {
            if (i + 1 < argc) {
                args.engine_path = argv[++i];
            } else {
                args.parse_error = true;
                args.error_message = "--engine requires a path argument";
                return args;
            }
        } else if (arg == "--confidence") {
            if (i + 1 < argc) {
                try {
                    args.confidence_threshold = std::stof(argv[++i]);
                } catch (const std::exception& e) {
                    args.parse_error = true;
                    args.error_message = "--confidence requires a valid float value";
                    return args;
                }
            } else {
                args.parse_error = true;
                args.error_message = "--confidence requires a float argument";
                return args;
            }
        } else if (arg == "--iou") {
            if (i + 1 < argc) {
                try {
                    args.iou_threshold = std::stof(argv[++i]);
                } catch (const std::exception& e) {
                    args.parse_error = true;
                    args.error_message = "--iou requires a valid float value";
                    return args;
                }
            } else {
                args.parse_error = true;
                args.error_message = "--iou requires a float argument";
                return args;
            }
        } else if (arg == "--no-display") {
            args.display_enabled = false;
        } else {
            args.parse_error = true;
            args.error_message = "Unknown argument: " + arg;
            return args;
        }
    }

    // 验证必需参数
    if (args.video_path.empty()) {
        args.parse_error = true;
        args.error_message = "--video is required";
        return args;
    }
    if (args.engine_path.empty()) {
        args.parse_error = true;
        args.error_message = "--engine is required";
        return args;
    }

    return args;
}

/**
 * @brief 预处理帧以供 YOLOv8 推理
 * @param frame 输入的 BGR 帧
 * @param input_tensor 要填充的输出张量
 * @return Letterbox 参数 (scale, pad_x, pad_y)
 */
std::tuple<float, int, int> preprocessFrame(
    const cv::Mat& frame,
    Tensor<float>& input_tensor
) {
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    // 计算 letterbox 参数
    float scale = std::min(
        static_cast<float>(INPUT_WIDTH) / orig_w,
        static_cast<float>(INPUT_HEIGHT) / orig_h
    );
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    int pad_x = (INPUT_WIDTH - new_w) / 2;
    int pad_y = (INPUT_HEIGHT - new_h) / 2;

    // 缩放并添加 letterbox 边框
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    cv::Mat padded(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // BGR 转 RGB 并归一化到 [0, 1]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 转换为浮点数并归一化
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 转换为 TensorRT 所需的 CHW 格式 (NCHW)
    std::vector<float> input_data(3 * INPUT_HEIGHT * INPUT_WIDTH);
    const float* src = reinterpret_cast<float*>(float_img.data);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < INPUT_HEIGHT; ++h) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                int src_idx = (h * INPUT_WIDTH + w) * 3 + c;
                int dst_idx = c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;
                input_data[dst_idx] = src[src_idx];
            }
        }
    }

    input_tensor.copyFromVector(input_data);

    return {scale, pad_x, pad_y};
}

/**
 * @brief 运行检测并创建 ROI 检测器回调
 */
class SegmentDetector {
public:
    SegmentDetector(
        TrtEngineMultiTs& engine,
        Tensor<float>& input_tensor,
        Tensor<float>& output0_tensor,
        Tensor<float>& output1_tensor,
        float conf_threshold = 0.5f,
        float iou_threshold = 0.45f
    ) : engine_(engine),
        input_tensor_(input_tensor),
        output0_tensor_(output0_tensor),
        output1_tensor_(output1_tensor),
        postprocessor_(NUM_CLASSES, conf_threshold, iou_threshold) {}

    std::vector<roi_extractor::ROIDetection> detect(
        const cv::Mat& frame,
        float conf_threshold
    ) {
        // 预处理
        auto [scale, pad_x, pad_y] = preprocessFrame(frame, input_tensor_);

        // 运行推理
        std::vector<Tensor<float>*> inputs = {&input_tensor_};
        std::vector<Tensor<float>*> outputs = {&output0_tensor_, &output1_tensor_};

        if (!engine_.infer(inputs, outputs)) {
            std::cerr << "Inference failed!" << std::endl;
            return {};
        }

        // 将输出复制到主机
        std::vector<float> output0_data, output1_data;
        output0_tensor_.copyToVector(output0_data);
        output1_tensor_.copyToVector(output1_data);

        // 后处理
        postprocessor_.setConfThreshold(conf_threshold);
        std::vector<Detection> detections = postprocessor_.process(
            output0_data, output1_data,
            frame.cols, frame.rows,
            INPUT_WIDTH, INPUT_HEIGHT
        );

        // 将 Detection 转换为 ROIDetection
        std::vector<cv::Mat> masks;
        for (const auto& det : detections) {
            cv::Mat mask(det.mask_height, det.mask_width, CV_32FC1);
            std::memcpy(mask.data, det.mask.data(), det.mask.size() * sizeof(float));

            // 将掩膜缩放到帧大小
            cv::Mat resized_mask;
            cv::resize(mask, resized_mask, cv::Size(frame.cols, frame.rows));
            masks.push_back(resized_mask > 0.5f);
        }

        return roi_extractor::extractDetectionsFromYolo(
            detections, masks,
            cv::Size(frame.cols, frame.rows),
            roi_extractor::TARGET_CLASSES,
            conf_threshold
        );
    }

private:
    TrtEngineMultiTs& engine_;
    Tensor<float>& input_tensor_;
    Tensor<float>& output0_tensor_;
    Tensor<float>& output1_tensor_;
    YOLOv8PostProcessor postprocessor_;
};

int main(int argc, char* argv[]) {
    // 解析命令行参数
    Args args = parseArgs(argc, argv);

    if (args.help_requested) {
        printUsage(argv[0]);
        return 0;
    }

    if (args.parse_error) {
        std::cerr << "Error: " << args.error_message << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "  Fire Detection ROI Visualization\n";
    std::cout << "========================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Video: " << args.video_path << "\n";
    std::cout << "  Engine: " << args.engine_path << "\n";
    std::cout << "  Confidence threshold: " << args.confidence_threshold << "\n";
    std::cout << "  IoU threshold: " << args.iou_threshold << "\n";
    std::cout << "  Display: " << (args.display_enabled ? "enabled" : "disabled") << "\n\n";

    // 初始化 CUDA
    std::cout << "Initializing CUDA...\n";
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "Error: No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "  Using GPU: " << prop.name << "\n";
    std::cout << "  Compute capability: " << prop.major << "." << prop.minor << "\n\n";

    cudaSetDevice(0);

    // 加载 TensorRT 引擎
    std::cout << "Loading TensorRT engine...\n";
    TrtEngineMultiTs engine;
    if (!engine.loadFromFile(args.engine_path)) {
        std::cerr << "Error: Failed to load engine from " << args.engine_path << std::endl;
        return 1;
    }
    std::cout << "  Engine loaded successfully\n\n";

    // 创建执行上下文
    // YOLOv8-seg 输出：
    // - output0: (1, 116, 8400) - 检测结果 [x,y,w,h + 类别数 + 32个掩膜系数]
    // - output1: (1, 32, 160, 160) - 掩膜原型
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_dims = {nvinfer1::Dims4{1, 3, INPUT_HEIGHT, INPUT_WIDTH}};
    std::vector<std::string> output_names = {"output0", "output1"};

    if (!engine.createContext(input_names, input_dims, output_names)) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return 1;
    }
    std::cout << "  Execution context created\n\n";

    // 分配张量
    Tensor<float> input_tensor(TensorType::FLOAT32, 1, 3, INPUT_HEIGHT, INPUT_WIDTH);
    // output0: (1, 116, 8400) 其中 116 = 4 + 类别数 + 32个掩膜系数
    Tensor<float> output0_tensor(TensorType::FLOAT32, 1, 4 + NUM_CLASSES + NUM_MASK_COEFFS, 8400);
    // output1: (1, 32, 160, 160) 掩膜原型
    Tensor<float> output1_tensor(TensorType::FLOAT32, 1, NUM_MASK_COEFFS, MASK_PROTO_H, MASK_PROTO_W);

    // 打开视频文件
    std::cout << "Opening video file...\n";
    cv::VideoCapture cap(args.video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open video: " << args.video_path << std::endl;
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "  Resolution: " << frame_width << "x" << frame_height << "\n";
    std::cout << "  FPS: " << fps << "\n";
    std::cout << "  Total frames: " << total_frames << "\n\n";

    // 创建检测器
    SegmentDetector detector(
        engine, input_tensor, output0_tensor, output1_tensor,
        args.confidence_threshold, args.iou_threshold
    );

    // 创建带检测器回调的 ROI 流水线
    auto detector_callback = [&detector](const cv::Mat& frame, float conf) {
        return detector.detect(frame, conf);
    };

    roi_extractor::ROIPipeline roi_pipeline(
        detector_callback,
        args.confidence_threshold,
        true,   // 使用EMA平滑
        roi_extractor::EMA_ALPHA,
        roi_extractor::BBOX_PADDING_RATIO
    );

    // 创建显示窗口
    const std::string window_name = "Fire Detection ROI";
    if (args.display_enabled) {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    // 处理视频帧
    std::cout << "Processing video...\n";
    std::cout << "Press 'q' to quit, 'p' to pause\n\n";

    cv::Mat frame;
    int frame_idx = 0;
    bool paused = false;
    double total_inference_time = 0;

    while (true) {
        if (!paused) {
            if (!cap.read(frame)) {
                std::cout << "End of video\n";
                break;
            }

            auto start_time = std::chrono::high_resolution_clock::now();

            // 通过 ROI 流水线处理单帧
            std::vector<cv::Mat> frames = {frame};
            roi_extractor::BatchResult result = roi_pipeline.processBatch(
                frames,
                roi_extractor::DetectMode::ALL,
                false  // 不跳过无检测结果的帧
            );

            auto end_time = std::chrono::high_resolution_clock::now();
            double inference_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            total_inference_time += inference_ms;

            // 获取处理后的帧
            cv::Mat display_frame;
            if (!result.frames.empty()) {
                display_frame = result.frames[0].data;
            } else {
                display_frame = frame.clone();
            }

            // 添加叠加信息
            cv::putText(
                display_frame,
                "Frame: " + std::to_string(frame_idx) + "/" + std::to_string(total_frames),
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2
            );

            cv::putText(
                display_frame,
                "Inference: " + std::to_string(static_cast<int>(inference_ms)) + " ms",
                cv::Point(10, 60),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2
            );

            // 显示检测状态
            if (!result.frames.empty()) {
                const auto& processed = result.frames[0];
                std::string status;
                if (processed.has_fire && processed.has_smoke) {
                    status = "FIRE + SMOKE";
                } else if (processed.has_fire) {
                    status = "FIRE";
                } else if (processed.has_smoke) {
                    status = "SMOKE";
                } else {
                    status = "No detection";
                }

                cv::Scalar color = (processed.has_fire || processed.has_smoke)
                    ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);

                cv::putText(
                    display_frame,
                    "Status: " + status,
                    cv::Point(10, 90),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                );
            }

            // 显示帧
            if (args.display_enabled) {
                cv::imshow(window_name, display_frame);
            }

            frame_idx++;

            // 每100帧更新一次进度
            if (frame_idx % 100 == 0) {
                double avg_ms = total_inference_time / frame_idx;
                std::cout << "  Processed " << frame_idx << "/" << total_frames
                          << " frames (avg: " << static_cast<int>(avg_ms) << " ms/frame)\n";
            }
        }

        // 处理键盘输入
        int key = cv::waitKey(args.display_enabled ? 1 : 0);
        if (key == 'q' || key == 27) {  // 'q' 或 ESC
            std::cout << "Quit requested\n";
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << "\n";
        } else if (key == ' ' && paused) {
            // 暂停时单步前进一帧
            paused = false;
            // 下一帧后会再次暂停
        }
    }

    // 清理资源
    cap.release();
    if (args.display_enabled) {
        cv::destroyAllWindows();
    }

    // 打印统计信息
    std::cout << "\n========================================\n";
    std::cout << "  Statistics\n";
    std::cout << "========================================\n";
    std::cout << "  Total frames processed: " << frame_idx << "\n";
    if (frame_idx > 0) {
        double avg_ms = total_inference_time / frame_idx;
        std::cout << "  Average inference time: " << avg_ms << " ms\n";
        std::cout << "  Average FPS: " << 1000.0 / avg_ms << "\n";
    }
    std::cout << "========================================\n";

    return 0;
}
