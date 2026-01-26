/**
 * @file main.cpp
 * @brief Fire detection pipeline with real-time ROI visualization
 *
 * This program:
 * 1. Loads TensorRT segment engine using cudatt's TrtEngineMultiTs
 * 2. Reads video file frame by frame
 * 3. Uses ROI extractor to mask non-ROI regions to black
 * 4. Displays processed frames in real-time
 */

#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

// cudatt headers
#include <trt_engine/trt_engine.h>
#include <tensors/tensor.hpp>

// Local headers
#include "yolov8_postprocess.h"
#include "roi_extractor.h"

// YOLOv8-seg model constants
constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int NUM_CLASSES = 3;  // fire, person, smoke
constexpr int MASK_PROTO_H = 160;
constexpr int MASK_PROTO_W = 160;
constexpr int NUM_MASK_COEFFS = 32;

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
 * @brief Preprocess frame for YOLOv8 inference
 * @param frame Input BGR frame
 * @param input_tensor Output tensor to fill
 * @return Letterbox parameters (scale, pad_x, pad_y)
 */
std::tuple<float, int, int> preprocessFrame(
    const cv::Mat& frame,
    Tensor<float>& input_tensor
) {
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    // Compute letterbox parameters
    float scale = std::min(
        static_cast<float>(INPUT_WIDTH) / orig_w,
        static_cast<float>(INPUT_HEIGHT) / orig_h
    );
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    int pad_x = (INPUT_WIDTH - new_w) / 2;
    int pad_y = (INPUT_HEIGHT - new_h) / 2;

    // Resize and letterbox
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    cv::Mat padded(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // Convert BGR to RGB and normalize to [0, 1]
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // Convert to float and normalize
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // CHW format for TensorRT (NCHW)
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
 * @brief Run detection and create ROI detector callback
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
        // Preprocess
        auto [scale, pad_x, pad_y] = preprocessFrame(frame, input_tensor_);

        // Run inference
        std::vector<Tensor<float>*> inputs = {&input_tensor_};
        std::vector<Tensor<float>*> outputs = {&output0_tensor_, &output1_tensor_};

        if (!engine_.infer(inputs, outputs)) {
            std::cerr << "Inference failed!" << std::endl;
            return {};
        }

        // Copy outputs to host
        std::vector<float> output0_data, output1_data;
        output0_tensor_.copyToVector(output0_data);
        output1_tensor_.copyToVector(output1_data);

        // Postprocess
        postprocessor_.setConfThreshold(conf_threshold);
        std::vector<Detection> detections = postprocessor_.process(
            output0_data, output1_data,
            frame.cols, frame.rows,
            INPUT_WIDTH, INPUT_HEIGHT
        );

        // Convert Detection to ROIDetection
        std::vector<cv::Mat> masks;
        for (const auto& det : detections) {
            cv::Mat mask(det.mask_height, det.mask_width, CV_32FC1);
            std::memcpy(mask.data, det.mask.data(), det.mask.size() * sizeof(float));

            // Resize mask to frame size
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
    // Parse command line arguments
    std::string video_path;
    std::string engine_path;
    float confidence_threshold = 0.5f;
    float iou_threshold = 0.45f;
    bool display_enabled = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        } else if (arg == "--video" && i + 1 < argc) {
            video_path = argv[++i];
        } else if (arg == "--engine" && i + 1 < argc) {
            engine_path = argv[++i];
        } else if (arg == "--confidence" && i + 1 < argc) {
            confidence_threshold = std::stof(argv[++i]);
        } else if (arg == "--iou" && i + 1 < argc) {
            iou_threshold = std::stof(argv[++i]);
        } else if (arg == "--no-display") {
            display_enabled = false;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            printUsage(argv[0]);
            return 1;
        }
    }

    // Validate required arguments
    if (video_path.empty()) {
        std::cerr << "Error: --video is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }
    if (engine_path.empty()) {
        std::cerr << "Error: --engine is required" << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    std::cout << "========================================\n";
    std::cout << "  Fire Detection ROI Visualization\n";
    std::cout << "========================================\n\n";

    std::cout << "Configuration:\n";
    std::cout << "  Video: " << video_path << "\n";
    std::cout << "  Engine: " << engine_path << "\n";
    std::cout << "  Confidence threshold: " << confidence_threshold << "\n";
    std::cout << "  IoU threshold: " << iou_threshold << "\n";
    std::cout << "  Display: " << (display_enabled ? "enabled" : "disabled") << "\n\n";

    // Initialize CUDA
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

    // Load TensorRT engine
    std::cout << "Loading TensorRT engine...\n";
    TrtEngineMultiTs engine;
    if (!engine.loadFromFile(engine_path)) {
        std::cerr << "Error: Failed to load engine from " << engine_path << std::endl;
        return 1;
    }
    std::cout << "  Engine loaded successfully\n\n";

    // Create execution context
    // YOLOv8-seg outputs:
    // - output0: (1, 116, 8400) - detections [x,y,w,h + num_classes + 32 mask coeffs]
    // - output1: (1, 32, 160, 160) - mask prototypes
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_dims = {nvinfer1::Dims4{1, 3, INPUT_HEIGHT, INPUT_WIDTH}};
    std::vector<std::string> output_names = {"output0", "output1"};

    if (!engine.createContext(input_names, input_dims, output_names)) {
        std::cerr << "Error: Failed to create execution context" << std::endl;
        return 1;
    }
    std::cout << "  Execution context created\n\n";

    // Allocate tensors
    Tensor<float> input_tensor(TensorType::FLOAT32, 1, 3, INPUT_HEIGHT, INPUT_WIDTH);
    // output0: (1, 116, 8400) where 116 = 4 + num_classes + 32 mask coeffs
    Tensor<float> output0_tensor(TensorType::FLOAT32, 1, 4 + NUM_CLASSES + NUM_MASK_COEFFS, 8400);
    // output1: (1, 32, 160, 160) mask prototypes
    Tensor<float> output1_tensor(TensorType::FLOAT32, 1, NUM_MASK_COEFFS, MASK_PROTO_H, MASK_PROTO_W);

    // Open video file
    std::cout << "Opening video file...\n";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Failed to open video: " << video_path << std::endl;
        return 1;
    }

    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));

    std::cout << "  Resolution: " << frame_width << "x" << frame_height << "\n";
    std::cout << "  FPS: " << fps << "\n";
    std::cout << "  Total frames: " << total_frames << "\n\n";

    // Create detector
    SegmentDetector detector(
        engine, input_tensor, output0_tensor, output1_tensor,
        confidence_threshold, iou_threshold
    );

    // Create ROI pipeline with detector callback
    auto detector_callback = [&detector](const cv::Mat& frame, float conf) {
        return detector.detect(frame, conf);
    };

    roi_extractor::ROIPipeline roi_pipeline(
        detector_callback,
        confidence_threshold,
        true,   // use_ema
        roi_extractor::EMA_ALPHA,
        roi_extractor::BBOX_PADDING_RATIO
    );

    // Create display window
    const std::string window_name = "Fire Detection ROI";
    if (display_enabled) {
        cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
    }

    // Process video frames
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

            // Process single frame through ROI pipeline
            std::vector<cv::Mat> frames = {frame};
            roi_extractor::BatchResult result = roi_pipeline.processBatch(
                frames,
                roi_extractor::DetectMode::ALL,
                false  // don't skip frames without detection
            );

            auto end_time = std::chrono::high_resolution_clock::now();
            double inference_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
            total_inference_time += inference_ms;

            // Get processed frame
            cv::Mat display_frame;
            if (!result.frames.empty()) {
                display_frame = result.frames[0].data;
            } else {
                display_frame = frame.clone();
            }

            // Add overlay information
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

            // Show detection status
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

            // Display frame
            if (display_enabled) {
                cv::imshow(window_name, display_frame);
            }

            frame_idx++;

            // Progress update every 100 frames
            if (frame_idx % 100 == 0) {
                double avg_ms = total_inference_time / frame_idx;
                std::cout << "  Processed " << frame_idx << "/" << total_frames
                          << " frames (avg: " << static_cast<int>(avg_ms) << " ms/frame)\n";
            }
        }

        // Handle keyboard input
        int key = cv::waitKey(display_enabled ? 1 : 0);
        if (key == 'q' || key == 27) {  // 'q' or ESC
            std::cout << "Quit requested\n";
            break;
        } else if (key == 'p') {
            paused = !paused;
            std::cout << (paused ? "Paused" : "Resumed") << "\n";
        } else if (key == ' ' && paused) {
            // Step one frame when paused
            paused = false;
            // Will pause again after next frame
        }
    }

    // Cleanup
    cap.release();
    if (display_enabled) {
        cv::destroyAllWindows();
    }

    // Print statistics
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
