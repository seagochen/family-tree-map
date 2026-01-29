/**
 * @file segment_detector.cpp
 * @brief YOLOv8-seg 分割检测器实现
 */

#include "detector/segment_detector.h"

#include <iostream>
#include <cstring>

namespace fire_detection {
namespace internal {

SegmentDetector::SegmentDetector(float conf_threshold, float iou_threshold)
    : conf_threshold_(conf_threshold)
    , iou_threshold_(iou_threshold)
    , postprocessor_(NUM_CLASSES, conf_threshold, iou_threshold)
{
}

bool SegmentDetector::initialize(TrtEngineMultiTs& engine) {
    if (initialized_) {
        return true;
    }

    engine_ = &engine;

    // 创建执行上下文
    // YOLOv8-seg 输出：
    // - output0: (1, 116, 8400) - 检测结果 [x,y,w,h + 类别数 + 32个掩膜系数]
    // - output1: (1, 32, 160, 160) - 掩膜原型
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_dims = {
        nvinfer1::Dims4{1, 3, INPUT_HEIGHT, INPUT_WIDTH}
    };
    std::vector<std::string> output_names = {"output0", "output1"};

    if (!engine_->createContext(input_names, input_dims, output_names)) {
        std::cerr << "SegmentDetector: Failed to create execution context" << std::endl;
        return false;
    }

    // 分配张量
    input_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, 3, INPUT_HEIGHT, INPUT_WIDTH
    );

    // output0: (1, 4 + NUM_CLASSES + 32, 8400)
    output0_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, 4 + NUM_CLASSES + NUM_MASK_COEFFS, 8400
    );

    // output1: (1, 32, 160, 160) 掩膜原型
    output1_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, NUM_MASK_COEFFS, MASK_PROTO_H, MASK_PROTO_W
    );

    initialized_ = true;
    std::cout << "SegmentDetector: Initialized successfully" << std::endl;

    return true;
}

std::tuple<float, int, int> SegmentDetector::preprocessFrame(const cv::Mat& frame) {
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
    const float* src = reinterpret_cast<const float*>(float_img.data);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < INPUT_HEIGHT; ++h) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                int src_idx = (h * INPUT_WIDTH + w) * 3 + c;
                int dst_idx = c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;
                input_data[dst_idx] = src[src_idx];
            }
        }
    }

    input_tensor_->copyFromVector(input_data);

    return {scale, pad_x, pad_y};
}

std::vector<roi_extractor::ROIDetection> SegmentDetector::detect(const cv::Mat& frame) {
    if (!initialized_) {
        std::cerr << "SegmentDetector: Not initialized" << std::endl;
        return {};
    }

    // 预处理
    auto [scale, pad_x, pad_y] = preprocessFrame(frame);

    // 运行推理
    std::vector<Tensor<float>*> inputs = {input_tensor_.get()};
    std::vector<Tensor<float>*> outputs = {output0_tensor_.get(), output1_tensor_.get()};

    if (!engine_->infer(inputs, outputs)) {
        std::cerr << "SegmentDetector: Inference failed" << std::endl;
        return {};
    }

    // 将输出复制到主机
    std::vector<float> output0_data, output1_data;
    output0_tensor_->copyToVector(output0_data);
    output1_tensor_->copyToVector(output1_data);

    // 后处理
    postprocessor_.setConfThreshold(conf_threshold_);
    postprocessor_.setIoUThreshold(iou_threshold_);

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
        conf_threshold_
    );
}

void SegmentDetector::setConfThreshold(float threshold) {
    conf_threshold_ = threshold;
    postprocessor_.setConfThreshold(threshold);
}

void SegmentDetector::setIoUThreshold(float threshold) {
    iou_threshold_ = threshold;
    postprocessor_.setIoUThreshold(threshold);
}

}  // namespace internal
}  // namespace fire_detection
