/**
 * @file convlstm_classifier.cpp
 * @brief ConvLSTM 时序分类器实现
 */

#include "temporal/convlstm_classifier.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace convlstm {

ConvLSTMClassifier::ConvLSTMClassifier(TrtEngineMultiTs& engine, int seq_len)
    : engine_(engine)
    , seq_len_(seq_len)
    , initialized_(false) {
}

bool ConvLSTMClassifier::initialize() {
    if (initialized_) {
        return true;
    }

    // 创建执行上下文
    // 输入: (1, seq_len, 3, 640, 640) - 需要用 Dims 表示 5D 张量
    // 输出: (1, 3, 20, 20)
    std::vector<std::string> input_names = {"input"};
    // 注意: nvinfer1::Dims4 只能表示 4D，这里需要特殊处理
    // TrtEngineMultiTs 可能需要用 Dims 来表示 5D 输入
    // 暂时使用 Dims4 并在后续调整

    // 对于 5D 输入，我们可能需要使用 setBindingDimensions
    // 这里先尝试使用 Dims4，如果不行再调整
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 5;
    input_dims.d[0] = 1;          // batch
    input_dims.d[1] = seq_len_;   // time
    input_dims.d[2] = 3;          // channels
    input_dims.d[3] = INPUT_HEIGHT;
    input_dims.d[4] = INPUT_WIDTH;

    std::vector<nvinfer1::Dims> input_dims_vec = {input_dims};
    std::vector<std::string> output_names = {"output"};

    if (!engine_.createContext(input_names, input_dims_vec, output_names)) {
        std::cerr << "ConvLSTMClassifier: Failed to create execution context" << std::endl;
        return false;
    }

    // 分配张量
    // 输入: (1, seq_len, 3, 640, 640)
    int input_size = 1 * seq_len_ * 3 * INPUT_HEIGHT * INPUT_WIDTH;
    input_tensor_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, input_size);

    // 输出: (1, 3, 20, 20)
    int output_size = 1 * NUM_CLASSES * HEATMAP_SIZE * HEATMAP_SIZE;
    output_tensor_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, output_size);

    initialized_ = true;
    std::cout << "ConvLSTMClassifier: Initialized with seq_len=" << seq_len_ << std::endl;

    return true;
}

void ConvLSTMClassifier::reset() {
    frame_buffer_.clear();
    last_result_ = InferenceResult();
}

cv::Mat ConvLSTMClassifier::preprocessFrame(const cv::Mat& frame) {
    cv::Mat result;

    // 1. Resize 到 640x640 (使用 letterbox 保持比例)
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    float scale = std::min(
        static_cast<float>(INPUT_WIDTH) / orig_w,
        static_cast<float>(INPUT_HEIGHT) / orig_h
    );

    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);
    int pad_x = (INPUT_WIDTH - new_w) / 2;
    int pad_y = (INPUT_HEIGHT - new_h) / 2;

    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // 使用黑色填充（因为输入是 ROI 处理后的帧，非 ROI 区域已经是黑色）
    cv::Mat padded(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // 2. BGR → RGB
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 3. 归一化到 [0, 1] 并转换为 float32
    rgb.convertTo(result, CV_32FC3, 1.0 / 255.0);

    return result;
}

void ConvLSTMClassifier::addFrame(const cv::Mat& roi_frame) {
    // 预处理帧
    cv::Mat processed = preprocessFrame(roi_frame);

    // 添加到缓冲区
    frame_buffer_.push_back(processed);

    // 维护滑动窗口大小
    while (static_cast<int>(frame_buffer_.size()) > seq_len_) {
        frame_buffer_.pop_front();
    }
}

void ConvLSTMClassifier::assembleInputTensor() {
    // 输入张量布局: (1, seq_len, 3, H, W) - NTCHW
    // 需要将 frame_buffer_ 中的帧按顺序组装

    int frame_size = 3 * INPUT_HEIGHT * INPUT_WIDTH;
    std::vector<float> input_data(seq_len_ * frame_size);

    for (int t = 0; t < seq_len_; ++t) {
        const cv::Mat& frame = frame_buffer_[t];
        const float* src = reinterpret_cast<const float*>(frame.data);

        // 将 HWC 格式转换为 CHW 格式
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < INPUT_HEIGHT; ++h) {
                for (int w = 0; w < INPUT_WIDTH; ++w) {
                    int src_idx = (h * INPUT_WIDTH + w) * 3 + c;  // HWC
                    int dst_idx = t * frame_size + c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;  // TCHW
                    input_data[dst_idx] = src[src_idx];
                }
            }
        }
    }

    input_tensor_->copyFromVector(input_data);
}

void ConvLSTMClassifier::parseOutput() {
    // 获取输出数据
    std::vector<float> output_data;
    output_tensor_->copyToVector(output_data);

    // 输出形状: (1, 3, 20, 20)
    // 对每个类别的 20x20 热力图取最大值
    float max_scores[NUM_CLASSES] = {0.0f, 0.0f, 0.0f};
    int spatial_size = HEATMAP_SIZE * HEATMAP_SIZE;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int i = 0; i < spatial_size; ++i) {
            float val = output_data[c * spatial_size + i];
            max_scores[c] = std::max(max_scores[c], val);
        }
    }

    // 应用 softmax 得到概率
    float exp_scores[NUM_CLASSES];
    float sum_exp = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        exp_scores[c] = std::exp(max_scores[c]);
        sum_exp += exp_scores[c];
    }

    for (int c = 0; c < NUM_CLASSES; ++c) {
        last_result_.class_scores[c] = exp_scores[c] / sum_exp;
    }

    // 找到最大概率的类别
    int argmax = 0;
    float max_prob = last_result_.class_scores[0];
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (last_result_.class_scores[c] > max_prob) {
            max_prob = last_result_.class_scores[c];
            argmax = c;
        }
    }

    last_result_.classification = static_cast<Classification>(argmax);
    last_result_.confidence = max_prob;

    // 存储热力图（可选，用于可视化）
    last_result_.heatmap = cv::Mat(NUM_CLASSES, spatial_size, CV_32FC1);
    std::memcpy(last_result_.heatmap.data, output_data.data(),
                output_data.size() * sizeof(float));
}

bool ConvLSTMClassifier::infer() {
    if (!initialized_) {
        std::cerr << "ConvLSTMClassifier: Not initialized" << std::endl;
        return false;
    }

    if (!isBufferFull()) {
        std::cerr << "ConvLSTMClassifier: Buffer not full ("
                  << frame_buffer_.size() << "/" << seq_len_ << ")" << std::endl;
        return false;
    }

    // 组装输入张量
    assembleInputTensor();

    // 运行推理
    std::vector<Tensor<float>*> inputs = {input_tensor_.get()};
    std::vector<Tensor<float>*> outputs = {output_tensor_.get()};

    if (!engine_.infer(inputs, outputs)) {
        std::cerr << "ConvLSTMClassifier: Inference failed" << std::endl;
        return false;
    }

    // 解析输出
    parseOutput();

    return true;
}

}  // namespace convlstm
