/**
 * @file convlstm_classifier.cpp
 * @brief ConvLSTM 时序分类器实现
 *
 * 本模块实现基于ConvLSTM的时序分类功能，用于火灾检测的二次验证。
 * 主要功能包括：
 *   1. 接收ROI提取后的视频帧序列
 *   2. 使用ConvLSTM模型分析时序特征
 *   3. 输出分类结果（真火/假火/无火等）
 *
 * 业务背景说明：
 *   单帧检测容易产生误报（如红色物体、反光等被误判为火焰）。
 *   ConvLSTM通过分析连续多帧的时序变化特征（如火焰的闪烁、烟雾的扩散），
 *   可以有效区分真实火灾和误报源，大幅提高检测准确率。
 *
 * 数据流说明：
 *   ROI帧 → 预处理(letterbox+归一化) → 帧缓冲区 → 组装5D张量 → TensorRT推理 → 分类结果
 */

#include "temporal/convlstm_classifier.h"

#include <iostream>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace convlstm {

/**
 * @brief ConvLSTMClassifier构造函数
 *
 * 业务逻辑说明：
 *   初始化分类器的基本参数，但不分配GPU资源。
 *   实际的资源分配延迟到initialize()调用时进行，
 *   这样可以在确认需要使用分类器时才占用GPU内存。
 *
 * @param engine TensorRT引擎引用，负责实际的GPU推理
 * @param seq_len 序列长度，即需要多少连续帧才能进行一次分类
 */
ConvLSTMClassifier::ConvLSTMClassifier(TrtEngineMultiTs& engine, int seq_len)
    : engine_(engine)
    , seq_len_(seq_len)
    , initialized_(false) {
}

/**
 * @brief 初始化分类器
 *
 * 业务逻辑说明：
 *   创建TensorRT执行上下文并分配输入输出张量内存。
 *   这是GPU资源密集操作，应在确认需要时序分类时调用。
 *
 *   模型输入输出规格：
 *     - 输入: (1, seq_len, 3, 640, 640) - 批次×时间×通道×高×宽
 *     - 输出: (1, 3, 20, 20) - 批次×类别数×热力图高×热力图宽
 *
 *   5D张量说明：
 *     ConvLSTM需要5D输入来处理时序数据，与普通CNN的4D输入不同。
 *     TensorRT的Dims类可以表示任意维度的张量。
 *
 * @return 初始化成功返回true，失败返回false
 */
bool ConvLSTMClassifier::initialize() {
    // 防止重复初始化
    if (initialized_) {
        return true;
    }

    // 配置输入张量维度
    // 输入形状: (1, seq_len, 3, 640, 640) - NTCHW格式
    // N=batch, T=time, C=channel, H=height, W=width
    std::vector<std::string> input_names = {"input"};

    // 构建5D输入维度（nvinfer1::Dims4只支持4D，需要使用通用Dims）
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 5;
    input_dims.d[0] = 1;          // batch size（固定为1）
    input_dims.d[1] = seq_len_;   // 时间序列长度
    input_dims.d[2] = 3;          // RGB通道数
    input_dims.d[3] = INPUT_HEIGHT;  // 图像高度（640）
    input_dims.d[4] = INPUT_WIDTH;   // 图像宽度（640）

    std::vector<nvinfer1::Dims> input_dims_vec = {input_dims};
    std::vector<std::string> output_names = {"output"};

    // 创建TensorRT执行上下文
    if (!engine_.createContext(input_names, input_dims_vec, output_names)) {
        std::cerr << "ConvLSTMClassifier: Failed to create execution context" << std::endl;
        return false;
    }

    // 分配输入张量内存
    // 大小 = 1 × seq_len × 3 × 640 × 640
    int input_size = 1 * seq_len_ * 3 * INPUT_HEIGHT * INPUT_WIDTH;
    input_tensor_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, input_size);

    // 分配输出张量内存
    // 大小 = 1 × 3（类别数）× 20 × 20（热力图尺寸）
    int output_size = 1 * NUM_CLASSES * HEATMAP_SIZE * HEATMAP_SIZE;
    output_tensor_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, output_size);

    initialized_ = true;
    std::cout << "ConvLSTMClassifier: Initialized with seq_len=" << seq_len_ << std::endl;

    return true;
}

/**
 * @brief 重置分类器状态
 *
 * 业务逻辑说明：
 *   清空帧缓冲区和上次的分类结果。
 *   应在以下场景调用：
 *     - 视频源切换时
 *     - 场景发生重大变化时
 *     - 检测到连续误报需要重新校准时
 *
 *   注意：这不会释放GPU内存，仅清空缓冲数据。
 */
void ConvLSTMClassifier::reset() {
    frame_buffer_.clear();
    last_result_ = InferenceResult();
}

/**
 * @brief 预处理单帧图像
 *
 * 业务逻辑说明：
 *   将输入帧转换为模型所需的格式。处理步骤包括：
 *
 *   1. Letterbox缩放：
 *      - 保持原始宽高比缩放到640×640
 *      - 使用黑色填充空白区域
 *      - 这对于ROI帧很合适，因为非ROI区域本来就是黑色
 *
 *   2. 颜色空间转换：
 *      - OpenCV默认BGR，模型需要RGB
 *
 *   3. 归一化：
 *      - 像素值从[0,255]归一化到[0,1]
 *      - 转换为float32类型
 *
 *   为什么使用Letterbox而非直接拉伸：
 *     保持宽高比可以避免目标形变，使模型能正确识别火焰/烟雾的形态特征。
 *
 * @param frame 输入帧（BGR格式，任意尺寸）
 * @return 预处理后的帧（RGB格式，640×640，float32，归一化）
 */
cv::Mat ConvLSTMClassifier::preprocessFrame(const cv::Mat& frame) {
    cv::Mat result;

    // 步骤1：Letterbox缩放 - 保持宽高比缩放到640×640
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    // 计算缩放比例，取较小值以确保图像完全在目标区域内
    float scale = std::min(
        static_cast<float>(INPUT_WIDTH) / orig_w,
        static_cast<float>(INPUT_HEIGHT) / orig_h
    );

    // 计算缩放后的尺寸
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    // 计算居中填充的偏移量
    int pad_x = (INPUT_WIDTH - new_w) / 2;
    int pad_y = (INPUT_HEIGHT - new_h) / 2;

    // 执行缩放
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // 创建黑色背景并将缩放后的图像居中放置
    // 使用黑色填充是因为输入是ROI处理后的帧，非ROI区域已经是黑色
    cv::Mat padded(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(0, 0, 0));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // 步骤2：BGR → RGB颜色空间转换
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 步骤3：归一化到[0,1]并转换为float32
    rgb.convertTo(result, CV_32FC3, 1.0 / 255.0);

    return result;
}

/**
 * @brief 向帧缓冲区添加新帧
 *
 * 业务逻辑说明：
 *   实现滑动窗口机制，维护最近seq_len帧的缓冲区。
 *
 *   滑动窗口策略：
 *     - 新帧添加到队列末尾
 *     - 当超过seq_len时，移除最旧的帧
 *     - 这样可以实现连续的时序分类，每次只需添加1帧
 *
 *   调用时机：
 *     - 每当有新的ROI帧时调用
 *     - 调用后检查isBufferFull()判断是否可以推理
 *
 * @param roi_frame ROI提取后的帧（非ROI区域为黑色）
 */
void ConvLSTMClassifier::addFrame(const cv::Mat& roi_frame) {
    // 预处理帧（letterbox + 归一化）
    cv::Mat processed = preprocessFrame(roi_frame);

    // 添加到缓冲区末尾
    frame_buffer_.push_back(processed);

    // 维护滑动窗口大小，移除超出的旧帧
    while (static_cast<int>(frame_buffer_.size()) > seq_len_) {
        frame_buffer_.pop_front();
    }
}

/**
 * @brief 组装输入张量
 *
 * 业务逻辑说明：
 *   将帧缓冲区中的图像数据组装成模型所需的5D张量格式。
 *
 *   数据格式转换：
 *     - 输入：frame_buffer_中的cv::Mat，每帧为HWC格式（高×宽×通道）
 *     - 输出：NTCHW格式的连续内存（批次×时间×通道×高×宽）
 *
 *   为什么需要格式转换：
 *     - OpenCV使用HWC格式（Height-Width-Channel），像素交错存储
 *     - 深度学习模型通常使用CHW格式（Channel-Height-Width），通道分离存储
 *     - CHW格式更适合GPU的并行计算，卷积操作效率更高
 *
 *   内存布局示例（对于3×2×2的图像）：
 *     HWC: R00 G00 B00 R01 G01 B01 R10 G10 B10 R11 G11 B11
 *     CHW: R00 R01 R10 R11 G00 G01 G10 G11 B00 B01 B10 B11
 */
void ConvLSTMClassifier::assembleInputTensor() {
    // 单帧数据大小：3通道 × 640高 × 640宽
    int frame_size = 3 * INPUT_HEIGHT * INPUT_WIDTH;

    // 分配整个序列的内存
    std::vector<float> input_data(seq_len_ * frame_size);

    // 遍历时间序列中的每一帧
    for (int t = 0; t < seq_len_; ++t) {
        const cv::Mat& frame = frame_buffer_[t];
        const float* src = reinterpret_cast<const float*>(frame.data);

        // 将HWC格式转换为CHW格式
        // 遍历顺序：通道 → 高度 → 宽度
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < INPUT_HEIGHT; ++h) {
                for (int w = 0; w < INPUT_WIDTH; ++w) {
                    // HWC索引：(h * W + w) * C + c
                    int src_idx = (h * INPUT_WIDTH + w) * 3 + c;
                    // TCHW索引：t * (C*H*W) + c * (H*W) + h * W + w
                    int dst_idx = t * frame_size + c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;
                    input_data[dst_idx] = src[src_idx];
                }
            }
        }
    }

    // 将数据复制到GPU张量
    input_tensor_->copyFromVector(input_data);
}

/**
 * @brief 解析模型输出
 *
 * 业务逻辑说明：
 *   将模型的原始输出转换为可解释的分类结果。
 *
 *   模型输出格式：
 *     - 形状: (1, 3, 20, 20) - 3个类别，每个类别一个20×20的热力图
 *     - 热力图表示图像各区域属于该类别的置信度
 *
 *   后处理步骤：
 *     1. 对每个类别的热力图取最大值（最显著的区域）
 *     2. 对3个最大值应用softmax，转换为概率分布
 *     3. 选择概率最高的类别作为分类结果
 *
 *   类别定义（假设）：
 *     - 0: 无火
 *     - 1: 真火
 *     - 2: 假火/误报
 *
 *   为什么取热力图最大值：
 *     火焰通常只出现在图像的局部区域，取最大值可以捕捉到
 *     最显著的火焰特征，忽略背景区域的低响应。
 */
void ConvLSTMClassifier::parseOutput() {
    // 从GPU获取输出数据
    std::vector<float> output_data;
    output_tensor_->copyToVector(output_data);

    // 输出形状: (1, 3, 20, 20)
    // 对每个类别的20×20热力图取最大值
    float max_scores[NUM_CLASSES] = {0.0f, 0.0f, 0.0f};
    int spatial_size = HEATMAP_SIZE * HEATMAP_SIZE;  // 20×20 = 400

    // 遍历每个类别的热力图，找出最大响应值
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int i = 0; i < spatial_size; ++i) {
            float val = output_data[c * spatial_size + i];
            max_scores[c] = std::max(max_scores[c], val);
        }
    }

    // 应用softmax将原始分数转换为概率分布
    // softmax(x_i) = exp(x_i) / sum(exp(x_j))
    float exp_scores[NUM_CLASSES];
    float sum_exp = 0.0f;

    // 计算指数和累加和
    for (int c = 0; c < NUM_CLASSES; ++c) {
        exp_scores[c] = std::exp(max_scores[c]);
        sum_exp += exp_scores[c];
    }

    // 归一化得到概率
    for (int c = 0; c < NUM_CLASSES; ++c) {
        last_result_.class_scores[c] = exp_scores[c] / sum_exp;
    }

    // 找到概率最高的类别（argmax）
    int argmax = 0;
    float max_prob = last_result_.class_scores[0];
    for (int c = 1; c < NUM_CLASSES; ++c) {
        if (last_result_.class_scores[c] > max_prob) {
            max_prob = last_result_.class_scores[c];
            argmax = c;
        }
    }

    // 保存分类结果
    last_result_.classification = static_cast<Classification>(argmax);
    last_result_.confidence = max_prob;

    // 存储原始热力图（可选，用于可视化调试）
    // 将热力图数据复制到OpenCV Mat中
    last_result_.heatmap = cv::Mat(NUM_CLASSES, spatial_size, CV_32FC1);
    std::memcpy(last_result_.heatmap.data, output_data.data(),
                output_data.size() * sizeof(float));
}

/**
 * @brief 执行推理
 *
 * 业务逻辑说明：
 *   对当前帧缓冲区中的序列进行分类推理。
 *
 *   前置条件：
 *     1. 分类器已初始化（initialize()返回true）
 *     2. 帧缓冲区已满（isBufferFull()返回true）
 *
 *   执行流程：
 *     1. 组装输入张量（HWC→CHW格式转换）
 *     2. 调用TensorRT引擎执行GPU推理
 *     3. 解析输出得到分类结果
 *
 *   性能说明：
 *     - GPU推理是主要耗时操作
 *     - 帧预处理在addFrame时已完成，这里只需格式转换
 *     - 推理完成后结果保存在last_result_中
 *
 * @return 推理成功返回true，失败返回false
 */
bool ConvLSTMClassifier::infer() {
    // 检查是否已初始化
    if (!initialized_) {
        std::cerr << "ConvLSTMClassifier: Not initialized" << std::endl;
        return false;
    }

    // 检查帧缓冲区是否已满
    if (!isBufferFull()) {
        std::cerr << "ConvLSTMClassifier: Buffer not full ("
                  << frame_buffer_.size() << "/" << seq_len_ << ")" << std::endl;
        return false;
    }

    // 步骤1：组装输入张量（格式转换）
    assembleInputTensor();

    // 步骤2：执行TensorRT推理
    std::vector<Tensor<float>*> inputs = {input_tensor_.get()};
    std::vector<Tensor<float>*> outputs = {output_tensor_.get()};

    if (!engine_.infer(inputs, outputs)) {
        std::cerr << "ConvLSTMClassifier: Inference failed" << std::endl;
        return false;
    }

    // 步骤3：解析输出结果
    parseOutput();

    return true;
}

}  // namespace convlstm
