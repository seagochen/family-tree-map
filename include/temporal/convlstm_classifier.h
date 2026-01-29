/**
 * @file convlstm_classifier.h
 * @brief ConvLSTM 时序分类器，用于区分动态火焰、静态误报和无检测
 *
 * 使用滑动窗口维护最近 N 帧的 ROI 处理结果，
 * 通过 ConvLSTM 模型进行时序推理，判断火焰的动态特性。
 *
 * 输入: (1, seq_len, 3, 640, 640) - seq_len 帧 RGB 图像
 * 输出: (1, 3, 20, 20) - 3 个类别的热力图
 *
 * 类别定义:
 *   - STATIC (0): 静态图像（可能是误报，如红色物体、灯光等）
 *   - DYNAMIC (1): 动态火焰（真实火灾，具有闪烁、扩散等特征）
 *   - NEGATIVE (2): 无检测（无火焰/烟雾）
 */

#ifndef CONVLSTM_CLASSIFIER_H
#define CONVLSTM_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <deque>
#include <vector>
#include <string>
#include <memory>

#include <trt_engine/trt_engine.h>
#include <tensors/tensor.hpp>

namespace convlstm {

// ConvLSTM 模型常量
constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int NUM_CLASSES = 3;
constexpr int HEATMAP_SIZE = 20;
constexpr int DEFAULT_SEQ_LEN = 10;

/**
 * @brief 时序分类结果枚举
 */
enum class Classification {
    STATIC = 0,    // 静态（误报）
    DYNAMIC = 1,   // 动态（真火灾）
    NEGATIVE = 2   // 无检测
};

/**
 * @brief 获取分类结果的字符串表示
 */
inline const char* classificationToString(Classification cls) {
    switch (cls) {
        case Classification::STATIC:   return "STATIC";
        case Classification::DYNAMIC:  return "DYNAMIC";
        case Classification::NEGATIVE: return "NEGATIVE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief 获取分类结果的颜色 (BGR)
 */
inline cv::Scalar classificationToColor(Classification cls) {
    switch (cls) {
        case Classification::STATIC:   return cv::Scalar(0, 165, 255);   // 橙色
        case Classification::DYNAMIC:  return cv::Scalar(0, 0, 255);     // 红色
        case Classification::NEGATIVE: return cv::Scalar(0, 255, 0);     // 绿色
        default: return cv::Scalar(128, 128, 128);  // 灰色
    }
}

/**
 * @brief 推理结果结构体
 */
struct InferenceResult {
    Classification classification;  // 分类结果
    float confidence;               // 置信度 (softmax 后的最大值)
    float class_scores[NUM_CLASSES]; // 各类别得分
    cv::Mat heatmap;                // 原始热力图 (3, 20, 20)

    InferenceResult()
        : classification(Classification::NEGATIVE)
        , confidence(0.0f)
        , class_scores{0.0f, 0.0f, 0.0f} {}
};

/**
 * @brief ConvLSTM 时序分类器
 *
 * 使用滑动窗口维护最近 seq_len 帧，每帧都进行推理。
 *
 * 使用示例:
 * @code
 * // 加载引擎
 * TrtEngineMultiTs engine;
 * engine.loadFromFile("convlstm.engine");
 *
 * // 创建分类器
 * ConvLSTMClassifier classifier(engine, 10);
 *
 * // 主循环
 * while (cap.read(frame)) {
 *     // ROI 处理后的帧
 *     cv::Mat roi_frame = roi_pipeline.process(frame);
 *
 *     // 添加帧到缓冲区
 *     classifier.addFrame(roi_frame);
 *
 *     // 缓冲区满后可以推理
 *     if (classifier.isBufferFull()) {
 *         classifier.infer();
 *         auto result = classifier.getResult();
 *         std::cout << "Classification: " << classificationToString(result.classification)
 *                   << " (" << result.confidence << ")" << std::endl;
 *     }
 * }
 * @endcode
 */
class ConvLSTMClassifier {
public:
    /**
     * @brief 构造函数
     *
     * @param engine TensorRT 引擎引用（需要外部管理生命周期）
     * @param seq_len 序列长度（滑动窗口大小），默认 10
     */
    ConvLSTMClassifier(TrtEngineMultiTs& engine, int seq_len = DEFAULT_SEQ_LEN);

    /**
     * @brief 析构函数
     */
    ~ConvLSTMClassifier() = default;

    // 禁用拷贝
    ConvLSTMClassifier(const ConvLSTMClassifier&) = delete;
    ConvLSTMClassifier& operator=(const ConvLSTMClassifier&) = delete;

    /**
     * @brief 初始化执行上下文和张量
     *
     * @return 是否初始化成功
     */
    bool initialize();

    /**
     * @brief 添加一帧到滑动窗口
     *
     * 帧会被预处理（resize, BGR→RGB, 归一化）后添加到缓冲区。
     * 如果缓冲区已满，最旧的帧会被移除。
     *
     * @param roi_frame ROI 处理后的帧 (BGR, 任意尺寸)
     */
    void addFrame(const cv::Mat& roi_frame);

    /**
     * @brief 执行推理
     *
     * 需要缓冲区已满（调用 isBufferFull() 检查）。
     *
     * @return 是否推理成功
     */
    bool infer();

    /**
     * @brief 检查缓冲区是否已满
     */
    bool isBufferFull() const {
        return static_cast<int>(frame_buffer_.size()) >= seq_len_;
    }

    /**
     * @brief 获取当前缓冲区大小
     */
    int getBufferSize() const {
        return static_cast<int>(frame_buffer_.size());
    }

    /**
     * @brief 获取序列长度
     */
    int getSeqLen() const {
        return seq_len_;
    }

    /**
     * @brief 获取最近一次推理结果
     */
    const InferenceResult& getResult() const {
        return last_result_;
    }

    /**
     * @brief 获取分类结果
     */
    Classification getClassification() const {
        return last_result_.classification;
    }

    /**
     * @brief 获取置信度
     */
    float getConfidence() const {
        return last_result_.confidence;
    }

    /**
     * @brief 清空缓冲区
     */
    void reset();

private:
    TrtEngineMultiTs& engine_;
    int seq_len_;
    bool initialized_ = false;

    // 滑动窗口缓冲区（存储预处理后的帧）
    std::deque<cv::Mat> frame_buffer_;

    // TensorRT 张量
    std::unique_ptr<Tensor<float>> input_tensor_;   // (1, seq_len, 3, 640, 640)
    std::unique_ptr<Tensor<float>> output_tensor_;  // (1, 3, 20, 20)

    // 最近一次推理结果
    InferenceResult last_result_;

    /**
     * @brief 预处理单帧
     *
     * @param frame 输入帧 (BGR)
     * @return 预处理后的帧 (RGB, float32, [0,1], 640x640)
     */
    cv::Mat preprocessFrame(const cv::Mat& frame);

    /**
     * @brief 将缓冲区数据组装到输入张量
     */
    void assembleInputTensor();

    /**
     * @brief 解析输出张量
     */
    void parseOutput();
};

}  // namespace convlstm

#endif // CONVLSTM_CLASSIFIER_H
