/**
 * @file segment_detector.h
 * @brief YOLOv8-seg 分割检测器（内部使用）
 *
 * 封装 YOLOv8-seg 推理流程：预处理 → 推理 → 后处理
 */

#ifndef FIRE_DETECTION_INTERNAL_SEGMENT_DETECTOR_H
#define FIRE_DETECTION_INTERNAL_SEGMENT_DETECTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <memory>
#include <tuple>

#include <trt_engine/trt_engine.h>
#include <tensors/tensor.hpp>
#include "detector/yolov8_postprocess.h"
#include "roi/roi_extractor.h"

namespace fire_detection {
namespace internal {

// YOLOv8-seg 模型常量
constexpr int INPUT_WIDTH = 640;
constexpr int INPUT_HEIGHT = 640;
constexpr int NUM_CLASSES = 4;  // fire, flower, person, smoke
constexpr int MASK_PROTO_H = 160;
constexpr int MASK_PROTO_W = 160;
constexpr int NUM_MASK_COEFFS = 32;

/**
 * @brief 分割检测器
 *
 * 封装 YOLOv8-seg 推理流程：预处理 → 推理 → 后处理
 */
class SegmentDetector {
public:
    /**
     * @brief 构造函数
     * @param conf_threshold 置信度阈值
     * @param iou_threshold IoU 阈值
     */
    SegmentDetector(
        float conf_threshold = 0.5f,
        float iou_threshold = 0.45f
    );

    ~SegmentDetector() = default;

    // 禁用拷贝
    SegmentDetector(const SegmentDetector&) = delete;
    SegmentDetector& operator=(const SegmentDetector&) = delete;

    /**
     * @brief 初始化检测器
     * @param engine TensorRT 引擎引用
     * @return 是否成功
     */
    bool initialize(TrtEngineMultiTs& engine);

    /**
     * @brief 执行检测
     * @param frame BGR 格式输入帧
     * @return ROI 检测结果列表
     */
    std::vector<roi_extractor::ROIDetection> detect(const cv::Mat& frame);

    /**
     * @brief 检查是否已初始化
     */
    bool isInitialized() const { return initialized_; }

    // 阈值设置
    void setConfThreshold(float threshold);
    void setIoUThreshold(float threshold);

    // 阈值获取
    float getConfThreshold() const { return conf_threshold_; }
    float getIoUThreshold() const { return iou_threshold_; }

private:
    TrtEngineMultiTs* engine_ = nullptr;

    std::unique_ptr<Tensor<float>> input_tensor_;
    std::unique_ptr<Tensor<float>> output0_tensor_;
    std::unique_ptr<Tensor<float>> output1_tensor_;

    YOLOv8PostProcessor postprocessor_;

    float conf_threshold_;
    float iou_threshold_;
    bool initialized_ = false;

    /**
     * @brief 预处理帧（Letterbox + 归一化 + HWC→CHW）
     * @param frame 输入帧 (BGR)
     * @return (scale, pad_x, pad_y) Letterbox 参数
     */
    std::tuple<float, int, int> preprocessFrame(const cv::Mat& frame);
};

}  // namespace internal
}  // namespace fire_detection

#endif // FIRE_DETECTION_INTERNAL_SEGMENT_DETECTOR_H
