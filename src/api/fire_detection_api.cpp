/**
 * @file fire_detection_api.cpp
 * @brief Fire Detection API C++ 实现
 */

#include "fire_detection_api.h"
#include "detector/segment_detector.h"
#include "roi/roi_extractor.h"
#include "temporal/convlstm_classifier.h"

#include <trt_engine/trt_engine.h>
#include <iostream>

namespace fire_detection {

//=============================================================================
// FireDetector::Impl - PIMPL 实现
//=============================================================================

class FireDetector::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool initialize(const Config& config) {
        if (initialized_) {
            std::cerr << "FireDetector: Already initialized" << std::endl;
            return false;
        }

        if (!config.isValid()) {
            std::cerr << "FireDetector: Invalid configuration" << std::endl;
            return false;
        }

        config_ = config;

        // 加载 YOLOv8-seg 引擎
        std::cout << "FireDetector: Loading YOLOv8-seg engine..." << std::endl;
        yolo_engine_ = std::make_unique<TrtEngineMultiTs>();
        if (!yolo_engine_->loadFromFile(config_.yolo_engine_path)) {
            std::cerr << "FireDetector: Failed to load YOLO engine: "
                      << config_.yolo_engine_path << std::endl;
            return false;
        }

        // 初始化分割检测器
        segment_detector_ = std::make_unique<internal::SegmentDetector>(
            config_.confidence_threshold,
            config_.iou_threshold
        );
        if (!segment_detector_->initialize(*yolo_engine_)) {
            std::cerr << "FireDetector: Failed to initialize segment detector" << std::endl;
            return false;
        }

        // 创建 ROI 流水线
        auto detector_callback = [this](const cv::Mat& frame, float conf) {
            return segment_detector_->detect(frame);
        };

        roi_pipeline_ = std::make_unique<roi_extractor::ROIPipeline>(
            detector_callback,
            config_.confidence_threshold,
            true,  // 使用 EMA 平滑
            config_.ema_alpha,
            config_.bbox_padding
        );

        // 加载 ConvLSTM 引擎
        std::cout << "FireDetector: Loading ConvLSTM engine..." << std::endl;
        convlstm_engine_ = std::make_unique<TrtEngineMultiTs>();
        if (!convlstm_engine_->loadFromFile(config_.convlstm_engine_path)) {
            std::cerr << "FireDetector: Failed to load ConvLSTM engine: "
                      << config_.convlstm_engine_path << std::endl;
            return false;
        }

        // 初始化 ConvLSTM 分类器
        convlstm_classifier_ = std::make_unique<convlstm::ConvLSTMClassifier>(
            *convlstm_engine_,
            config_.convlstm_seq_len
        );
        if (!convlstm_classifier_->initialize()) {
            std::cerr << "FireDetector: Failed to initialize ConvLSTM classifier" << std::endl;
            return false;
        }

        initialized_ = true;
        std::cout << "FireDetector: Initialized successfully" << std::endl;
        std::cout << "  YOLO engine: " << config_.yolo_engine_path << std::endl;
        std::cout << "  ConvLSTM engine: " << config_.convlstm_engine_path << std::endl;
        std::cout << "  Sequence length: " << config_.convlstm_seq_len << std::endl;

        return true;
    }

    bool isReady() const {
        return initialized_;
    }

    DetectionResult processFrame(const cv::Mat& frame) {
        DetectionResult result;

        if (!initialized_) {
            std::cerr << "FireDetector: Not initialized" << std::endl;
            return result;
        }

        if (frame.empty()) {
            std::cerr << "FireDetector: Empty frame" << std::endl;
            return result;
        }

        // 通过 ROI 流水线处理
        std::vector<cv::Mat> frames = {frame};
        roi_extractor::BatchResult batch_result = roi_pipeline_->processBatch(
            frames,
            roi_extractor::DetectMode::ALL,
            false  // 不跳过无检测的帧
        );

        // 获取处理后的帧
        cv::Mat processed_frame;
        if (!batch_result.frames.empty()) {
            const auto& processed = batch_result.frames[0];

            result.frame.has_fire = processed.has_fire;
            result.frame.has_smoke = processed.has_smoke;

            // 如果启用 ROI 提取，使用处理后的帧；否则使用原始帧
            if (config_.enable_roi_extraction) {
                processed_frame = processed.data;
                result.frame.roi_frame = processed.data.clone();
            } else {
                processed_frame = frame;
                result.frame.roi_frame = frame.clone();
            }

            // 转换检测框
            for (const auto& det : processed.detections) {
                BoundingBox bbox;
                bbox.x1 = static_cast<float>(det.bbox.x);
                bbox.y1 = static_cast<float>(det.bbox.y);
                bbox.x2 = static_cast<float>(det.bbox.x + det.bbox.width);
                bbox.y2 = static_cast<float>(det.bbox.y + det.bbox.height);
                bbox.confidence = det.confidence;
                bbox.class_id = static_cast<DetectionClass>(det.class_id);
                result.frame.detections.push_back(bbox);
            }
        } else {
            processed_frame = frame;
            result.frame.roi_frame = frame.clone();
        }

        // 保存最后的 ROI 帧
        last_roi_frame_ = result.frame.roi_frame;

        // ConvLSTM 时序分类
        convlstm_classifier_->addFrame(processed_frame);

        if (convlstm_classifier_->isBufferFull()) {
            if (convlstm_classifier_->infer()) {
                result.temporal_valid = true;

                auto cls = convlstm_classifier_->getClassification();
                result.temporal_class = static_cast<TemporalClass>(static_cast<int>(cls));
                result.temporal_confidence = convlstm_classifier_->getConfidence();

                // 获取各类别得分
                const auto& inference_result = convlstm_classifier_->getResult();
                for (int i = 0; i < 3; ++i) {
                    result.class_scores[i] = inference_result.class_scores[i];
                }
            }
        }

        return result;
    }

    DetectionResult processFrameRaw(const uint8_t* data, int width, int height, int channels) {
        if (channels != 3) {
            std::cerr << "FireDetector: Only 3-channel images are supported" << std::endl;
            return DetectionResult();
        }

        cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(data));
        return processFrame(frame);
    }

    void reset() {
        if (convlstm_classifier_) {
            convlstm_classifier_->reset();
        }
        last_roi_frame_ = cv::Mat();
    }

    int getBufferSize() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->getBufferSize();
        }
        return 0;
    }

    int getBufferCapacity() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->getSeqLen();
        }
        return 0;
    }

    bool isBufferFull() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->isBufferFull();
        }
        return false;
    }

    const cv::Mat& getLastRoiFrame() const {
        return last_roi_frame_;
    }

private:
    bool initialized_ = false;
    Config config_;

    // TensorRT 引擎
    std::unique_ptr<TrtEngineMultiTs> yolo_engine_;
    std::unique_ptr<TrtEngineMultiTs> convlstm_engine_;

    // 检测组件
    std::unique_ptr<internal::SegmentDetector> segment_detector_;
    std::unique_ptr<roi_extractor::ROIPipeline> roi_pipeline_;
    std::unique_ptr<convlstm::ConvLSTMClassifier> convlstm_classifier_;

    // 最后的 ROI 帧（用于 C 接口）
    cv::Mat last_roi_frame_;
};

//=============================================================================
// FireDetector 实现
//=============================================================================

FireDetector::FireDetector()
    : impl_(std::make_unique<Impl>())
{
}

FireDetector::~FireDetector() = default;

FireDetector::FireDetector(FireDetector&&) noexcept = default;
FireDetector& FireDetector::operator=(FireDetector&&) noexcept = default;

bool FireDetector::initialize(const Config& config) {
    return impl_->initialize(config);
}

bool FireDetector::isReady() const {
    return impl_->isReady();
}

DetectionResult FireDetector::processFrame(const cv::Mat& frame) {
    return impl_->processFrame(frame);
}

DetectionResult FireDetector::processFrameRaw(
    const uint8_t* data,
    int width, int height,
    int channels
) {
    return impl_->processFrameRaw(data, width, height, channels);
}

void FireDetector::reset() {
    impl_->reset();
}

int FireDetector::getBufferSize() const {
    return impl_->getBufferSize();
}

int FireDetector::getBufferCapacity() const {
    return impl_->getBufferCapacity();
}

bool FireDetector::isBufferFull() const {
    return impl_->isBufferFull();
}

const cv::Mat& FireDetector::getLastRoiFrame() const {
    return impl_->getLastRoiFrame();
}

//=============================================================================
// 工具函数实现
//=============================================================================

const char* temporalClassToString(TemporalClass cls) {
    switch (cls) {
        case TemporalClass::STATIC:   return "STATIC";
        case TemporalClass::DYNAMIC:  return "DYNAMIC";
        case TemporalClass::NEGATIVE: return "NEGATIVE";
        default: return "UNKNOWN";
    }
}

cv::Scalar temporalClassToColor(TemporalClass cls) {
    switch (cls) {
        case TemporalClass::STATIC:   return cv::Scalar(0, 165, 255);   // 橙色
        case TemporalClass::DYNAMIC:  return cv::Scalar(0, 0, 255);     // 红色
        case TemporalClass::NEGATIVE: return cv::Scalar(0, 255, 0);     // 绿色
        default: return cv::Scalar(128, 128, 128);  // 灰色
    }
}

const char* detectionClassToString(DetectionClass cls) {
    switch (cls) {
        case DetectionClass::FIRE:   return "FIRE";
        case DetectionClass::PERSON: return "PERSON";
        case DetectionClass::SMOKE:  return "SMOKE";
        default: return "UNKNOWN";
    }
}

const char* getVersion() {
    return FIRE_DETECTION_VERSION;
}

}  // namespace fire_detection
