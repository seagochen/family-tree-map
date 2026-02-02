/**
 * @file scene_controller.cpp
 * @brief 场景控制器实现
 *
 * 管理球形摄像机的多场景切换，每个场景维护独立的帧队列和时序分析状态。
 */

#include "scene/scene_controller.h"
#include "detector/segment_detector.h"
#include "roi/roi_extractor.h"
#include "temporal/convlstm_classifier.h"

#include <trt_engine/trt_engine.h>
#include <tensors/tensor.hpp>
#include <unordered_map>
#include <deque>
#include <mutex>
#include <algorithm>
#include <iostream>
#include <cmath>

namespace fire_detection {

//=============================================================================
// 内部：单场景上下文
//=============================================================================

/**
 * @brief 单个场景的上下文数据
 *
 * 每个场景维护独立的时间戳队列和 ConvLSTM 分类器实例。
 * ConvLSTMClassifier 内部管理帧缓冲区和 GPU 推理。
 */
struct SceneContext {
    int32_t scene_id;

    /// 时间戳队列（用于时间间隔检查，按时间排序）
    std::deque<int64_t> timestamp_queue;

    /// 独立的 ConvLSTM 分类器（内部管理帧缓冲区和推理）
    std::unique_ptr<convlstm::ConvLSTMClassifier> classifier;

    /// 最后处理的时间戳
    int64_t last_processed_timestamp = 0;

    /// 最后的ROI帧
    cv::Mat last_roi_frame;

    /// 构造函数
    explicit SceneContext(int32_t id) : scene_id(id) {}

    /// 清空状态
    void reset() {
        timestamp_queue.clear();
        if (classifier) {
            classifier->reset();
        }
        last_processed_timestamp = 0;
        last_roi_frame = cv::Mat();
    }
};

//=============================================================================
// SceneController::Impl - PIMPL 实现
//=============================================================================

class SceneController::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    bool initialize(const SceneConfig& config);
    bool isReady() const { return initialized_; }

    // 场景管理
    bool registerScene(int32_t scene_id);
    bool unregisterScene(int32_t scene_id);
    bool hasScene(int32_t scene_id) const;
    std::vector<int32_t> getSceneIds() const;
    size_t getSceneCount() const;

    // 帧处理
    SceneDetectionResult processFrame(
        int32_t scene_id,
        const cv::Mat& frame,
        int64_t timestamp_ms
    );

    SceneDetectionResult processFrameRaw(
        int32_t scene_id,
        const uint8_t* data,
        int width, int height, int channels,
        int64_t timestamp_ms
    );

    // 场景状态
    int getSceneQueueSize(int32_t scene_id) const;
    int getSceneBufferSize(int32_t scene_id) const;
    bool isSceneReadyForInference(int32_t scene_id) const;
    void resetScene(int32_t scene_id);
    void resetAllScenes();
    cv::Mat getLastRoiFrame(int32_t scene_id) const;

private:
    bool initialized_ = false;
    SceneConfig config_;

    // 共享的检测组件（所有场景共用）
    std::unique_ptr<TrtEngineMultiTs> yolo_engine_;
    std::unique_ptr<TrtEngineMultiTs> convlstm_engine_;
    std::unique_ptr<internal::SegmentDetector> segment_detector_;
    std::unique_ptr<roi_extractor::ROIPipeline> roi_pipeline_;

    // 场景映射表
    std::unordered_map<int32_t, std::unique_ptr<SceneContext>> scenes_;

    // 线程安全
    mutable std::mutex scenes_mutex_;

    // 内部方法
    void checkAndPurgeOldFrames(SceneContext& ctx, int64_t current_timestamp);
    void insertTimestamp(SceneContext& ctx, int64_t timestamp_ms);
    cv::Mat runDetection(const cv::Mat& frame, FrameResult& frame_result);

    // 为场景创建并初始化 ConvLSTMClassifier
    bool initializeSceneClassifier(SceneContext& ctx);
};

//=============================================================================
// 初始化
//=============================================================================

bool SceneController::Impl::initialize(const SceneConfig& config) {
    if (initialized_) {
        std::cerr << "SceneController: Already initialized" << std::endl;
        return false;
    }

    if (!config.isValid()) {
        std::cerr << "SceneController: Invalid configuration" << std::endl;
        return false;
    }

    config_ = config;

    // 加载 YOLOv8-seg TensorRT 引擎
    std::cout << "SceneController: Loading YOLOv8-seg engine..." << std::endl;
    yolo_engine_ = std::make_unique<TrtEngineMultiTs>();
    if (!yolo_engine_->loadFromFile(config_.detector_config.yolo_engine_path)) {
        std::cerr << "SceneController: Failed to load YOLO engine: "
                  << config_.detector_config.yolo_engine_path << std::endl;
        return false;
    }

    // 初始化分割检测器
    segment_detector_ = std::make_unique<internal::SegmentDetector>(
        config_.detector_config.confidence_threshold,
        config_.detector_config.iou_threshold
    );
    if (!segment_detector_->initialize(*yolo_engine_)) {
        std::cerr << "SceneController: Failed to initialize segment detector" << std::endl;
        return false;
    }

    // 创建 ROI 提取流水线
    auto detector_callback = [this](const cv::Mat& frame, float conf) {
        return segment_detector_->detect(frame);
    };

    roi_pipeline_ = std::make_unique<roi_extractor::ROIPipeline>(
        detector_callback,
        config_.detector_config.confidence_threshold,
        true,  // 使用 EMA 平滑
        config_.detector_config.ema_alpha,
        config_.detector_config.bbox_padding
    );

    // 加载 ConvLSTM TensorRT 引擎（共享引擎，每个场景有独立的分类器实例）
    std::cout << "SceneController: Loading ConvLSTM engine..." << std::endl;
    convlstm_engine_ = std::make_unique<TrtEngineMultiTs>();
    if (!convlstm_engine_->loadFromFile(config_.detector_config.convlstm_engine_path)) {
        std::cerr << "SceneController: Failed to load ConvLSTM engine: "
                  << config_.detector_config.convlstm_engine_path << std::endl;
        return false;
    }

    initialized_ = true;
    std::cout << "SceneController: Initialized successfully" << std::endl;
    std::cout << "  Max frame gap: " << config_.max_frame_gap_ms << "ms" << std::endl;
    std::cout << "  Min sample interval: " << config_.min_sample_interval_ms << "ms" << std::endl;
    std::cout << "  Max queue size: " << config_.max_queue_size << std::endl;

    return true;
}

//=============================================================================
// 场景管理
//=============================================================================

bool SceneController::Impl::registerScene(int32_t scene_id) {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    if (scenes_.find(scene_id) != scenes_.end()) {
        return false;  // 场景已存在
    }

    scenes_[scene_id] = std::make_unique<SceneContext>(scene_id);
    std::cout << "SceneController: Registered scene " << scene_id << std::endl;
    return true;
}

bool SceneController::Impl::unregisterScene(int32_t scene_id) {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it == scenes_.end()) {
        return false;  // 场景不存在
    }

    scenes_.erase(it);
    std::cout << "SceneController: Unregistered scene " << scene_id << std::endl;
    return true;
}

bool SceneController::Impl::hasScene(int32_t scene_id) const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);
    return scenes_.find(scene_id) != scenes_.end();
}

std::vector<int32_t> SceneController::Impl::getSceneIds() const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    std::vector<int32_t> ids;
    ids.reserve(scenes_.size());
    for (const auto& pair : scenes_) {
        ids.push_back(pair.first);
    }
    return ids;
}

size_t SceneController::Impl::getSceneCount() const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);
    return scenes_.size();
}

//=============================================================================
// 核心算法
//=============================================================================

/**
 * @brief 检查并清除过期帧
 *
 * 如果新帧与队列中最新帧的时间差超过阈值，清空整个队列。
 */
void SceneController::Impl::checkAndPurgeOldFrames(
    SceneContext& ctx,
    int64_t current_timestamp
) {
    if (ctx.timestamp_queue.empty()) {
        return;
    }

    // 获取队列中最新的时间戳
    int64_t newest_in_queue = ctx.timestamp_queue.back();

    // 计算时间差
    int64_t time_gap = std::abs(current_timestamp - newest_in_queue);

    if (time_gap > config_.max_frame_gap_ms) {
        std::cout << "Scene " << ctx.scene_id
                  << ": Time gap " << time_gap << "ms exceeds threshold ("
                  << config_.max_frame_gap_ms << "ms), purging queue" << std::endl;
        ctx.reset();
    }
}

/**
 * @brief 按时间戳排序插入
 */
void SceneController::Impl::insertTimestamp(
    SceneContext& ctx,
    int64_t timestamp_ms
) {
    // 使用二分查找找到插入位置
    auto it = std::lower_bound(
        ctx.timestamp_queue.begin(),
        ctx.timestamp_queue.end(),
        timestamp_ms
    );

    // 检查是否为重复时间戳
    if (it != ctx.timestamp_queue.end() && *it == timestamp_ms) {
        // 时间戳重复，忽略
        return;
    }

    // 插入新时间戳
    ctx.timestamp_queue.insert(it, timestamp_ms);

    // 维护队列最大大小
    while (static_cast<int>(ctx.timestamp_queue.size()) > config_.max_queue_size) {
        ctx.timestamp_queue.pop_front();  // 移除最旧的时间戳
    }
}

/**
 * @brief 运行空间检测
 */
cv::Mat SceneController::Impl::runDetection(
    const cv::Mat& frame,
    FrameResult& frame_result
) {
    std::vector<cv::Mat> frames = {frame};
    roi_extractor::BatchResult batch_result = roi_pipeline_->processBatch(
        frames,
        roi_extractor::DetectMode::ALL,
        false
    );

    cv::Mat processed_frame;

    if (!batch_result.frames.empty()) {
        const auto& processed = batch_result.frames[0];

        frame_result.has_fire = processed.has_fire;
        frame_result.has_smoke = processed.has_smoke;

        if (config_.detector_config.enable_roi_extraction) {
            processed_frame = processed.data;
            frame_result.roi_frame = processed.data.clone();
        } else {
            processed_frame = frame;
            frame_result.roi_frame = frame.clone();
        }

        // 坐标缩放
        const float scale_x = static_cast<float>(frame.cols) / 640.0f;
        const float scale_y = static_cast<float>(frame.rows) / 640.0f;

        for (const auto& det : processed.detections) {
            BoundingBox bbox;
            bbox.x1 = static_cast<float>(det.bbox.x) * scale_x;
            bbox.y1 = static_cast<float>(det.bbox.y) * scale_y;
            bbox.x2 = static_cast<float>(det.bbox.x + det.bbox.width) * scale_x;
            bbox.y2 = static_cast<float>(det.bbox.y + det.bbox.height) * scale_y;
            bbox.confidence = det.confidence;
            bbox.class_id = static_cast<DetectionClass>(det.class_id);
            frame_result.detections.push_back(bbox);
        }
    } else {
        processed_frame = frame;
        frame_result.roi_frame = frame.clone();
    }

    return processed_frame;
}

/**
 * @brief 为场景创建并初始化 ConvLSTMClassifier
 */
bool SceneController::Impl::initializeSceneClassifier(SceneContext& ctx) {
    if (ctx.classifier) {
        return true;  // 已经初始化
    }

    int seq_len = config_.detector_config.convlstm_seq_len;
    ctx.classifier = std::make_unique<convlstm::ConvLSTMClassifier>(
        *convlstm_engine_, seq_len
    );

    if (!ctx.classifier->initialize()) {
        std::cerr << "SceneController: Failed to initialize classifier for scene "
                  << ctx.scene_id << std::endl;
        ctx.classifier.reset();
        return false;
    }

    return true;
}

//=============================================================================
// 帧处理
//=============================================================================

SceneDetectionResult SceneController::Impl::processFrame(
    int32_t scene_id,
    const cv::Mat& frame,
    int64_t timestamp_ms
) {
    SceneDetectionResult result;
    result.scene_id = scene_id;
    result.last_frame_timestamp = timestamp_ms;

    if (!initialized_) {
        std::cerr << "SceneController: Not initialized" << std::endl;
        return result;
    }

    if (frame.empty()) {
        std::cerr << "SceneController: Empty frame" << std::endl;
        return result;
    }

    std::lock_guard<std::mutex> lock(scenes_mutex_);

    // 自动注册不存在的场景
    if (scenes_.find(scene_id) == scenes_.end()) {
        scenes_[scene_id] = std::make_unique<SceneContext>(scene_id);
        std::cout << "SceneController: Auto-registered scene " << scene_id << std::endl;
    }

    SceneContext& ctx = *scenes_[scene_id];

    // 时间间隔检查
    checkAndPurgeOldFrames(ctx, timestamp_ms);

    // 排序插入时间戳
    insertTimestamp(ctx, timestamp_ms);

    // 更新状态
    result.queue_size = static_cast<int>(ctx.timestamp_queue.size());
    ctx.last_processed_timestamp = timestamp_ms;

    // 运行空间检测
    FrameResult frame_result;
    cv::Mat processed_frame = runDetection(frame, frame_result);

    result.has_fire = frame_result.has_fire;
    result.has_smoke = frame_result.has_smoke;
    ctx.last_roi_frame = frame_result.roi_frame;

    // 提取火焰和烟雾检测框
    for (const auto& bbox : frame_result.detections) {
        if (bbox.class_id == DetectionClass::FIRE) {
            result.fire_boxes.push_back(bbox);
        } else if (bbox.class_id == DetectionClass::SMOKE) {
            result.smoke_boxes.push_back(bbox);
        }
    }

    // 时序分析
    if (result.has_fire || result.has_smoke) {
        // 确保分类器已初始化
        if (!ctx.classifier && !initializeSceneClassifier(ctx)) {
            std::cerr << "SceneController: Failed to initialize classifier" << std::endl;
            return result;
        }

        // 添加帧到分类器缓冲区（内部完成预处理）
        ctx.classifier->addFrame(processed_frame);

        // 缓冲区满后执行推理
        if (ctx.classifier->isBufferFull()) {
            if (ctx.classifier->infer()) {
                const auto& cls_result = ctx.classifier->getResult();

                result.temporal_valid = true;
                result.temporal_class = static_cast<TemporalClass>(
                    static_cast<int>(cls_result.classification)
                );
                result.temporal_confidence = cls_result.confidence;
                result.is_fire_confirmed = (cls_result.classification == convlstm::Classification::DYNAMIC);

                for (int i = 0; i < 3; ++i) {
                    result.class_scores[i] = cls_result.class_scores[i];
                }
            }
        }
    }

    result.convlstm_buffer_size = ctx.classifier ? ctx.classifier->getBufferSize() : 0;

    return result;
}

SceneDetectionResult SceneController::Impl::processFrameRaw(
    int32_t scene_id,
    const uint8_t* data,
    int width, int height, int channels,
    int64_t timestamp_ms
) {
    if (channels != 3) {
        std::cerr << "SceneController: Only 3-channel images are supported" << std::endl;
        return SceneDetectionResult();
    }

    cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(data));
    return processFrame(scene_id, frame, timestamp_ms);
}

//=============================================================================
// 场景状态
//=============================================================================

int SceneController::Impl::getSceneQueueSize(int32_t scene_id) const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it == scenes_.end()) {
        return -1;
    }
    return static_cast<int>(it->second->timestamp_queue.size());
}

int SceneController::Impl::getSceneBufferSize(int32_t scene_id) const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it == scenes_.end()) {
        return -1;
    }
    return it->second->classifier ? it->second->classifier->getBufferSize() : 0;
}

bool SceneController::Impl::isSceneReadyForInference(int32_t scene_id) const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it == scenes_.end()) {
        return false;
    }
    return it->second->classifier && it->second->classifier->isBufferFull();
}

void SceneController::Impl::resetScene(int32_t scene_id) {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it != scenes_.end()) {
        it->second->reset();
        std::cout << "SceneController: Reset scene " << scene_id << std::endl;
    }
}

void SceneController::Impl::resetAllScenes() {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    for (auto& pair : scenes_) {
        pair.second->reset();
    }
    std::cout << "SceneController: Reset all scenes" << std::endl;
}

cv::Mat SceneController::Impl::getLastRoiFrame(int32_t scene_id) const {
    std::lock_guard<std::mutex> lock(scenes_mutex_);

    auto it = scenes_.find(scene_id);
    if (it == scenes_.end()) {
        return cv::Mat();
    }
    return it->second->last_roi_frame.clone();
}

//=============================================================================
// SceneController 公开接口实现
//=============================================================================

SceneController::SceneController()
    : impl_(std::make_unique<Impl>())
{
}

SceneController::~SceneController() = default;

SceneController::SceneController(SceneController&&) noexcept = default;

SceneController& SceneController::operator=(SceneController&&) noexcept = default;

bool SceneController::initialize(const SceneConfig& config) {
    return impl_->initialize(config);
}

bool SceneController::isReady() const {
    return impl_->isReady();
}

bool SceneController::registerScene(int32_t scene_id) {
    return impl_->registerScene(scene_id);
}

bool SceneController::unregisterScene(int32_t scene_id) {
    return impl_->unregisterScene(scene_id);
}

bool SceneController::hasScene(int32_t scene_id) const {
    return impl_->hasScene(scene_id);
}

std::vector<int32_t> SceneController::getSceneIds() const {
    return impl_->getSceneIds();
}

size_t SceneController::getSceneCount() const {
    return impl_->getSceneCount();
}

SceneDetectionResult SceneController::processFrame(
    int32_t scene_id,
    const cv::Mat& frame,
    int64_t timestamp_ms
) {
    return impl_->processFrame(scene_id, frame, timestamp_ms);
}

SceneDetectionResult SceneController::processFrameRaw(
    int32_t scene_id,
    const uint8_t* data,
    int width, int height, int channels,
    int64_t timestamp_ms
) {
    return impl_->processFrameRaw(scene_id, data, width, height, channels, timestamp_ms);
}

int SceneController::getSceneQueueSize(int32_t scene_id) const {
    return impl_->getSceneQueueSize(scene_id);
}

int SceneController::getSceneBufferSize(int32_t scene_id) const {
    return impl_->getSceneBufferSize(scene_id);
}

bool SceneController::isSceneReadyForInference(int32_t scene_id) const {
    return impl_->isSceneReadyForInference(scene_id);
}

void SceneController::resetScene(int32_t scene_id) {
    impl_->resetScene(scene_id);
}

void SceneController::resetAllScenes() {
    impl_->resetAllScenes();
}

cv::Mat SceneController::getLastRoiFrame(int32_t scene_id) const {
    return impl_->getLastRoiFrame(scene_id);
}

}  // namespace fire_detection
