/**
 * @file roi_extractor.h
 * @brief ROI (Region of Interest) extraction module for fire/smoke detection
 *
 * Provides functionality to extract fire and smoke regions from video frames:
 * - fire (class 0): Uses segmentation mask + EMA smoothing
 * - flower (class 1): Ignored (negative samples)
 * - person (class 2): Ignored (negative samples)
 * - smoke (class 3): Uses batch-wise bbox union strategy
 *
 * Processing pipeline:
 * 1. Run detection on all frames
 * 2. Compute smoke union bbox for entire batch
 * 3. Apply fire mask (EMA smoothed) + smoke mask (union bbox) for each frame
 * 4. Set unwanted pixels to 0
 */

#ifndef TRT_ENGINE_ROI_EXTRACTOR_H
#define TRT_ENGINE_ROI_EXTRACTOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <memory>
#include <optional>
#include <unordered_set>
#include <unordered_map>

#include "detector/yolov8_postprocess.h"

namespace roi_extractor {

// =============================================================================
// Constants (corresponding to Python constants.py)
// =============================================================================

// Class names (corresponding to data.yaml)
inline const std::unordered_map<int, std::string> CLASS_NAMES = {
    {0, "fire"},
    {1, "flower"},  // Negative sample, ignored during processing
    {2, "person"},  // Negative sample, ignored during processing
    {3, "smoke"}
};

// Target classes (classes to process)
inline const std::unordered_set<int> TARGET_CLASSES = {0, 3};  // fire, smoke

// Classes using segmentation mask (fire)
inline const std::unordered_set<int> SEGMENT_CLASSES = {0};

// Classes using bbox (smoke)
inline const std::unordered_set<int> BBOX_CLASSES = {3};

// Negative sample classes (ignored)
inline const std::unordered_set<int> NEGATIVE_CLASSES = {1, 2};  // flower, person

// Class colors (BGR format for visualization)
inline const std::unordered_map<int, cv::Scalar> CLASS_COLORS = {
    {0, cv::Scalar(0, 0, 255)},     // fire - red
    {1, cv::Scalar(0, 255, 0)},     // flower - green (visualization only)
    {2, cv::Scalar(255, 0, 0)},     // person - blue (visualization only)
    {3, cv::Scalar(0, 255, 255)}    // smoke - yellow
};

// YOLO inference size
constexpr int INFERENCE_WIDTH = 640;
constexpr int INFERENCE_HEIGHT = 640;

// Default confidence threshold
constexpr float DEFAULT_CONFIDENCE_THRESHOLD = 0.5f;

// EMA smoothing factor (for mask smoothing)
// Lower value = smoother but more delay (e.g., 0.2 ~ 5 frame delay)
// Higher value = faster response but less smooth (e.g., 0.5 ~ 2 frame delay)
constexpr float EMA_ALPHA = 0.2f;

// Gaussian blur kernel size (for mask edge smoothing)
constexpr int MASK_BLUR_KERNEL_SIZE = 5;

// Bbox padding ratio (for smoke bbox)
constexpr float BBOX_PADDING_RATIO = 0.05f;  // 5%

// Video analysis sample rate (sample every N frames)
constexpr int DEFAULT_SAMPLE_RATE = 5;

// =============================================================================
// Detection Mode
// =============================================================================

enum class DetectMode {
    ALL,    // Detect both fire and smoke
    FIRE,   // Detect fire only
    SMOKE   // Detect smoke only
};

// =============================================================================
// ROI Detection (extends Detection from yolov8_postprocess.h)
// =============================================================================

/**
 * @brief Extended detection with segmentation mask support
 */
struct ROIDetection {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
    cv::Mat mask;  // Optional segmentation mask

    // Construct from basic Detection
    static ROIDetection fromDetection(const Detection& det, const cv::Mat& mask_data = cv::Mat());

    // Get integer bbox
    cv::Rect getBBox() const;

    // Get bbox dimensions
    float getWidth() const { return x2 - x1; }
    float getHeight() const { return y2 - y1; }
    float getArea() const { return getWidth() * getHeight(); }
};

// =============================================================================
// Utility Functions
// =============================================================================

/**
 * @brief Get color for a class ID
 */
cv::Scalar getClassColor(int class_id);

/**
 * @brief Extract ROIDetection objects from YOLO results
 *
 * @param detections YOLO detection results
 * @param masks Segmentation masks (if available)
 * @param frame_shape Frame size (height, width)
 * @param active_classes Classes to include
 * @param conf_threshold Minimum confidence threshold
 * @return List of ROIDetection objects with masks (if available)
 */
std::vector<ROIDetection> extractDetectionsFromYolo(
    const std::vector<Detection>& detections,
    const std::vector<cv::Mat>& masks,
    const cv::Size& frame_shape,
    const std::unordered_set<int>& active_classes,
    float conf_threshold = 0.5f
);

/**
 * @brief Compute union bbox of all input bboxes
 *
 * @param bboxes List of (x1, y1, x2, y2) tuples
 * @return Union bbox, or nullopt if list is empty
 */
std::optional<cv::Rect> computeBBoxUnion(const std::vector<cv::Rect>& bboxes);

/**
 * @brief Add padding to bbox and clip to frame boundaries
 *
 * @param bbox Input bbox
 * @param frame_shape Frame size (height, width)
 * @param padding_ratio Padding ratio
 * @return Padded bbox clipped to frame boundaries
 */
cv::Rect addBBoxPadding(
    const cv::Rect& bbox,
    const cv::Size& frame_shape,
    float padding_ratio = BBOX_PADDING_RATIO
);

/**
 * @brief Apply bbox mask to frame (set pixels outside bbox to 0)
 */
cv::Mat applyBBoxMask(const cv::Mat& frame, const cv::Rect& bbox);

/**
 * @brief Apply segmentation mask to frame
 */
cv::Mat applySegmentationMask(const cv::Mat& frame, const std::vector<ROIDetection>& detections);

// =============================================================================
// MaskEMA - Exponential Moving Average for Mask Smoothing
// =============================================================================

/**
 * @brief Mask EMA smoother
 *
 * Reduces mask flickering through temporal smoothing.
 * Also uses morphological operations and Gaussian blur for edge smoothing.
 */
class MaskEMA {
public:
    /**
     * @brief Constructor
     * @param alpha EMA smoothing factor (0-1)
     * @param blur_kernel Gaussian blur kernel size
     */
    explicit MaskEMA(float alpha = EMA_ALPHA, int blur_kernel = MASK_BLUR_KERNEL_SIZE);

    /**
     * @brief Update EMA mask and return smoothed mask
     *
     * @param current_mask Current frame's binary mask (H, W), values 0 or 1
     * @return Smoothed binary mask with smooth edges
     */
    cv::Mat update(const cv::Mat& current_mask);

    /**
     * @brief Reset EMA state (call when processing new video)
     */
    void reset();

private:
    float alpha_;
    int blur_kernel_;
    cv::Mat ema_mask_;
    bool has_ema_mask_ = false;

    /**
     * @brief Smooth mask edges using morphological ops and Gaussian blur
     */
    cv::Mat smoothEdges(const cv::Mat& mask);
};

// =============================================================================
// ProcessedFrame - Processed frame data
// =============================================================================

/**
 * @brief Detection info for a frame
 */
struct DetectionInfo {
    int class_id;
    std::string class_name;
    float confidence;
    cv::Rect bbox;
};

/**
 * @brief Processed frame result
 */
struct ProcessedFrame {
    int frame_idx;                          // Frame index in batch
    cv::Mat data;                           // Processed frame (after ROI extraction)
    std::string roi_type;                   // ROI type: "fire", "smoke", "fire+smoke", "none"
    bool has_fire = false;
    bool has_smoke = false;
    std::vector<DetectionInfo> detections;  // Detection info list

    bool hasDetections() const { return has_fire || has_smoke; }
};

// =============================================================================
// BatchResult - Batch processing result
// =============================================================================

/**
 * @brief Batch processing result
 */
struct BatchResult {
    std::vector<ProcessedFrame> frames;           // Processed frames
    std::optional<cv::Rect> smoke_union_bbox;     // Smoke union bbox
    int frames_with_fire = 0;
    int frames_with_smoke = 0;
};

// =============================================================================
// ROIPipeline - ROI Extraction Pipeline
// =============================================================================

// Forward declaration of detection callback type
using DetectionCallback = std::function<std::vector<ROIDetection>(
    const cv::Mat& frame, float conf_threshold)>;

/**
 * @brief ROI extraction pipeline
 *
 * Processes a batch of consecutive frames, extracts fire/smoke regions,
 * and sets unwanted pixels to 0.
 *
 * Usage:
 *     // With external detector callback
 *     auto detector = [&model](const cv::Mat& frame, float conf) {
 *         return runYoloDetection(model, frame, conf);
 *     };
 *     ROIPipeline pipeline(detector);
 *
 *     std::vector<cv::Mat> frames = {frame1, frame2, ..., frame10};
 *     BatchResult result = pipeline.processBatch(frames);
 *
 *     for (const auto& processed : result.frames) {
 *         cv::imshow("ROI", processed.data);
 *     }
 */
class ROIPipeline {
public:
    /**
     * @brief Constructor
     *
     * @param detector Detection callback function
     * @param confidence_threshold Confidence threshold
     * @param use_ema Whether to use EMA smoothing for fire mask
     * @param ema_alpha EMA smoothing factor
     * @param bbox_padding Smoke bbox padding ratio
     */
    explicit ROIPipeline(
        DetectionCallback detector,
        float confidence_threshold = DEFAULT_CONFIDENCE_THRESHOLD,
        bool use_ema = true,
        float ema_alpha = EMA_ALPHA,
        float bbox_padding = BBOX_PADDING_RATIO
    );

    /**
     * @brief Process a batch of consecutive frames
     *
     * @param frames Consecutive frame list (BGR)
     * @param detect_mode Detection mode (ALL, FIRE, SMOKE)
     * @param skip_no_detection Skip frames without detections
     * @return BatchResult
     */
    BatchResult processBatch(
        const std::vector<cv::Mat>& frames,
        DetectMode detect_mode = DetectMode::ALL,
        bool skip_no_detection = false
    );

    // Setters
    void setConfidenceThreshold(float threshold) { confidence_threshold_ = threshold; }
    void setUseEma(bool use_ema) { use_ema_ = use_ema; }
    void setEmaAlpha(float alpha) { ema_alpha_ = alpha; }
    void setBBoxPadding(float padding) { bbox_padding_ = padding; }

private:
    DetectionCallback detector_;
    float confidence_threshold_;
    bool use_ema_;
    float ema_alpha_;
    float bbox_padding_;

    // Preprocess frame (resize to inference size)
    cv::Mat preprocessFrame(const cv::Mat& frame);

    // Filter detections by mode
    std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>
    filterDetections(const std::vector<ROIDetection>& detections, DetectMode mode);

    // Compute batch smoke union bbox
    std::optional<cv::Rect> computeBatchSmokeUnion(
        const std::vector<std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>>& all_filtered,
        const cv::Size& shape
    );

    // Build fire mask from detections (with EMA smoothing)
    cv::Mat buildFireMask(
        const std::vector<ROIDetection>& fire_dets,
        const cv::Size& shape,
        MaskEMA* mask_ema
    );

    // Build smoke mask from union bbox
    cv::Mat buildSmokeMask(
        const std::optional<cv::Rect>& smoke_union_bbox,
        const cv::Size& shape
    );

    // Apply mask to frame (set unwanted pixels to 0)
    std::pair<cv::Mat, std::string> applyMaskToFrame(
        const cv::Mat& frame,
        const cv::Mat& fire_mask,
        const cv::Mat& smoke_mask
    );

    // Build detection info list
    std::vector<DetectionInfo> buildDetectionInfo(
        const std::vector<ROIDetection>& fire_dets,
        const std::vector<ROIDetection>& smoke_dets
    );
};

}  // namespace roi_extractor

#endif // TRT_ENGINE_ROI_EXTRACTOR_H
