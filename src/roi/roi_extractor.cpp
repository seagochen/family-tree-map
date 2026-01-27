/**
 * @file roi_extractor.cpp
 * @brief Implementation of ROI extraction module
 */

#include "roi/roi_extractor.h"
#include <algorithm>
#include <cmath>

namespace roi_extractor {

// =============================================================================
// ROIDetection Implementation
// =============================================================================

ROIDetection ROIDetection::fromDetection(const Detection& det, const cv::Mat& mask_data) {
    ROIDetection roi_det;
    roi_det.x1 = det.bbox.x1;
    roi_det.y1 = det.bbox.y1;
    roi_det.x2 = det.bbox.x2;
    roi_det.y2 = det.bbox.y2;
    roi_det.confidence = det.bbox.confidence;
    roi_det.class_id = det.bbox.class_id;
    roi_det.mask = mask_data.clone();
    return roi_det;
}

cv::Rect ROIDetection::getBBox() const {
    int ix1 = static_cast<int>(x1);
    int iy1 = static_cast<int>(y1);
    int ix2 = static_cast<int>(x2);
    int iy2 = static_cast<int>(y2);
    return cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1);
}

// =============================================================================
// Utility Functions Implementation
// =============================================================================

cv::Scalar getClassColor(int class_id) {
    static const std::vector<cv::Scalar> default_colors = {
        cv::Scalar(0, 0, 255),    // Red
        cv::Scalar(0, 255, 0),    // Green
        cv::Scalar(255, 0, 0),    // Blue
        cv::Scalar(0, 255, 255),  // Yellow
        cv::Scalar(255, 0, 255),  // Magenta
    };

    auto it = CLASS_COLORS.find(class_id);
    if (it != CLASS_COLORS.end()) {
        return it->second;
    }
    return default_colors[class_id % default_colors.size()];
}

std::vector<ROIDetection> extractDetectionsFromYolo(
    const std::vector<Detection>& detections,
    const std::vector<cv::Mat>& masks,
    const cv::Size& frame_shape,
    const std::unordered_set<int>& active_classes,
    float conf_threshold
) {
    std::vector<ROIDetection> result;
    int h = frame_shape.height;
    int w = frame_shape.width;

    // Process detections with masks
    if (!masks.empty() && masks.size() == detections.size()) {
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            if (active_classes.count(det.bbox.class_id) > 0 && det.bbox.confidence >= conf_threshold) {
                cv::Mat mask_resized;
                if (masks[i].size() != frame_shape) {
                    cv::resize(masks[i], mask_resized, frame_shape, 0, 0, cv::INTER_NEAREST);
                } else {
                    mask_resized = masks[i].clone();
                }

                // Binarize mask
                cv::Mat mask_binary;
                cv::threshold(mask_resized, mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
                mask_binary.convertTo(mask_binary, CV_8U);

                result.push_back(ROIDetection::fromDetection(det, mask_binary));
            }
        }
    }
    // Process detections without masks (bbox only)
    else {
        for (const auto& det : detections) {
            if (active_classes.count(det.bbox.class_id) > 0 && det.bbox.confidence >= conf_threshold) {
                result.push_back(ROIDetection::fromDetection(det));
            }
        }
    }

    return result;
}

std::optional<cv::Rect> computeBBoxUnion(const std::vector<cv::Rect>& bboxes) {
    if (bboxes.empty()) {
        return std::nullopt;
    }

    int x1_min = std::numeric_limits<int>::max();
    int y1_min = std::numeric_limits<int>::max();
    int x2_max = std::numeric_limits<int>::min();
    int y2_max = std::numeric_limits<int>::min();

    for (const auto& bbox : bboxes) {
        x1_min = std::min(x1_min, bbox.x);
        y1_min = std::min(y1_min, bbox.y);
        x2_max = std::max(x2_max, bbox.x + bbox.width);
        y2_max = std::max(y2_max, bbox.y + bbox.height);
    }

    return cv::Rect(x1_min, y1_min, x2_max - x1_min, y2_max - y1_min);
}

cv::Rect addBBoxPadding(
    const cv::Rect& bbox,
    const cv::Size& frame_shape,
    float padding_ratio
) {
    int h = frame_shape.height;
    int w = frame_shape.width;

    int pad_w = static_cast<int>(bbox.width * padding_ratio);
    int pad_h = static_cast<int>(bbox.height * padding_ratio);

    int x1 = std::max(0, bbox.x - pad_w);
    int y1 = std::max(0, bbox.y - pad_h);
    int x2 = std::min(w, bbox.x + bbox.width + pad_w);
    int y2 = std::min(h, bbox.y + bbox.height + pad_h);

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

cv::Mat applyBBoxMask(const cv::Mat& frame, const cv::Rect& bbox) {
    cv::Mat masked_frame = cv::Mat::zeros(frame.size(), frame.type());

    // Clip bbox to frame boundaries
    cv::Rect clipped_bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);

    if (clipped_bbox.area() > 0) {
        frame(clipped_bbox).copyTo(masked_frame(clipped_bbox));
    }

    return masked_frame;
}

cv::Mat applySegmentationMask(const cv::Mat& frame, const std::vector<ROIDetection>& detections) {
    if (detections.empty()) {
        return cv::Mat::zeros(frame.size(), frame.type());
    }

    int h = frame.rows;
    int w = frame.cols;
    cv::Mat combined_mask = cv::Mat::zeros(h, w, CV_8U);

    for (const auto& det : detections) {
        if (!det.mask.empty()) {
            cv::Mat mask_resized;
            if (det.mask.size() != cv::Size(w, h)) {
                cv::resize(det.mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
            } else {
                mask_resized = det.mask;
            }
            cv::max(combined_mask, mask_resized, combined_mask);
        }
    }

    cv::Mat masked_frame = frame.clone();
    masked_frame.setTo(cv::Scalar::all(0), combined_mask == 0);

    return masked_frame;
}

// =============================================================================
// MaskEMA Implementation
// =============================================================================

MaskEMA::MaskEMA(float alpha, int blur_kernel)
    : alpha_(alpha), blur_kernel_(blur_kernel), has_ema_mask_(false) {}

cv::Mat MaskEMA::update(const cv::Mat& current_mask) {
    cv::Mat current_float;
    current_mask.convertTo(current_float, CV_32F);

    if (!has_ema_mask_) {
        ema_mask_ = current_float.clone();
        has_ema_mask_ = true;
    } else {
        // EMA formula: new = alpha * current + (1 - alpha) * old
        ema_mask_ = alpha_ * current_float + (1.0f - alpha_) * ema_mask_;
    }

    // Threshold to get binary mask
    cv::Mat smoothed_mask;
    cv::threshold(ema_mask_, smoothed_mask, 0.5, 1.0, cv::THRESH_BINARY);
    smoothed_mask.convertTo(smoothed_mask, CV_8U);

    // Apply edge smoothing
    smoothed_mask = smoothEdges(smoothed_mask);

    return smoothed_mask;
}

cv::Mat MaskEMA::smoothEdges(const cv::Mat& mask) {
    if (cv::countNonZero(mask) == 0) {
        return mask;
    }

    int kernel_size = blur_kernel_;
    cv::Mat mask_uint8;
    mask.convertTo(mask_uint8, CV_8U);
    mask_uint8 *= 255;

    // Morphological closing to fill small holes
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat mask_closed;
    cv::morphologyEx(mask_uint8, mask_closed, cv::MORPH_CLOSE, kernel);

    // Gaussian blur to smooth edges
    cv::Mat mask_blurred;
    cv::GaussianBlur(mask_closed, mask_blurred, cv::Size(kernel_size, kernel_size), 0);

    // Re-threshold
    cv::Mat mask_smooth;
    cv::threshold(mask_blurred, mask_smooth, 127, 1, cv::THRESH_BINARY);
    mask_smooth.convertTo(mask_smooth, CV_8U);

    return mask_smooth;
}

void MaskEMA::reset() {
    ema_mask_.release();
    has_ema_mask_ = false;
}

// =============================================================================
// ROIPipeline Implementation
// =============================================================================

ROIPipeline::ROIPipeline(
    DetectionCallback detector,
    float confidence_threshold,
    bool use_ema,
    float ema_alpha,
    float bbox_padding
)
    : detector_(std::move(detector))
    , confidence_threshold_(confidence_threshold)
    , use_ema_(use_ema)
    , ema_alpha_(ema_alpha)
    , bbox_padding_(bbox_padding)
{}

BatchResult ROIPipeline::processBatch(
    const std::vector<cv::Mat>& frames,
    DetectMode detect_mode,
    bool skip_no_detection
) {
    if (frames.empty()) {
        return BatchResult{};
    }

    // 1. Preprocess all frames
    std::vector<cv::Mat> resized_frames;
    resized_frames.reserve(frames.size());
    for (const auto& f : frames) {
        resized_frames.push_back(preprocessFrame(f));
    }
    cv::Size shape(resized_frames[0].cols, resized_frames[0].rows);

    // 2. Run detection on all frames
    std::vector<std::vector<ROIDetection>> all_detections;
    all_detections.reserve(frames.size());
    for (const auto& f : resized_frames) {
        all_detections.push_back(detector_(f, confidence_threshold_));
    }

    // 3. Filter detections by mode
    std::vector<std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>> all_filtered;
    all_filtered.reserve(frames.size());
    for (const auto& dets : all_detections) {
        all_filtered.push_back(filterDetections(dets, detect_mode));
    }

    // 4. Compute smoke union bbox for entire batch
    auto smoke_union_bbox = computeBatchSmokeUnion(all_filtered, shape);

    // 5. Initialize EMA
    std::unique_ptr<MaskEMA> mask_ema;
    if (use_ema_) {
        mask_ema = std::make_unique<MaskEMA>(ema_alpha_);
    }

    // 6. Process each frame
    std::vector<ProcessedFrame> results;
    int frames_with_fire = 0;
    int frames_with_smoke = 0;

    for (size_t idx = 0; idx < resized_frames.size(); ++idx) {
        const auto& resized_frame = resized_frames[idx];
        const auto& [fire_dets, smoke_dets] = all_filtered[idx];

        bool has_fire = !fire_dets.empty();
        bool has_smoke = !smoke_dets.empty() && smoke_union_bbox.has_value();

        if (has_fire) {
            frames_with_fire++;
        }
        if (has_smoke) {
            frames_with_smoke++;
        }

        // Build masks
        cv::Mat fire_mask = has_fire ?
            buildFireMask(fire_dets, shape, mask_ema.get()) :
            cv::Mat::zeros(shape, CV_8U);
        cv::Mat smoke_mask = has_smoke ?
            buildSmokeMask(smoke_union_bbox, shape) :
            cv::Mat::zeros(shape, CV_8U);

        // Apply masks
        auto [output_frame, roi_type] = applyMaskToFrame(resized_frame, fire_mask, smoke_mask);

        ProcessedFrame result;
        result.frame_idx = static_cast<int>(idx);
        result.data = output_frame;
        result.roi_type = roi_type;
        result.has_fire = has_fire;
        result.has_smoke = has_smoke;
        result.detections = buildDetectionInfo(fire_dets, smoke_dets);

        if (!skip_no_detection || result.hasDetections()) {
            results.push_back(std::move(result));
        }
    }

    BatchResult batch_result;
    batch_result.frames = std::move(results);
    batch_result.smoke_union_bbox = smoke_union_bbox;
    batch_result.frames_with_fire = frames_with_fire;
    batch_result.frames_with_smoke = frames_with_smoke;

    return batch_result;
}

cv::Mat ROIPipeline::preprocessFrame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INFERENCE_WIDTH, INFERENCE_HEIGHT), 0, 0, cv::INTER_LINEAR);
    return resized;
}

std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>
ROIPipeline::filterDetections(const std::vector<ROIDetection>& detections, DetectMode mode) {
    std::vector<ROIDetection> fire_dets;
    std::vector<ROIDetection> smoke_dets;

    for (const auto& det : detections) {
        if (mode == DetectMode::FIRE || mode == DetectMode::ALL) {
            if (SEGMENT_CLASSES.count(det.class_id) > 0) {
                fire_dets.push_back(det);
            }
        }
        if (mode == DetectMode::SMOKE || mode == DetectMode::ALL) {
            if (BBOX_CLASSES.count(det.class_id) > 0) {
                smoke_dets.push_back(det);
            }
        }
    }

    if (mode == DetectMode::FIRE) {
        smoke_dets.clear();
    } else if (mode == DetectMode::SMOKE) {
        fire_dets.clear();
    }

    return {fire_dets, smoke_dets};
}

std::optional<cv::Rect> ROIPipeline::computeBatchSmokeUnion(
    const std::vector<std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>>& all_filtered,
    const cv::Size& shape
) {
    std::vector<cv::Rect> all_smoke_bboxes;

    for (const auto& [fire_dets, smoke_dets] : all_filtered) {
        for (const auto& det : smoke_dets) {
            all_smoke_bboxes.push_back(det.getBBox());
        }
    }

    if (all_smoke_bboxes.empty()) {
        return std::nullopt;
    }

    auto union_bbox = computeBBoxUnion(all_smoke_bboxes);
    if (union_bbox.has_value()) {
        return addBBoxPadding(union_bbox.value(), shape, bbox_padding_);
    }

    return std::nullopt;
}

cv::Mat ROIPipeline::buildFireMask(
    const std::vector<ROIDetection>& fire_dets,
    const cv::Size& shape,
    MaskEMA* mask_ema
) {
    int h = shape.height;
    int w = shape.width;
    cv::Mat fire_mask = cv::Mat::zeros(h, w, CV_8U);

    for (const auto& det : fire_dets) {
        if (!det.mask.empty()) {
            cv::Mat mask_resized;
            if (det.mask.size() != cv::Size(w, h)) {
                cv::resize(det.mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
            } else {
                mask_resized = det.mask;
            }
            cv::max(fire_mask, mask_resized, fire_mask);
        }
    }

    // Apply EMA smoothing
    if (mask_ema != nullptr) {
        fire_mask = mask_ema->update(fire_mask);
    }

    return fire_mask;
}

cv::Mat ROIPipeline::buildSmokeMask(
    const std::optional<cv::Rect>& smoke_union_bbox,
    const cv::Size& shape
) {
    int h = shape.height;
    int w = shape.width;
    cv::Mat smoke_mask = cv::Mat::zeros(h, w, CV_8U);

    if (smoke_union_bbox.has_value()) {
        cv::Rect bbox = smoke_union_bbox.value();
        // Clip to frame boundaries
        bbox = bbox & cv::Rect(0, 0, w, h);
        if (bbox.area() > 0) {
            smoke_mask(bbox).setTo(1);
        }
    }

    return smoke_mask;
}

std::pair<cv::Mat, std::string> ROIPipeline::applyMaskToFrame(
    const cv::Mat& frame,
    const cv::Mat& fire_mask,
    const cv::Mat& smoke_mask
) {
    cv::Mat combined_mask;
    cv::max(fire_mask, smoke_mask, combined_mask);

    bool has_fire = cv::countNonZero(fire_mask) > 0;
    bool has_smoke = cv::countNonZero(smoke_mask) > 0;

    if (cv::countNonZero(combined_mask) > 0) {
        cv::Mat output_frame = frame.clone();
        output_frame.setTo(cv::Scalar::all(0), combined_mask == 0);

        std::string roi_type;
        if (has_fire && has_smoke) {
            roi_type = "fire+smoke";
        } else if (has_fire) {
            roi_type = "fire";
        } else {
            roi_type = "smoke";
        }

        return {output_frame, roi_type};
    } else {
        return {cv::Mat::zeros(frame.size(), frame.type()), "none"};
    }
}

std::vector<DetectionInfo> ROIPipeline::buildDetectionInfo(
    const std::vector<ROIDetection>& fire_dets,
    const std::vector<ROIDetection>& smoke_dets
) {
    std::vector<DetectionInfo> result;
    result.reserve(fire_dets.size() + smoke_dets.size());

    auto addDetections = [&result](const std::vector<ROIDetection>& dets) {
        for (const auto& det : dets) {
            DetectionInfo info;
            info.class_id = det.class_id;
            auto it = CLASS_NAMES.find(det.class_id);
            info.class_name = (it != CLASS_NAMES.end()) ? it->second : "unknown";
            info.confidence = det.confidence;
            info.bbox = det.getBBox();
            result.push_back(info);
        }
    };

    addDetections(fire_dets);
    addDetections(smoke_dets);

    return result;
}

}  // namespace roi_extractor
