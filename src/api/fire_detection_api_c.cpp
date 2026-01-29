/**
 * @file fire_detection_api_c.cpp
 * @brief Fire Detection API C 接口实现
 */

#include "fire_detection_api.h"
#include <cstring>
#include <algorithm>

//=============================================================================
// C 接口实现
//=============================================================================

extern "C" {

FireDetectorHandle fire_detector_create(void) {
    return new fire_detection::FireDetector();
}

void fire_detector_destroy(FireDetectorHandle handle) {
    if (handle) {
        delete static_cast<fire_detection::FireDetector*>(handle);
    }
}

int fire_detector_initialize(
    FireDetectorHandle handle,
    const FireDetectorConfig* config
) {
    if (!handle || !config) {
        return 0;
    }

    auto* detector = static_cast<fire_detection::FireDetector*>(handle);

    fire_detection::Config cpp_config;
    cpp_config.yolo_engine_path = config->yolo_engine_path ? config->yolo_engine_path : "";
    cpp_config.convlstm_engine_path = config->convlstm_engine_path ? config->convlstm_engine_path : "";
    cpp_config.confidence_threshold = config->confidence_threshold;
    cpp_config.iou_threshold = config->iou_threshold;
    cpp_config.convlstm_seq_len = config->convlstm_seq_len;
    cpp_config.enable_roi_extraction = config->enable_roi_extraction != 0;
    cpp_config.ema_alpha = config->ema_alpha;
    cpp_config.bbox_padding = config->bbox_padding;

    return detector->initialize(cpp_config) ? 1 : 0;
}

int fire_detector_process_frame(
    FireDetectorHandle handle,
    const unsigned char* data,
    int width, int height, int channels,
    FireDetectionResultC* result
) {
    if (!handle || !data || !result) {
        return 0;
    }

    auto* detector = static_cast<fire_detection::FireDetector*>(handle);

    auto cpp_result = detector->processFrameRaw(data, width, height, channels);

    // ========== 填充原有字段 ==========
    result->has_fire = cpp_result.frame.has_fire ? 1 : 0;
    result->has_smoke = cpp_result.frame.has_smoke ? 1 : 0;
    result->temporal_valid = cpp_result.temporal_valid ? 1 : 0;
    result->temporal_class = static_cast<int>(cpp_result.temporal_class);
    result->temporal_confidence = cpp_result.temporal_confidence;
    std::memcpy(result->class_scores, cpp_result.class_scores, sizeof(result->class_scores));

    // ========== 填充检测框信息 ==========
    const auto& detections = cpp_result.frame.detections;

    // 限制检测数量不超过最大值
    result->detection_count = static_cast<int>(
        std::min(detections.size(), static_cast<size_t>(FIRE_DETECTION_MAX_BOXES))
    );

    // 复制检测框数据
    for (int i = 0; i < result->detection_count; ++i) {
        const auto& src = detections[i];
        BoundingBoxC& dst = result->detections[i];

        dst.x1 = src.x1;
        dst.y1 = src.y1;
        dst.x2 = src.x2;
        dst.y2 = src.y2;
        dst.confidence = src.confidence;
        dst.class_id = static_cast<int>(src.class_id);
    }

    // 清零未使用的数组元素（提高安全性）
    if (result->detection_count < FIRE_DETECTION_MAX_BOXES) {
        std::memset(
            &result->detections[result->detection_count],
            0,
            (FIRE_DETECTION_MAX_BOXES - result->detection_count) * sizeof(BoundingBoxC)
        );
    }

    return 1;
}

int fire_detector_get_roi_frame(
    FireDetectorHandle handle,
    unsigned char* output_data,
    int* width, int* height
) {
    if (!handle || !width || !height) {
        return 0;
    }

    auto* detector = static_cast<fire_detection::FireDetector*>(handle);
    const cv::Mat& roi_frame = detector->getLastRoiFrame();

    if (roi_frame.empty()) {
        *width = 0;
        *height = 0;
        return 0;
    }

    *width = roi_frame.cols;
    *height = roi_frame.rows;

    if (output_data) {
        size_t data_size = roi_frame.total() * roi_frame.elemSize();
        std::memcpy(output_data, roi_frame.data, data_size);
    }

    return 1;
}

void fire_detector_reset(FireDetectorHandle handle) {
    if (handle) {
        auto* detector = static_cast<fire_detection::FireDetector*>(handle);
        detector->reset();
    }
}

int fire_detector_is_ready(FireDetectorHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* detector = static_cast<fire_detection::FireDetector*>(handle);
    return detector->isReady() ? 1 : 0;
}

int fire_detector_get_buffer_size(FireDetectorHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* detector = static_cast<fire_detection::FireDetector*>(handle);
    return detector->getBufferSize();
}

int fire_detector_is_buffer_full(FireDetectorHandle handle) {
    if (!handle) {
        return 0;
    }
    auto* detector = static_cast<fire_detection::FireDetector*>(handle);
    return detector->isBufferFull() ? 1 : 0;
}

int fire_detection_result_get_fire_boxes(
    const FireDetectionResultC* result,
    float confidence_threshold,
    BoundingBoxC* output_boxes,
    int max_boxes
) {
    if (!result || !output_boxes || max_boxes <= 0) {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < result->detection_count && count < max_boxes; ++i) {
        const BoundingBoxC& box = result->detections[i];
        // class_id 0 = FIRE
        if (box.class_id == 0 && box.confidence >= confidence_threshold) {
            output_boxes[count++] = box;
        }
    }
    return count;
}

int fire_detection_result_get_smoke_boxes(
    const FireDetectionResultC* result,
    float confidence_threshold,
    BoundingBoxC* output_boxes,
    int max_boxes
) {
    if (!result || !output_boxes || max_boxes <= 0) {
        return 0;
    }

    int count = 0;
    for (int i = 0; i < result->detection_count && count < max_boxes; ++i) {
        const BoundingBoxC& box = result->detections[i];
        // class_id 2 = SMOKE
        if (box.class_id == 2 && box.confidence >= confidence_threshold) {
            output_boxes[count++] = box;
        }
    }
    return count;
}

const char* fire_detector_get_version(void) {
    return fire_detection::getVersion();
}

}  // extern "C"
