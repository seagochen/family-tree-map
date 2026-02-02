/**
 * @file scene_controller_c.cpp
 * @brief 场景控制器C语言接口实现
 *
 * 提供纯C接口，便于FFI绑定（Python、C#等）。
 */

#include "scene/scene_controller.h"
#include <iostream>
#include <algorithm>

using namespace fire_detection;

//=============================================================================
// C 语言接口实现
//=============================================================================

extern "C" {

SceneControllerHandle scene_controller_create(void) {
    try {
        return new SceneController();
    } catch (const std::exception& e) {
        std::cerr << "scene_controller_create failed: " << e.what() << std::endl;
        return nullptr;
    }
}

void scene_controller_destroy(SceneControllerHandle handle) {
    if (handle) {
        delete static_cast<SceneController*>(handle);
    }
}

int scene_controller_initialize(
    SceneControllerHandle handle,
    const SceneControllerConfig* config
) {
    if (!handle || !config) {
        return 0;
    }

    try {
        SceneController* controller = static_cast<SceneController*>(handle);

        SceneConfig cpp_config;

        // 转换检测器配置
        if (config->detector_config.yolo_engine_path) {
            cpp_config.detector_config.yolo_engine_path = config->detector_config.yolo_engine_path;
        }
        if (config->detector_config.convlstm_engine_path) {
            cpp_config.detector_config.convlstm_engine_path = config->detector_config.convlstm_engine_path;
        }
        cpp_config.detector_config.confidence_threshold = config->detector_config.confidence_threshold;
        cpp_config.detector_config.iou_threshold = config->detector_config.iou_threshold;
        cpp_config.detector_config.convlstm_seq_len = config->detector_config.convlstm_seq_len;
        cpp_config.detector_config.enable_roi_extraction = config->detector_config.enable_roi_extraction != 0;
        cpp_config.detector_config.ema_alpha = config->detector_config.ema_alpha;
        cpp_config.detector_config.bbox_padding = config->detector_config.bbox_padding;

        // 转换场景配置
        cpp_config.max_frame_gap_ms = config->max_frame_gap_ms;
        cpp_config.min_sample_interval_ms = config->min_sample_interval_ms;
        cpp_config.max_queue_size = config->max_queue_size;

        return controller->initialize(cpp_config) ? 1 : 0;
    } catch (const std::exception& e) {
        std::cerr << "scene_controller_initialize failed: " << e.what() << std::endl;
        return 0;
    }
}

int scene_controller_is_ready(SceneControllerHandle handle) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->isReady() ? 1 : 0;
}

int scene_controller_register_scene(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->registerScene(scene_id) ? 1 : 0;
}

int scene_controller_unregister_scene(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->unregisterScene(scene_id) ? 1 : 0;
}

int scene_controller_has_scene(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->hasScene(scene_id) ? 1 : 0;
}

int scene_controller_get_scene_count(SceneControllerHandle handle) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return static_cast<int>(controller->getSceneCount());
}

int scene_controller_process_frame(
    SceneControllerHandle handle,
    int32_t scene_id,
    const unsigned char* data,
    int width, int height, int channels,
    int64_t timestamp_ms,
    SceneDetectionResultC* result
) {
    if (!handle || !data || !result) {
        return 0;
    }

    try {
        SceneController* controller = static_cast<SceneController*>(handle);

        SceneDetectionResult cpp_result = controller->processFrameRaw(
            scene_id, data, width, height, channels, timestamp_ms
        );

        // 转换结果
        result->scene_id = cpp_result.scene_id;
        result->has_fire = cpp_result.has_fire ? 1 : 0;
        result->has_smoke = cpp_result.has_smoke ? 1 : 0;

        // 复制火焰检测框
        result->fire_box_count = std::min(
            static_cast<int>(cpp_result.fire_boxes.size()),
            FIRE_DETECTION_MAX_BOXES
        );
        for (int i = 0; i < result->fire_box_count; ++i) {
            const auto& src = cpp_result.fire_boxes[i];
            result->fire_boxes[i].x1 = src.x1;
            result->fire_boxes[i].y1 = src.y1;
            result->fire_boxes[i].x2 = src.x2;
            result->fire_boxes[i].y2 = src.y2;
            result->fire_boxes[i].confidence = src.confidence;
            result->fire_boxes[i].class_id = static_cast<int>(src.class_id);
        }

        // 复制烟雾检测框
        result->smoke_box_count = std::min(
            static_cast<int>(cpp_result.smoke_boxes.size()),
            FIRE_DETECTION_MAX_BOXES
        );
        for (int i = 0; i < result->smoke_box_count; ++i) {
            const auto& src = cpp_result.smoke_boxes[i];
            result->smoke_boxes[i].x1 = src.x1;
            result->smoke_boxes[i].y1 = src.y1;
            result->smoke_boxes[i].x2 = src.x2;
            result->smoke_boxes[i].y2 = src.y2;
            result->smoke_boxes[i].confidence = src.confidence;
            result->smoke_boxes[i].class_id = static_cast<int>(src.class_id);
        }

        // 时序分析结果
        result->temporal_valid = cpp_result.temporal_valid ? 1 : 0;
        result->is_fire_confirmed = cpp_result.is_fire_confirmed ? 1 : 0;
        result->temporal_confidence = cpp_result.temporal_confidence;
        result->temporal_class = static_cast<int>(cpp_result.temporal_class);

        for (int i = 0; i < 3; ++i) {
            result->class_scores[i] = cpp_result.class_scores[i];
        }

        // 场景状态
        result->queue_size = cpp_result.queue_size;
        result->convlstm_buffer_size = cpp_result.convlstm_buffer_size;
        result->last_frame_timestamp = cpp_result.last_frame_timestamp;

        return 1;
    } catch (const std::exception& e) {
        std::cerr << "scene_controller_process_frame failed: " << e.what() << std::endl;
        return 0;
    }
}

int scene_controller_get_queue_size(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (!handle) {
        return -1;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->getSceneQueueSize(scene_id);
}

int scene_controller_is_scene_ready(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (!handle) {
        return 0;
    }
    SceneController* controller = static_cast<SceneController*>(handle);
    return controller->isSceneReadyForInference(scene_id) ? 1 : 0;
}

void scene_controller_reset_scene(
    SceneControllerHandle handle,
    int32_t scene_id
) {
    if (handle) {
        SceneController* controller = static_cast<SceneController*>(handle);
        controller->resetScene(scene_id);
    }
}

void scene_controller_reset_all(SceneControllerHandle handle) {
    if (handle) {
        SceneController* controller = static_cast<SceneController*>(handle);
        controller->resetAllScenes();
    }
}

}  // extern "C"
