/**
 * @file scene_controller.h
 * @brief 场景控制器 - 管理球形摄像机的多场景切换
 *
 * 设计用于处理球形摄像机周期性旋转切换不同场景的情况。
 * 每个场景维护独立的帧队列和时序分析状态。
 *
 * 主要功能：
 * - 场景注册/删除
 * - 独立的帧队列管理
 * - 时间间隔检查（清除过期帧）
 * - 乱序处理和采样
 *
 * @author Fire Detection Team
 * @version 1.0.0
 */

#ifndef SCENE_CONTROLLER_H
#define SCENE_CONTROLLER_H

#include "fire_detection_api.h"
#include <cstdint>
#include <memory>
#include <vector>

namespace fire_detection {

//=============================================================================
// 场景配置
//=============================================================================

/**
 * @brief 场景控制器配置
 */
struct FIRE_API SceneConfig {
    /// 基础检测器配置
    Config detector_config;

    /// 帧间最大时间间隔（毫秒），超过则清空队列，默认30秒
    int64_t max_frame_gap_ms = 30000;

    /// 最小采样间隔（毫秒），用于降采样，默认100ms
    int64_t min_sample_interval_ms = 100;

    /// 每个场景最大队列帧数
    int max_queue_size = 30;

    /**
     * @brief 验证配置是否有效
     * @return 是否有效
     */
    bool isValid() const {
        return detector_config.isValid() &&
               max_frame_gap_ms > 0 &&
               min_sample_interval_ms > 0 &&
               max_queue_size > 0;
    }
};

//=============================================================================
// 带时间戳的帧
//=============================================================================

/**
 * @brief 带时间戳的帧数据
 */
struct FIRE_API TimestampedFrame {
    cv::Mat frame;          ///< BGR格式图像
    int64_t timestamp_ms;   ///< 采样时间戳（毫秒）

    TimestampedFrame() : timestamp_ms(0) {}
    TimestampedFrame(const cv::Mat& f, int64_t ts)
        : frame(f.clone()), timestamp_ms(ts) {}
};

//=============================================================================
// 场景检测结果
//=============================================================================

/**
 * @brief 单个场景的检测结果
 */
struct FIRE_API SceneDetectionResult {
    int32_t scene_id = 0;               ///< 场景ID

    // ========== 空间检测结果（单帧）==========
    bool has_fire = false;              ///< 是否检测到火焰
    bool has_smoke = false;             ///< 是否检测到烟雾
    std::vector<BoundingBox> fire_boxes;    ///< 火焰检测框列表
    std::vector<BoundingBox> smoke_boxes;   ///< 烟雾检测框列表

    // ========== 时序分析结果 ==========
    bool temporal_valid = false;        ///< 时序结果是否有效
    bool is_fire_confirmed = false;     ///< 是否确认火灾（temporal_valid && DYNAMIC）
    float temporal_confidence = 0.0f;   ///< 时序置信度
    TemporalClass temporal_class = TemporalClass::NEGATIVE; ///< 时序分类
    float class_scores[3] = {0, 0, 0};  ///< 各类别得分 [STATIC, DYNAMIC, NEGATIVE]

    // ========== 场景状态 ==========
    int queue_size = 0;                 ///< 当前队列大小
    int convlstm_buffer_size = 0;       ///< ConvLSTM缓冲区大小
    int64_t last_frame_timestamp = 0;   ///< 最后一帧时间戳

    /// 便捷查询：是否发出火灾警报
    bool isFireAlert() const {
        return temporal_valid && temporal_class == TemporalClass::DYNAMIC;
    }
};

//=============================================================================
// SceneController 主类
//=============================================================================

/**
 * @brief 场景控制器 - 管理球形摄像机的多场景切换
 *
 * @par 使用示例
 * @code
 * fire_detection::SceneConfig config;
 * config.detector_config.yolo_engine_path = "segment.engine";
 * config.detector_config.convlstm_engine_path = "convlstm.engine";
 * config.max_frame_gap_ms = 30000;  // 30秒
 *
 * fire_detection::SceneController controller;
 * if (!controller.initialize(config)) {
 *     return 1;
 * }
 *
 * // 注册场景
 * controller.registerScene(1);  // 场景1
 * controller.registerScene(2);  // 场景2
 *
 * // 处理帧（场景1）
 * int64_t timestamp = getCurrentTimeMs();
 * auto result = controller.processFrame(1, frame, timestamp);
 *
 * if (result.isFireAlert()) {
 *     std::cout << "Scene " << result.scene_id << " FIRE ALERT!" << std::endl;
 * }
 * @endcode
 */
class FIRE_API SceneController {
public:
    SceneController();
    ~SceneController();

    // 禁用拷贝
    SceneController(const SceneController&) = delete;
    SceneController& operator=(const SceneController&) = delete;

    // 允许移动
    SceneController(SceneController&&) noexcept;
    SceneController& operator=(SceneController&&) noexcept;

    //=========================================================================
    // 初始化
    //=========================================================================

    /**
     * @brief 初始化场景控制器
     * @param config 场景配置
     * @return 是否成功
     */
    bool initialize(const SceneConfig& config);

    /**
     * @brief 检查是否已初始化
     */
    bool isReady() const;

    //=========================================================================
    // 场景管理
    //=========================================================================

    /**
     * @brief 注册新场景
     * @param scene_id 场景ID（INT32）
     * @return 是否成功（场景已存在返回false）
     */
    bool registerScene(int32_t scene_id);

    /**
     * @brief 删除场景
     * @param scene_id 场景ID
     * @return 是否成功（场景不存在返回false）
     */
    bool unregisterScene(int32_t scene_id);

    /**
     * @brief 检查场景是否存在
     * @param scene_id 场景ID
     */
    bool hasScene(int32_t scene_id) const;

    /**
     * @brief 获取所有已注册的场景ID
     */
    std::vector<int32_t> getSceneIds() const;

    /**
     * @brief 获取场景数量
     */
    size_t getSceneCount() const;

    //=========================================================================
    // 帧处理
    //=========================================================================

    /**
     * @brief 处理单帧图像
     *
     * 将帧添加到指定场景的队列，执行时间检查、排序、采样，
     * 然后进行检测和时序分析。
     *
     * @param scene_id 场景ID
     * @param frame BGR格式图像
     * @param timestamp_ms 采样时间戳（毫秒，由调用者传入）
     * @return 场景检测结果
     *
     * @note 如果场景不存在，会自动注册
     */
    SceneDetectionResult processFrame(
        int32_t scene_id,
        const cv::Mat& frame,
        int64_t timestamp_ms
    );

    /**
     * @brief 处理原始图像数据
     */
    SceneDetectionResult processFrameRaw(
        int32_t scene_id,
        const uint8_t* data,
        int width, int height, int channels,
        int64_t timestamp_ms
    );

    //=========================================================================
    // 场景状态
    //=========================================================================

    /**
     * @brief 获取场景队列大小
     * @param scene_id 场景ID
     * @return 队列大小，场景不存在返回-1
     */
    int getSceneQueueSize(int32_t scene_id) const;

    /**
     * @brief 获取场景ConvLSTM缓冲区大小
     * @param scene_id 场景ID
     * @return 缓冲区大小，场景不存在返回-1
     */
    int getSceneBufferSize(int32_t scene_id) const;

    /**
     * @brief 检查场景是否可以进行时序推理
     * @param scene_id 场景ID
     */
    bool isSceneReadyForInference(int32_t scene_id) const;

    /**
     * @brief 重置指定场景状态
     * @param scene_id 场景ID
     */
    void resetScene(int32_t scene_id);

    /**
     * @brief 重置所有场景状态
     */
    void resetAllScenes();

    /**
     * @brief 获取最后处理的ROI帧
     * @param scene_id 场景ID
     * @return ROI帧，场景不存在返回空Mat
     */
    cv::Mat getLastRoiFrame(int32_t scene_id) const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace fire_detection

//=============================================================================
// C 语言接口 (便于 FFI 绑定)
//=============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 场景控制器不透明句柄
 */
typedef void* SceneControllerHandle;

/**
 * @brief C风格场景配置
 */
typedef struct {
    FireDetectorConfig detector_config; ///< 基础检测器配置
    int64_t max_frame_gap_ms;           ///< 帧间最大时间间隔
    int64_t min_sample_interval_ms;     ///< 最小采样间隔
    int max_queue_size;                 ///< 每个场景最大队列帧数
} SceneControllerConfig;

/**
 * @brief C风格场景检测结果
 */
typedef struct {
    int32_t scene_id;                   ///< 场景ID

    // 空间检测结果
    int has_fire;                       ///< 是否检测到火焰 (0/1)
    int has_smoke;                      ///< 是否检测到烟雾 (0/1)
    int fire_box_count;                 ///< 火焰检测框数量
    BoundingBoxC fire_boxes[FIRE_DETECTION_MAX_BOXES];  ///< 火焰检测框
    int smoke_box_count;                ///< 烟雾检测框数量
    BoundingBoxC smoke_boxes[FIRE_DETECTION_MAX_BOXES]; ///< 烟雾检测框

    // 时序分析结果
    int temporal_valid;                 ///< 时序结果有效 (0/1)
    int is_fire_confirmed;              ///< 是否确认火灾 (0/1)
    float temporal_confidence;          ///< 时序置信度
    int temporal_class;                 ///< 时序分类 (0=STATIC, 1=DYNAMIC, 2=NEGATIVE)
    float class_scores[3];              ///< 各类别得分

    // 场景状态
    int queue_size;                     ///< 当前队列大小
    int convlstm_buffer_size;           ///< ConvLSTM缓冲区大小
    int64_t last_frame_timestamp;       ///< 最后一帧时间戳
} SceneDetectionResultC;

/**
 * @brief 创建场景控制器
 * @return 场景控制器句柄
 */
FIRE_API SceneControllerHandle scene_controller_create(void);

/**
 * @brief 销毁场景控制器
 * @param handle 场景控制器句柄
 */
FIRE_API void scene_controller_destroy(SceneControllerHandle handle);

/**
 * @brief 初始化场景控制器
 * @param handle 场景控制器句柄
 * @param config 配置指针
 * @return 是否成功 (0/1)
 */
FIRE_API int scene_controller_initialize(
    SceneControllerHandle handle,
    const SceneControllerConfig* config
);

/**
 * @brief 检查是否就绪
 * @param handle 场景控制器句柄
 * @return 是否就绪 (0/1)
 */
FIRE_API int scene_controller_is_ready(SceneControllerHandle handle);

/**
 * @brief 注册场景
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @return 是否成功 (0/1)
 */
FIRE_API int scene_controller_register_scene(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 删除场景
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @return 是否成功 (0/1)
 */
FIRE_API int scene_controller_unregister_scene(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 检查场景是否存在
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @return 是否存在 (0/1)
 */
FIRE_API int scene_controller_has_scene(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 获取场景数量
 * @param handle 场景控制器句柄
 * @return 场景数量
 */
FIRE_API int scene_controller_get_scene_count(SceneControllerHandle handle);

/**
 * @brief 处理帧
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @param data 图像数据 (BGR HWC)
 * @param width 宽度
 * @param height 高度
 * @param channels 通道数
 * @param timestamp_ms 时间戳（毫秒）
 * @param result 输出结果指针
 * @return 是否成功 (0/1)
 */
FIRE_API int scene_controller_process_frame(
    SceneControllerHandle handle,
    int32_t scene_id,
    const unsigned char* data,
    int width, int height, int channels,
    int64_t timestamp_ms,
    SceneDetectionResultC* result
);

/**
 * @brief 获取场景队列大小
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @return 队列大小，场景不存在返回-1
 */
FIRE_API int scene_controller_get_queue_size(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 检查场景是否可以进行时序推理
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 * @return 是否就绪 (0/1)
 */
FIRE_API int scene_controller_is_scene_ready(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 重置指定场景
 * @param handle 场景控制器句柄
 * @param scene_id 场景ID
 */
FIRE_API void scene_controller_reset_scene(
    SceneControllerHandle handle,
    int32_t scene_id
);

/**
 * @brief 重置所有场景
 * @param handle 场景控制器句柄
 */
FIRE_API void scene_controller_reset_all(SceneControllerHandle handle);

#ifdef __cplusplus
}
#endif

#endif // SCENE_CONTROLLER_H
