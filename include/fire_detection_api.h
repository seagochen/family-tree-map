/**
 * @file fire_detection_api.h
 * @brief Fire Detection API - 火灾检测公共接口
 *
 * 集成 YOLOv8-seg 分割检测、ROI 提取和 ConvLSTM 时序分类的完整火灾检测 API。
 *
 * 特性：
 * - 单帧流式处理，内部自动管理滑动窗口
 * - 同时提供 C++ 和 C 接口
 * - PIMPL 模式隐藏实现细节，ABI 稳定
 *
 * @author Fire Detection Team
 * @version 1.0.0
 */

#ifndef FIRE_DETECTION_API_H
#define FIRE_DETECTION_API_H

#include <opencv2/opencv.hpp>
#include <memory>
#include <string>
#include <vector>

// 库导出宏
#ifdef _WIN32
  #ifdef FIRE_DETECTION_EXPORTS
    #define FIRE_API __declspec(dllexport)
  #else
    #define FIRE_API __declspec(dllimport)
  #endif
#else
  #define FIRE_API __attribute__((visibility("default")))
#endif

namespace fire_detection {

//=============================================================================
// 版本信息
//=============================================================================

#define FIRE_DETECTION_VERSION_MAJOR 1
#define FIRE_DETECTION_VERSION_MINOR 0
#define FIRE_DETECTION_VERSION_PATCH 0
#define FIRE_DETECTION_VERSION "1.0.0"

//=============================================================================
// 枚举和常量
//=============================================================================

/**
 * @brief 时序分类结果
 */
enum class TemporalClass {
    STATIC = 0,    ///< 静态（误报，如图片、红色物体、灯光等）
    DYNAMIC = 1,   ///< 动态（真实火灾，具有闪烁、扩散等特征）
    NEGATIVE = 2   ///< 无检测
};

/**
 * @brief 检测类别
 */
enum class DetectionClass {
    FIRE = 0,      ///< 火焰
    PERSON = 1,    ///< 人（负样本，忽略）
    SMOKE = 2      ///< 烟雾
};

//=============================================================================
// 配置结构
//=============================================================================

/**
 * @brief 检测器配置
 */
struct FIRE_API Config {
    std::string yolo_engine_path;       ///< YOLOv8-seg TensorRT 引擎路径（必需）
    std::string convlstm_engine_path;   ///< ConvLSTM TensorRT 引擎路径（必需）

    float confidence_threshold = 0.5f;  ///< 检测置信度阈值 [0, 1]
    float iou_threshold = 0.45f;        ///< NMS IoU 阈值 [0, 1]

    int convlstm_seq_len = 10;          ///< ConvLSTM 序列长度（滑动窗口大小）
    bool enable_roi_extraction = true;  ///< 启用 ROI 提取（非ROI区域置黑）

    float ema_alpha = 0.2f;             ///< EMA 平滑系数 [0, 1]
    float bbox_padding = 0.05f;         ///< BBox 填充比例 [0, 1]

    /**
     * @brief 验证配置是否有效
     * @return 是否有效
     */
    bool isValid() const {
        return !yolo_engine_path.empty() &&
               !convlstm_engine_path.empty() &&
               confidence_threshold > 0 && confidence_threshold <= 1 &&
               iou_threshold > 0 && iou_threshold <= 1 &&
               convlstm_seq_len > 0 &&
               ema_alpha > 0 && ema_alpha <= 1 &&
               bbox_padding >= 0 && bbox_padding <= 1;
    }
};

//=============================================================================
// 结果结构
//=============================================================================

/**
 * @brief 单个检测框
 */
struct FIRE_API BoundingBox {
    float x1, y1, x2, y2;       ///< 边界框坐标
    float confidence;           ///< 置信度
    DetectionClass class_id;    ///< 类别

    /// 获取宽度
    float width() const { return x2 - x1; }
    /// 获取高度
    float height() const { return y2 - y1; }
    /// 获取面积
    float area() const { return width() * height(); }
    /// 获取中心点 X
    float centerX() const { return (x1 + x2) / 2; }
    /// 获取中心点 Y
    float centerY() const { return (y1 + y2) / 2; }
};

/**
 * @brief 单帧检测结果
 */
struct FIRE_API FrameResult {
    bool has_fire = false;              ///< 是否检测到火焰
    bool has_smoke = false;             ///< 是否检测到烟雾
    std::vector<BoundingBox> detections; ///< 检测框列表
    cv::Mat roi_frame;                  ///< ROI 处理后的帧（非 ROI 区域置黑）

    /// 是否有任何检测
    bool hasDetection() const { return has_fire || has_smoke; }
};

/**
 * @brief 完整检测结果
 */
struct FIRE_API DetectionResult {
    FrameResult frame;                  ///< 单帧检测结果

    // 时序分类结果
    bool temporal_valid = false;        ///< 时序结果是否有效（缓冲区满时为 true）
    TemporalClass temporal_class = TemporalClass::NEGATIVE; ///< 时序分类
    float temporal_confidence = 0.0f;   ///< 时序置信度
    float class_scores[3] = {0, 0, 0};  ///< 各类别得分 [STATIC, DYNAMIC, NEGATIVE]

    /// 是否为真实火灾警报
    bool isFireAlert() const {
        return temporal_valid && temporal_class == TemporalClass::DYNAMIC;
    }
};

//=============================================================================
// 主类
//=============================================================================

/**
 * @brief 火灾检测器
 *
 * 集成 YOLOv8-seg 分割检测、ROI 提取和 ConvLSTM 时序分类。
 *
 * @par 使用示例 (C++)
 * @code
 * #include "fire_detection_api.h"
 *
 * fire_detection::Config config;
 * config.yolo_engine_path = "segment.engine";
 * config.convlstm_engine_path = "convlstm.engine";
 *
 * fire_detection::FireDetector detector;
 * if (!detector.initialize(config)) {
 *     std::cerr << "Failed to initialize" << std::endl;
 *     return 1;
 * }
 *
 * cv::VideoCapture cap("video.mp4");
 * cv::Mat frame;
 *
 * while (cap.read(frame)) {
 *     auto result = detector.processFrame(frame);
 *
 *     if (result.isFireAlert()) {
 *         std::cout << "FIRE ALERT! Confidence: "
 *                   << result.temporal_confidence * 100 << "%" << std::endl;
 *     }
 *
 *     cv::imshow("ROI", result.frame.roi_frame);
 *     if (cv::waitKey(1) == 'q') break;
 * }
 * @endcode
 */
class FIRE_API FireDetector {
public:
    FireDetector();
    ~FireDetector();

    // 禁用拷贝
    FireDetector(const FireDetector&) = delete;
    FireDetector& operator=(const FireDetector&) = delete;

    // 允许移动
    FireDetector(FireDetector&&) noexcept;
    FireDetector& operator=(FireDetector&&) noexcept;

    /**
     * @brief 初始化检测器
     * @param config 配置参数
     * @return 是否成功
     */
    bool initialize(const Config& config);

    /**
     * @brief 检查是否已初始化
     */
    bool isReady() const;

    /**
     * @brief 处理单帧图像
     *
     * 滑动窗口自动管理，每帧调用此函数即可。
     * 前 seq_len-1 帧的 temporal_valid 为 false。
     *
     * @param frame BGR 格式图像 (cv::Mat)
     * @return 检测结果
     */
    DetectionResult processFrame(const cv::Mat& frame);

    /**
     * @brief 处理原始图像数据
     *
     * @param data 图像数据指针 (BGR HWC 格式)
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 通道数 (默认 3)
     * @return 检测结果
     */
    DetectionResult processFrameRaw(
        const uint8_t* data,
        int width, int height,
        int channels = 3
    );

    /**
     * @brief 重置状态
     *
     * 清空滑动窗口缓冲区，适用于切换视频源时调用。
     */
    void reset();

    /**
     * @brief 获取当前缓冲区帧数
     */
    int getBufferSize() const;

    /**
     * @brief 获取缓冲区容量（seq_len）
     */
    int getBufferCapacity() const;

    /**
     * @brief 检查缓冲区是否已满
     */
    bool isBufferFull() const;

    /**
     * @brief 获取最后一次 ROI 处理后的帧
     *
     * 用于 C 接口获取 ROI 帧数据。
     *
     * @return ROI 帧引用
     */
    const cv::Mat& getLastRoiFrame() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

//=============================================================================
// 工具函数
//=============================================================================

/**
 * @brief 获取时序分类名称
 * @param cls 时序分类
 * @return 名称字符串
 */
FIRE_API const char* temporalClassToString(TemporalClass cls);

/**
 * @brief 获取时序分类颜色 (BGR)
 * @param cls 时序分类
 * @return OpenCV 颜色
 */
FIRE_API cv::Scalar temporalClassToColor(TemporalClass cls);

/**
 * @brief 获取检测类别名称
 * @param cls 检测类别
 * @return 名称字符串
 */
FIRE_API const char* detectionClassToString(DetectionClass cls);

/**
 * @brief 获取库版本
 * @return 版本字符串
 */
FIRE_API const char* getVersion();

}  // namespace fire_detection

//=============================================================================
// C 语言接口 (便于 FFI 绑定)
//=============================================================================

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 单帧最大检测数量
 *
 * 设计考量：
 * - 火灾场景中通常不会有超过 32 个独立的火焰/烟雾区域
 * - 32 个 BoundingBoxC 占用约 768 字节，内存开销可接受
 * - 如果实际检测数超过此值，将保留置信度最高的前 N 个
 */
#define FIRE_DETECTION_MAX_BOXES 32

/**
 * @brief C 风格的检测框结构
 */
typedef struct {
    float x1;           ///< 左上角 X 坐标
    float y1;           ///< 左上角 Y 坐标
    float x2;           ///< 右下角 X 坐标
    float y2;           ///< 右下角 Y 坐标
    float confidence;   ///< 置信度 [0, 1]
    int class_id;       ///< 类别 ID (0=FIRE, 1=PERSON, 2=SMOKE)
} BoundingBoxC;

/**
 * @brief 不透明句柄类型
 */
typedef void* FireDetectorHandle;

/**
 * @brief C 风格配置结构
 */
typedef struct {
    const char* yolo_engine_path;       ///< YOLOv8-seg 引擎路径
    const char* convlstm_engine_path;   ///< ConvLSTM 引擎路径
    float confidence_threshold;         ///< 置信度阈值
    float iou_threshold;                ///< IoU 阈值
    int convlstm_seq_len;               ///< 序列长度
    int enable_roi_extraction;          ///< 启用 ROI 提取 (0/1)
    float ema_alpha;                    ///< EMA 系数
    float bbox_padding;                 ///< BBox 填充
} FireDetectorConfig;

/**
 * @brief C 风格检测结果
 *
 * 包含单帧检测结果和时序分类结果。
 * 新增 detection_count 和 detections 字段用于返回检测框坐标。
 */
typedef struct {
    // ========== 原有字段（保持向后兼容）==========
    int has_fire;                       ///< 是否有火焰 (0/1)
    int has_smoke;                      ///< 是否有烟雾 (0/1)
    int temporal_valid;                 ///< 时序结果有效 (0/1)
    int temporal_class;                 ///< 时序分类 (0=STATIC, 1=DYNAMIC, 2=NEGATIVE)
    float temporal_confidence;          ///< 时序置信度
    float class_scores[3];              ///< 各类别得分

    // ========== 新增字段：检测框信息 ==========
    int detection_count;                                ///< 实际检测框数量 [0, FIRE_DETECTION_MAX_BOXES]
    BoundingBoxC detections[FIRE_DETECTION_MAX_BOXES];  ///< 检测框数组
} FireDetectionResultC;

/**
 * @brief 创建检测器
 * @return 检测器句柄
 */
FIRE_API FireDetectorHandle fire_detector_create(void);

/**
 * @brief 销毁检测器
 * @param handle 检测器句柄
 */
FIRE_API void fire_detector_destroy(FireDetectorHandle handle);

/**
 * @brief 初始化检测器
 * @param handle 检测器句柄
 * @param config 配置指针
 * @return 是否成功 (0/1)
 */
FIRE_API int fire_detector_initialize(
    FireDetectorHandle handle,
    const FireDetectorConfig* config
);

/**
 * @brief 处理帧
 * @param handle 检测器句柄
 * @param data 图像数据 (BGR HWC)
 * @param width 宽度
 * @param height 高度
 * @param channels 通道数
 * @param result 输出结果指针
 * @return 是否成功 (0/1)
 */
FIRE_API int fire_detector_process_frame(
    FireDetectorHandle handle,
    const unsigned char* data,
    int width, int height, int channels,
    FireDetectionResultC* result
);

/**
 * @brief 获取 ROI 帧数据
 * @param handle 检测器句柄
 * @param output_data 输出缓冲区（需预分配 width*height*3）
 * @param width 输出宽度
 * @param height 输出高度
 * @return 是否成功 (0/1)
 */
FIRE_API int fire_detector_get_roi_frame(
    FireDetectorHandle handle,
    unsigned char* output_data,
    int* width, int* height
);

/**
 * @brief 重置状态
 * @param handle 检测器句柄
 */
FIRE_API void fire_detector_reset(FireDetectorHandle handle);

/**
 * @brief 检查是否就绪
 * @param handle 检测器句柄
 * @return 是否就绪 (0/1)
 */
FIRE_API int fire_detector_is_ready(FireDetectorHandle handle);

/**
 * @brief 获取缓冲区大小
 * @param handle 检测器句柄
 * @return 缓冲区帧数
 */
FIRE_API int fire_detector_get_buffer_size(FireDetectorHandle handle);

/**
 * @brief 检查缓冲区是否已满
 * @param handle 检测器句柄
 * @return 是否已满 (0/1)
 */
FIRE_API int fire_detector_is_buffer_full(FireDetectorHandle handle);

/**
 * @brief 获取版本号
 * @return 版本字符串
 */
FIRE_API const char* fire_detector_get_version(void);

#ifdef __cplusplus
}
#endif

#endif // FIRE_DETECTION_API_H
