/**
 * @file fire_detection_api.cpp
 * @brief 火焰检测 API C++ 实现
 *
 * 本文件是火焰检测库的核心 API 实现，提供完整的火焰/烟雾检测功能。
 *
 * 【模块架构概述】
 * 本模块采用 PIMPL (Pointer to Implementation) 设计模式，将接口与实现分离：
 * - FireDetector: 公开的 API 类，提供稳定的二进制接口
 * - FireDetector::Impl: 内部实现类，封装所有检测逻辑
 *
 * 【检测流水线架构】
 * 整个检测系统由三个主要组件串联组成：
 *
 * 1. YOLOv8-seg 分割检测器 (SegmentDetector)
 *    - 功能：实时目标检测和实例分割
 *    - 输入：原始视频帧
 *    - 输出：火焰/烟雾的边界框和分割掩膜
 *
 * 2. ROI 提取流水线 (ROIPipeline)
 *    - 功能：从检测结果中提取感兴趣区域
 *    - 特性：支持 EMA 平滑、边界框填充
 *    - 输出：裁剪后的 ROI 图像
 *
 * 3. ConvLSTM 时序分类器 (ConvLSTMClassifier)
 *    - 功能：分析连续帧序列，判断火焰真实性
 *    - 目的：区分真实火焰(DYNAMIC)、静态火焰图片(STATIC)、误检(NEGATIVE)
 *    - 原理：真实火焰具有动态闪烁特征，静态图片无时序变化
 *
 * 【数据流向】
 * 输入帧 -> YOLO检测 -> ROI提取 -> ConvLSTM时序分析 -> 最终结果
 *
 * 【使用场景】
 * - 视频监控中的火灾预警
 * - 工业场所的火焰检测
 * - 森林火灾早期发现
 * - 智能安防系统集成
 */

#include "fire_detection_api.h"
#include "detector/segment_detector.h"
#include "roi/roi_extractor.h"
#include "temporal/convlstm_classifier.h"

#include <trt_engine/trt_engine.h>
#include <iostream>

namespace fire_detection {

//=============================================================================
// FireDetector::Impl - PIMPL 实现类
//=============================================================================

/**
 * @class FireDetector::Impl
 * @brief 火焰检测器的内部实现类
 *
 * 【PIMPL 设计模式说明】
 * PIMPL (Pointer to Implementation) 是一种编译防火墙技术：
 * - 将实现细节隐藏在 cpp 文件中，头文件只声明前向指针
 * - 修改实现不需要重新编译依赖该头文件的代码
 * - 保持 ABI (应用二进制接口) 稳定性
 * - 隐藏第三方库依赖，简化用户的编译配置
 *
 * 【成员组件说明】
 * - yolo_engine_: YOLOv8-seg 的 TensorRT 推理引擎
 * - convlstm_engine_: ConvLSTM 的 TensorRT 推理引擎
 * - segment_detector_: 分割检测器，封装 YOLO 的前处理和后处理
 * - roi_pipeline_: ROI 提取流水线，处理检测结果并提取感兴趣区域
 * - convlstm_classifier_: 时序分类器，分析帧序列判断火焰真实性
 */
class FireDetector::Impl {
public:
    Impl() = default;
    ~Impl() = default;

    /**
     * @brief 初始化检测器 - 加载模型并配置各组件
     *
     * @param config 配置参数，包含模型路径和检测阈值
     * @return 初始化是否成功
     *
     * 【初始化流程详解】
     *
     * 整个初始化过程按依赖顺序执行：
     *
     * 1. 配置验证
     *    - 检查是否重复初始化（防止资源泄漏）
     *    - 验证配置参数有效性（路径存在、阈值范围合理）
     *
     * 2. 加载 YOLOv8-seg 引擎
     *    - 从 .engine 文件加载序列化的 TensorRT 模型
     *    - TensorRT 引擎针对特定 GPU 优化，首次加载可能需要重新构建
     *
     * 3. 初始化分割检测器
     *    - 创建 SegmentDetector 实例
     *    - 配置置信度阈值和 NMS IoU 阈值
     *    - 分配 GPU 内存用于推理
     *
     * 4. 创建 ROI 流水线
     *    - 使用 lambda 回调封装检测器调用
     *    - 配置 EMA 平滑参数（减少掩膜抖动）
     *    - 设置边界框填充比例（确保完整捕获目标）
     *
     * 5. 加载 ConvLSTM 引擎
     *    - 加载时序分析模型
     *    - ConvLSTM 用于分析连续帧的时间特征
     *
     * 6. 初始化 ConvLSTM 分类器
     *    - 配置序列长度（通常 8-16 帧）
     *    - 分配帧缓冲区
     *
     * 【错误处理策略】
     * - 任何步骤失败都会立即返回 false
     * - 已分配的资源通过 unique_ptr 自动释放
     * - 错误信息输出到 stderr 便于调试
     */
    bool initialize(const Config& config) {
        // ========== 步骤1: 配置验证 ==========
        // 防止重复初始化，避免内存泄漏和状态混乱
        if (initialized_) {
            std::cerr << "FireDetector: Already initialized" << std::endl;
            return false;
        }

        // 验证配置参数有效性
        // isValid() 检查: 模型路径非空、阈值在合理范围内等
        if (!config.isValid()) {
            std::cerr << "FireDetector: Invalid configuration" << std::endl;
            return false;
        }

        config_ = config;

        // ========== 步骤2: 加载 YOLOv8-seg TensorRT 引擎 ==========
        // TensorRT 引擎文件 (.engine) 是针对特定 GPU 预编译的推理模型
        // 加载过程: 反序列化引擎 -> 创建执行上下文 -> 分配 GPU 内存
        std::cout << "FireDetector: Loading YOLOv8-seg engine..." << std::endl;
        yolo_engine_ = std::make_unique<TrtEngineMultiTs>();
        if (!yolo_engine_->loadFromFile(config_.yolo_engine_path)) {
            std::cerr << "FireDetector: Failed to load YOLO engine: "
                      << config_.yolo_engine_path << std::endl;
            return false;
        }

        // ========== 步骤3: 初始化分割检测器 ==========
        // SegmentDetector 封装了 YOLOv8-seg 的完整推理流程：
        // - 图像预处理（letterbox 缩放、归一化）
        // - GPU 推理
        // - 后处理（NMS、掩膜生成）
        segment_detector_ = std::make_unique<internal::SegmentDetector>(
            config_.confidence_threshold,  // 置信度阈值，过滤低置信度检测
            config_.iou_threshold          // NMS IoU 阈值，过滤重叠框
        );
        if (!segment_detector_->initialize(*yolo_engine_)) {
            std::cerr << "FireDetector: Failed to initialize segment detector" << std::endl;
            return false;
        }

        // ========== 步骤4: 创建 ROI 提取流水线 ==========
        // ROI 流水线将检测结果转换为适合时序分析的输入
        //
        // 使用 lambda 回调封装检测器调用：
        // - 这种设计使 ROIPipeline 与具体检测器解耦
        // - 便于替换不同的检测后端
        auto detector_callback = [this](const cv::Mat& frame, float conf) {
            return segment_detector_->detect(frame);
        };

        // ROIPipeline 参数说明:
        // - detector_callback: 检测器回调函数
        // - confidence_threshold: 置信度过滤阈值
        // - use_ema: 启用 EMA 平滑（减少掩膜帧间抖动）
        // - ema_alpha: EMA 平滑系数（0.1-0.3 较平滑，0.5-0.8 响应快）
        // - bbox_padding: 边界框填充比例（0.1 表示各方向扩展 10%）
        roi_pipeline_ = std::make_unique<roi_extractor::ROIPipeline>(
            detector_callback,
            config_.confidence_threshold,
            true,  // 使用 EMA 平滑，减少检测框的帧间跳动
            config_.ema_alpha,
            config_.bbox_padding
        );

        // ========== 步骤5: 加载 ConvLSTM TensorRT 引擎 ==========
        // ConvLSTM (Convolutional LSTM) 是一种时序模型：
        // - 结合 CNN 的空间特征提取能力
        // - 和 LSTM 的时序建模能力
        // - 适合分析视频帧序列中的动态特征
        std::cout << "FireDetector: Loading ConvLSTM engine..." << std::endl;
        convlstm_engine_ = std::make_unique<TrtEngineMultiTs>();
        if (!convlstm_engine_->loadFromFile(config_.convlstm_engine_path)) {
            std::cerr << "FireDetector: Failed to load ConvLSTM engine: "
                      << config_.convlstm_engine_path << std::endl;
            return false;
        }

        // ========== 步骤6: 初始化 ConvLSTM 时序分类器 ==========
        // ConvLSTMClassifier 维护一个滑动窗口帧缓冲区：
        // - 收集连续的 ROI 帧
        // - 缓冲区满时执行时序推理
        // - 输出三分类结果: STATIC/DYNAMIC/NEGATIVE
        convlstm_classifier_ = std::make_unique<convlstm::ConvLSTMClassifier>(
            *convlstm_engine_,
            config_.convlstm_seq_len  // 序列长度，通常 8-16 帧
        );
        if (!convlstm_classifier_->initialize()) {
            std::cerr << "FireDetector: Failed to initialize ConvLSTM classifier" << std::endl;
            return false;
        }

        // ========== 初始化完成 ==========
        initialized_ = true;
        std::cout << "FireDetector: Initialized successfully" << std::endl;
        std::cout << "  YOLO engine: " << config_.yolo_engine_path << std::endl;
        std::cout << "  ConvLSTM engine: " << config_.convlstm_engine_path << std::endl;
        std::cout << "  Sequence length: " << config_.convlstm_seq_len << std::endl;

        return true;
    }

    /**
     * @brief 检查检测器是否已就绪
     * @return true 表示已初始化且可以处理帧
     */
    bool isReady() const {
        return initialized_;
    }

    /**
     * @brief 处理单帧图像 - 核心检测流水线
     *
     * @param frame 输入的 BGR 格式图像
     * @return 检测结果，包含空间检测和时序分类结果
     *
     * 【处理流程详解】
     *
     * 1. 输入验证
     *    - 检查初始化状态
     *    - 验证输入帧非空
     *
     * 2. ROI 流水线处理
     *    - 调用 YOLO 检测器获取火焰/烟雾检测结果
     *    - 根据检测掩膜提取 ROI 区域
     *    - 应用 EMA 平滑减少抖动
     *
     * 3. 结果转换
     *    - 将内部 Detection 格式转换为 API 的 BoundingBox 格式
     *    - 保存 ROI 帧用于后续访问（如可视化）
     *
     * 4. ConvLSTM 时序分析
     *    - 将处理后的帧添加到滑动窗口缓冲区
     *    - 缓冲区满时执行时序推理
     *    - 输出三分类结果及各类别置信度
     *
     * 【时序分类的意义】
     * - DYNAMIC: 真实火焰，具有动态闪烁特征
     * - STATIC: 静态火焰图片（如海报、屏幕显示的火焰图）
     * - NEGATIVE: 误检，非火焰目标
     *
     * 通过时序分析可以有效区分真实火灾和火焰图片/视频，
     * 大幅降低误报率，提高系统实用性。
     */
    DetectionResult processFrame(const cv::Mat& frame) {
        DetectionResult result;

        // ========== 步骤1: 输入验证 ==========
        if (!initialized_) {
            std::cerr << "FireDetector: Not initialized" << std::endl;
            return result;
        }

        if (frame.empty()) {
            std::cerr << "FireDetector: Empty frame" << std::endl;
            return result;
        }

        // ========== 步骤2: ROI 流水线处理 ==========
        // processBatch 支持批量处理，这里只处理单帧
        // 参数说明:
        // - frames: 输入帧列表
        // - DetectMode::ALL: 检测所有目标类别（火焰和烟雾）
        // - skip_empty=false: 不跳过无检测结果的帧（保持帧序列连续性）
        std::vector<cv::Mat> frames = {frame};
        roi_extractor::BatchResult batch_result = roi_pipeline_->processBatch(
            frames,
            roi_extractor::DetectMode::ALL,
            false  // 不跳过无检测的帧，确保时序连续性
        );

        // ========== 步骤3: 处理检测结果 ==========
        cv::Mat processed_frame;
        if (!batch_result.frames.empty()) {
            const auto& processed = batch_result.frames[0];

            // 提取检测标志
            result.frame.has_fire = processed.has_fire;
            result.frame.has_smoke = processed.has_smoke;

            // 根据配置决定使用 ROI 帧还是原始帧
            // - 启用 ROI 提取：使用裁剪后的目标区域，聚焦于火焰/烟雾
            // - 禁用 ROI 提取：使用完整原始帧
            if (config_.enable_roi_extraction) {
                processed_frame = processed.data;
                result.frame.roi_frame = processed.data.clone();
            } else {
                processed_frame = frame;
                result.frame.roi_frame = frame.clone();
            }

            // 转换检测框格式
            // 从内部 ROIDetection 格式转换为 API 的 BoundingBox 格式
            // BoundingBox 使用 (x1, y1, x2, y2) 表示左上角和右下角坐标
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
            // 无检测结果时，使用原始帧
            processed_frame = frame;
            result.frame.roi_frame = frame.clone();
        }

        // 保存最后的 ROI 帧，供外部访问（如 C 接口或可视化）
        last_roi_frame_ = result.frame.roi_frame;

        // ========== 步骤4: ConvLSTM 时序分析 ==========
        // 将处理后的帧添加到 ConvLSTM 的滑动窗口缓冲区
        // ConvLSTM 需要连续多帧才能分析时序特征
        convlstm_classifier_->addFrame(processed_frame);

        // 检查缓冲区是否已满（收集够足够的帧）
        // 只有缓冲区满时才执行时序推理
        if (convlstm_classifier_->isBufferFull()) {
            // 执行 ConvLSTM 推理
            if (convlstm_classifier_->infer()) {
                result.temporal_valid = true;  // 标记时序结果有效

                // 获取分类结果
                // 三分类: STATIC(静态图片), DYNAMIC(真实火焰), NEGATIVE(误检)
                auto cls = convlstm_classifier_->getClassification();
                result.temporal_class = static_cast<TemporalClass>(static_cast<int>(cls));
                result.temporal_confidence = convlstm_classifier_->getConfidence();

                // 获取各类别的置信度分数
                // class_scores[0]: STATIC 概率
                // class_scores[1]: DYNAMIC 概率
                // class_scores[2]: NEGATIVE 概率
                const auto& inference_result = convlstm_classifier_->getResult();
                for (int i = 0; i < 3; ++i) {
                    result.class_scores[i] = inference_result.class_scores[i];
                }
            }
        }

        return result;
    }

    /**
     * @brief 处理原始像素数据 - 提供更底层的接口
     *
     * @param data 像素数据指针，BGR 格式连续存储
     * @param width 图像宽度
     * @param height 图像高度
     * @param channels 通道数，必须为 3
     * @return 检测结果
     *
     * 【使用场景】
     * - 与其他语言/框架集成（如 Python、C#）
     * - 处理来自硬件采集卡的原始数据
     * - 避免 OpenCV Mat 对象的跨库传递
     *
     * 【数据布局要求】
     * - 像素按行优先顺序存储
     * - 每像素 3 字节: B, G, R
     * - 总大小: width × height × 3 字节
     */
    DetectionResult processFrameRaw(const uint8_t* data, int width, int height, int channels) {
        // 当前仅支持 3 通道 BGR 图像
        if (channels != 3) {
            std::cerr << "FireDetector: Only 3-channel images are supported" << std::endl;
            return DetectionResult();
        }

        // 使用外部数据创建 cv::Mat（不复制数据）
        // 注意：调用方必须确保 data 在处理完成前有效
        cv::Mat frame(height, width, CV_8UC3, const_cast<uint8_t*>(data));
        return processFrame(frame);
    }

    /**
     * @brief 重置检测器状态
     *
     * 【使用场景】
     * - 切换视频源时清空历史状态
     * - 检测场景变化时重新开始时序分析
     * - 测试时重置到初始状态
     *
     * 【重置内容】
     * - 清空 ConvLSTM 帧缓冲区
     * - 清空最后保存的 ROI 帧
     * - 不影响模型和配置
     */
    void reset() {
        if (convlstm_classifier_) {
            convlstm_classifier_->reset();
        }
        last_roi_frame_ = cv::Mat();
    }

    /**
     * @brief 获取当前帧缓冲区大小
     * @return 缓冲区中已有的帧数
     *
     * 【业务说明】
     * 用于监控时序分析的进度，判断还需要多少帧才能开始推理
     */
    int getBufferSize() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->getBufferSize();
        }
        return 0;
    }

    /**
     * @brief 获取帧缓冲区容量
     * @return 缓冲区总容量（序列长度）
     *
     * 【业务说明】
     * 返回 ConvLSTM 模型需要的输入序列长度
     * 通常为 8-16 帧，对应约 0.3-0.6 秒的视频（按 25fps 计算）
     */
    int getBufferCapacity() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->getSeqLen();
        }
        return 0;
    }

    /**
     * @brief 检查帧缓冲区是否已满
     * @return true 表示可以执行时序推理
     *
     * 【业务说明】
     * 缓冲区满时，processFrame 会自动执行时序推理
     * 调用方可用此函数提前判断下一帧是否会产生时序结果
     */
    bool isBufferFull() const {
        if (convlstm_classifier_) {
            return convlstm_classifier_->isBufferFull();
        }
        return false;
    }

    /**
     * @brief 获取最后处理的 ROI 帧
     * @return ROI 帧的常量引用
     *
     * 【使用场景】
     * - 用于可视化显示 ROI 区域
     * - C 接口需要访问 ROI 帧时使用
     * - 调试和日志记录
     */
    const cv::Mat& getLastRoiFrame() const {
        return last_roi_frame_;
    }

private:
    bool initialized_ = false;  ///< 初始化状态标志
    Config config_;             ///< 配置参数副本

    // ========== TensorRT 推理引擎 ==========
    std::unique_ptr<TrtEngineMultiTs> yolo_engine_;      ///< YOLOv8-seg 引擎
    std::unique_ptr<TrtEngineMultiTs> convlstm_engine_;  ///< ConvLSTM 引擎

    // ========== 检测组件 ==========
    std::unique_ptr<internal::SegmentDetector> segment_detector_;     ///< 分割检测器
    std::unique_ptr<roi_extractor::ROIPipeline> roi_pipeline_;        ///< ROI 提取流水线
    std::unique_ptr<convlstm::ConvLSTMClassifier> convlstm_classifier_; ///< 时序分类器

    // ========== 缓存数据 ==========
    cv::Mat last_roi_frame_;  ///< 最后处理的 ROI 帧，供外部访问
};

//=============================================================================
// FireDetector 公开接口实现
//=============================================================================

/**
 * @brief 构造函数 - 创建 PIMPL 实现对象
 *
 * 【设计说明】
 * 构造函数仅创建 Impl 对象，不执行任何初始化
 * 实际初始化在 initialize() 方法中进行，允许延迟加载模型
 */
FireDetector::FireDetector()
    : impl_(std::make_unique<Impl>())
{
}

/// 析构函数使用默认实现，unique_ptr 自动释放 Impl
FireDetector::~FireDetector() = default;

/// 移动构造函数
FireDetector::FireDetector(FireDetector&&) noexcept = default;

/// 移动赋值运算符
FireDetector& FireDetector::operator=(FireDetector&&) noexcept = default;

/// 委托给 Impl::initialize
bool FireDetector::initialize(const Config& config) {
    return impl_->initialize(config);
}

/// 委托给 Impl::isReady
bool FireDetector::isReady() const {
    return impl_->isReady();
}

/// 委托给 Impl::processFrame
DetectionResult FireDetector::processFrame(const cv::Mat& frame) {
    return impl_->processFrame(frame);
}

/// 委托给 Impl::processFrameRaw
DetectionResult FireDetector::processFrameRaw(
    const uint8_t* data,
    int width, int height,
    int channels
) {
    return impl_->processFrameRaw(data, width, height, channels);
}

/// 委托给 Impl::reset
void FireDetector::reset() {
    impl_->reset();
}

/// 委托给 Impl::getBufferSize
int FireDetector::getBufferSize() const {
    return impl_->getBufferSize();
}

/// 委托给 Impl::getBufferCapacity
int FireDetector::getBufferCapacity() const {
    return impl_->getBufferCapacity();
}

/// 委托给 Impl::isBufferFull
bool FireDetector::isBufferFull() const {
    return impl_->isBufferFull();
}

/// 委托给 Impl::getLastRoiFrame
const cv::Mat& FireDetector::getLastRoiFrame() const {
    return impl_->getLastRoiFrame();
}

//=============================================================================
// 工具函数实现
//=============================================================================

/**
 * @brief 将时序分类枚举转换为字符串
 *
 * @param cls 时序分类枚举值
 * @return 对应的字符串表示
 *
 * 【分类含义】
 * - STATIC: 静态火焰图片，如海报、电视屏幕上的火焰
 * - DYNAMIC: 动态真实火焰，具有闪烁和运动特征
 * - NEGATIVE: 负样本/误检，非火焰目标
 */
const char* temporalClassToString(TemporalClass cls) {
    switch (cls) {
        case TemporalClass::STATIC:   return "STATIC";
        case TemporalClass::DYNAMIC:  return "DYNAMIC";
        case TemporalClass::NEGATIVE: return "NEGATIVE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief 将时序分类转换为可视化颜色
 *
 * @param cls 时序分类枚举值
 * @return BGR 格式的颜色值，用于 OpenCV 绑定
 *
 * 【颜色编码规则】
 * - STATIC (橙色): 警告级别，存在火焰检测但为静态图片
 * - DYNAMIC (红色): 高危级别，检测到真实火焰
 * - NEGATIVE (绿色): 安全级别，无火焰或为误检
 *
 * 【使用场景】
 * 用于在可视化界面上用颜色区分不同的检测状态
 */
cv::Scalar temporalClassToColor(TemporalClass cls) {
    switch (cls) {
        case TemporalClass::STATIC:   return cv::Scalar(0, 165, 255);   // 橙色 (BGR)
        case TemporalClass::DYNAMIC:  return cv::Scalar(0, 0, 255);     // 红色 (BGR)
        case TemporalClass::NEGATIVE: return cv::Scalar(0, 255, 0);     // 绿色 (BGR)
        default: return cv::Scalar(128, 128, 128);  // 灰色，未知状态
    }
}

/**
 * @brief 将检测类别枚举转换为字符串
 *
 * @param cls 检测类别枚举值
 * @return 对应的字符串表示
 *
 * 【类别说明】
 * - FIRE: 火焰目标
 * - PERSON: 人员目标（可选检测，用于安全评估）
 * - SMOKE: 烟雾目标（火灾早期征兆）
 */
const char* detectionClassToString(DetectionClass cls) {
    switch (cls) {
        case DetectionClass::FIRE:   return "FIRE";
        case DetectionClass::PERSON: return "PERSON";
        case DetectionClass::SMOKE:  return "SMOKE";
        default: return "UNKNOWN";
    }
}

/**
 * @brief 获取库版本号
 * @return 版本字符串，格式为 "major.minor.patch"
 *
 * 【使用场景】
 * - 日志记录库版本信息
 * - 运行时版本兼容性检查
 * - 技术支持时提供版本信息
 */
const char* getVersion() {
    return FIRE_DETECTION_VERSION;
}

}  // namespace fire_detection
