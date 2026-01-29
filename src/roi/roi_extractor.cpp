/**
 * @file roi_extractor.cpp
 * @brief ROI（感兴趣区域）提取模块的实现
 *
 * 本模块负责从视频帧中提取火焰和烟雾的感兴趣区域。
 * 主要功能包括：
 *   1. 从YOLO检测结果中提取ROI检测信息
 *   2. 对检测掩码进行EMA（指数移动平均）平滑处理，减少帧间抖动
 *   3. 批量处理视频帧，生成火焰/烟雾掩码
 *   4. 支持多种检测模式：仅火焰、仅烟雾、全部检测
 *
 * 业务流程说明：
 *   - 火焰检测使用实例分割掩码，能够精确提取火焰轮廓
 *   - 烟雾检测使用边界框并取并集，因为烟雾通常是扩散的
 *   - EMA平滑用于视频流处理，使掩码在时间上保持连续性
 */

#include "roi/roi_extractor.h"
#include <algorithm>
#include <cmath>

namespace roi_extractor {

// =============================================================================
// ROIDetection 类实现
// =============================================================================

/**
 * @brief 从Detection对象创建ROIDetection
 *
 * 业务逻辑说明：
 *   将YOLO检测器输出的Detection结构转换为ROI处理所需的ROIDetection结构。
 *   Detection包含原始检测信息，ROIDetection增加了用于ROI处理的掩码数据。
 *   如果提供了分割掩码，会进行深拷贝以确保数据独立性。
 *
 * @param det 原始检测结果，包含边界框坐标、置信度和类别ID
 * @param mask_data 分割掩码数据（可选），用于火焰等需要精确轮廓的目标
 * @return 转换后的ROIDetection对象
 */
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

/**
 * @brief 获取检测结果的边界框
 *
 * 业务逻辑说明：
 *   将浮点坐标转换为整数像素坐标，生成OpenCV的Rect对象。
 *   该边界框用于后续的ROI裁剪和掩码应用操作。
 *
 * @return cv::Rect 整数像素坐标的边界框
 */
cv::Rect ROIDetection::getBBox() const {
    int ix1 = static_cast<int>(x1);
    int iy1 = static_cast<int>(y1);
    int ix2 = static_cast<int>(x2);
    int iy2 = static_cast<int>(y2);
    return cv::Rect(ix1, iy1, ix2 - ix1, iy2 - iy1);
}

// =============================================================================
// 工具函数实现
// =============================================================================

/**
 * @brief 根据类别ID获取对应的可视化颜色
 *
 * 业务逻辑说明：
 *   为不同类别的检测结果分配不同颜色，便于可视化区分。
 *   优先使用CLASS_COLORS中预定义的颜色（如火焰用红色、烟雾用灰色等）。
 *   如果类别未预定义，则循环使用默认颜色列表。
 *
 * @param class_id 类别ID
 * @return cv::Scalar BGR格式的颜色值
 */
cv::Scalar getClassColor(int class_id) {
    // 默认颜色列表，用于未预定义的类别
    static const std::vector<cv::Scalar> default_colors = {
        cv::Scalar(0, 0, 255),    // 红色
        cv::Scalar(0, 255, 0),    // 绿色
        cv::Scalar(255, 0, 0),    // 蓝色
        cv::Scalar(0, 255, 255),  // 黄色
        cv::Scalar(255, 0, 255),  // 品红色
    };

    // 优先查找预定义颜色
    auto it = CLASS_COLORS.find(class_id);
    if (it != CLASS_COLORS.end()) {
        return it->second;
    }
    // 使用默认颜色，通过取模循环使用
    return default_colors[class_id % default_colors.size()];
}

/**
 * @brief 从YOLO检测结果中提取ROI检测信息
 *
 * 业务逻辑说明：
 *   这是检测后处理的核心函数，负责筛选和转换YOLO输出。
 *   处理流程：
 *     1. 根据active_classes过滤目标类别（如只保留火焰和烟雾）
 *     2. 根据conf_threshold过滤低置信度检测
 *     3. 如果有分割掩码，将掩码缩放到帧尺寸并二值化
 *     4. 如果没有掩码，仅使用边界框信息
 *
 *   掩码处理策略：
 *     - 使用最近邻插值(INTER_NEAREST)保持掩码边缘锐利
 *     - 阈值0.5将概率掩码转换为二值掩码
 *
 * @param detections YOLO检测器输出的检测结果列表
 * @param masks 对应的分割掩码列表（可能为空）
 * @param frame_shape 目标帧尺寸，用于掩码缩放
 * @param active_classes 需要保留的类别ID集合
 * @param conf_threshold 置信度阈值
 * @return 筛选并转换后的ROIDetection列表
 */
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

    // 处理带掩码的检测结果（实例分割模式）
    if (!masks.empty() && masks.size() == detections.size()) {
        for (size_t i = 0; i < detections.size(); ++i) {
            const auto& det = detections[i];
            // 检查类别和置信度是否满足要求
            if (active_classes.count(det.bbox.class_id) > 0 && det.bbox.confidence >= conf_threshold) {
                cv::Mat mask_resized;
                // 将掩码缩放到帧尺寸
                if (masks[i].size() != frame_shape) {
                    cv::resize(masks[i], mask_resized, frame_shape, 0, 0, cv::INTER_NEAREST);
                } else {
                    mask_resized = masks[i].clone();
                }

                // 将概率掩码二值化，阈值0.5
                cv::Mat mask_binary;
                cv::threshold(mask_resized, mask_binary, 0.5, 1.0, cv::THRESH_BINARY);
                mask_binary.convertTo(mask_binary, CV_8U);

                result.push_back(ROIDetection::fromDetection(det, mask_binary));
            }
        }
    }
    // 处理无掩码的检测结果（仅边界框模式）
    else {
        for (const auto& det : detections) {
            if (active_classes.count(det.bbox.class_id) > 0 && det.bbox.confidence >= conf_threshold) {
                result.push_back(ROIDetection::fromDetection(det));
            }
        }
    }

    return result;
}

/**
 * @brief 计算多个边界框的并集区域
 *
 * 业务逻辑说明：
 *   主要用于烟雾检测场景。烟雾通常是扩散的，多个检测框可能检测到
 *   同一烟雾团的不同部分。计算并集可以获得完整的烟雾覆盖区域。
 *
 *   算法：遍历所有边界框，找出最小x1/y1和最大x2/y2，构成外接矩形。
 *
 * @param bboxes 边界框列表
 * @return 并集边界框，如果列表为空返回std::nullopt
 */
std::optional<cv::Rect> computeBBoxUnion(const std::vector<cv::Rect>& bboxes) {
    if (bboxes.empty()) {
        return std::nullopt;
    }

    // 初始化为极值，便于比较
    int x1_min = std::numeric_limits<int>::max();
    int y1_min = std::numeric_limits<int>::max();
    int x2_max = std::numeric_limits<int>::min();
    int y2_max = std::numeric_limits<int>::min();

    // 遍历所有边界框，更新极值
    for (const auto& bbox : bboxes) {
        x1_min = std::min(x1_min, bbox.x);
        y1_min = std::min(y1_min, bbox.y);
        x2_max = std::max(x2_max, bbox.x + bbox.width);
        y2_max = std::max(y2_max, bbox.y + bbox.height);
    }

    return cv::Rect(x1_min, y1_min, x2_max - x1_min, y2_max - y1_min);
}

/**
 * @brief 为边界框添加内边距
 *
 * 业务逻辑说明：
 *   扩展边界框以包含更多上下文信息，这对于火灾检测的二次分析很重要。
 *   例如，后续的时序模型可能需要观察火焰周围的区域来判断燃烧趋势。
 *
 *   边距计算：基于边界框自身尺寸的比例，而非固定像素值，
 *   这样可以适应不同大小的检测目标。
 *
 *   边界处理：确保扩展后的边界框不超出图像边界。
 *
 * @param bbox 原始边界框
 * @param frame_shape 帧尺寸，用于边界裁剪
 * @param padding_ratio 边距比例（相对于边界框尺寸）
 * @return 添加边距后的边界框
 */
cv::Rect addBBoxPadding(
    const cv::Rect& bbox,
    const cv::Size& frame_shape,
    float padding_ratio
) {
    int h = frame_shape.height;
    int w = frame_shape.width;

    // 计算宽度和高度方向的边距像素值
    int pad_w = static_cast<int>(bbox.width * padding_ratio);
    int pad_h = static_cast<int>(bbox.height * padding_ratio);

    // 扩展边界框并裁剪到图像边界内
    int x1 = std::max(0, bbox.x - pad_w);
    int y1 = std::max(0, bbox.y - pad_h);
    int x2 = std::min(w, bbox.x + bbox.width + pad_w);
    int y2 = std::min(h, bbox.y + bbox.height + pad_h);

    return cv::Rect(x1, y1, x2 - x1, y2 - y1);
}

/**
 * @brief 使用边界框掩码处理图像
 *
 * 业务逻辑说明：
 *   将边界框外的区域置为黑色，只保留边界框内的图像内容。
 *   这用于烟雾检测模式，因为烟雾使用边界框而非分割掩码。
 *
 *   实现细节：
 *     - 创建全黑背景图像
 *     - 将边界框区域的原始像素复制到结果图像
 *     - 使用OpenCV的ROI操作确保高效处理
 *
 * @param frame 原始图像帧
 * @param bbox 要保留的边界框区域
 * @return 处理后的图像，边界框外区域为黑色
 */
cv::Mat applyBBoxMask(const cv::Mat& frame, const cv::Rect& bbox) {
    cv::Mat masked_frame = cv::Mat::zeros(frame.size(), frame.type());

    // 将边界框裁剪到图像边界内，使用位运算符&实现矩形交集
    cv::Rect clipped_bbox = bbox & cv::Rect(0, 0, frame.cols, frame.rows);

    // 只有当裁剪后的边界框有效时才复制像素
    if (clipped_bbox.area() > 0) {
        frame(clipped_bbox).copyTo(masked_frame(clipped_bbox));
    }

    return masked_frame;
}

/**
 * @brief 使用分割掩码处理图像
 *
 * 业务逻辑说明：
 *   将多个检测目标的分割掩码合并，并应用到原始图像。
 *   主要用于火焰检测，因为火焰需要精确的轮廓提取。
 *
 *   处理流程：
 *     1. 创建空的组合掩码
 *     2. 将每个检测的掩码缩放到帧尺寸
 *     3. 使用max操作合并掩码（取并集）
 *     4. 将掩码外的区域置为黑色
 *
 * @param frame 原始图像帧
 * @param detections 包含分割掩码的检测结果列表
 * @return 处理后的图像，掩码外区域为黑色
 */
cv::Mat applySegmentationMask(const cv::Mat& frame, const std::vector<ROIDetection>& detections) {
    if (detections.empty()) {
        return cv::Mat::zeros(frame.size(), frame.type());
    }

    int h = frame.rows;
    int w = frame.cols;
    cv::Mat combined_mask = cv::Mat::zeros(h, w, CV_8U);

    // 合并所有检测的掩码
    for (const auto& det : detections) {
        if (!det.mask.empty()) {
            cv::Mat mask_resized;
            // 确保掩码尺寸与帧尺寸一致
            if (det.mask.size() != cv::Size(w, h)) {
                cv::resize(det.mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
            } else {
                mask_resized = det.mask;
            }
            // 使用max操作合并，实现掩码并集
            cv::max(combined_mask, mask_resized, combined_mask);
        }
    }

    // 应用掩码：将掩码为0的区域置为黑色
    cv::Mat masked_frame = frame.clone();
    masked_frame.setTo(cv::Scalar::all(0), combined_mask == 0);

    return masked_frame;
}

// =============================================================================
// MaskEMA 类实现 - 掩码指数移动平均平滑器
// =============================================================================

/**
 * @brief MaskEMA构造函数
 *
 * 业务逻辑说明：
 *   EMA（指数移动平均）用于视频流中的掩码平滑。
 *   在连续帧中，火焰的位置和形状可能有轻微变化，直接使用每帧的掩码
 *   会导致ROI区域抖动。EMA通过融合历史信息，使掩码边界更加稳定。
 *
 *   参数说明：
 *     - alpha: 平滑系数，值越大表示当前帧权重越高，响应越快但更抖动
 *     - blur_kernel: 边缘平滑的卷积核大小，用于后处理
 *
 * @param alpha EMA平滑系数，范围(0,1)，推荐值0.3-0.7
 * @param blur_kernel 边缘平滑核大小，必须为奇数
 */
MaskEMA::MaskEMA(float alpha, int blur_kernel)
    : alpha_(alpha), blur_kernel_(blur_kernel), has_ema_mask_(false) {}

/**
 * @brief 使用当前帧掩码更新EMA状态
 *
 * 业务逻辑说明：
 *   这是掩码平滑的核心函数，实现帧间的时序平滑。
 *
 *   EMA公式：new_mask = alpha * current + (1 - alpha) * old_mask
 *     - alpha较大时，新掩码权重高，跟踪快但抖动大
 *     - alpha较小时，历史权重高，稳定但可能滞后
 *
 *   处理流程：
 *     1. 将当前掩码转换为浮点数以便计算
 *     2. 如果是第一帧，直接使用当前掩码初始化
 *     3. 否则，用EMA公式融合当前帧和历史状态
 *     4. 将浮点结果二值化得到最终掩码
 *     5. 应用边缘平滑处理，减少锯齿
 *
 * @param current_mask 当前帧的二值掩码
 * @return 平滑后的二值掩码
 */
cv::Mat MaskEMA::update(const cv::Mat& current_mask) {
    cv::Mat current_float;
    current_mask.convertTo(current_float, CV_32F);

    if (!has_ema_mask_) {
        // 第一帧：直接初始化EMA状态
        ema_mask_ = current_float.clone();
        has_ema_mask_ = true;
    } else {
        // EMA公式：new = alpha * current + (1 - alpha) * old
        ema_mask_ = alpha_ * current_float + (1.0f - alpha_) * ema_mask_;
    }

    // 将浮点EMA结果二值化，阈值0.5
    cv::Mat smoothed_mask;
    cv::threshold(ema_mask_, smoothed_mask, 0.5, 1.0, cv::THRESH_BINARY);
    smoothed_mask.convertTo(smoothed_mask, CV_8U);

    // 应用边缘平滑，减少掩码边界的锯齿
    smoothed_mask = smoothEdges(smoothed_mask);

    return smoothed_mask;
}

/**
 * @brief 平滑掩码边缘
 *
 * 业务逻辑说明：
 *   分割掩码的边缘通常呈锯齿状，影响视觉效果和后续处理。
 *   本函数通过形态学操作和高斯模糊来平滑边缘。
 *
 *   处理流程：
 *     1. 形态学闭运算：填充掩码内部的小孔洞
 *        - 使用椭圆形结构元素，边缘更自然
 *     2. 高斯模糊：软化边缘，减少锯齿
 *     3. 重新二值化：恢复为二值掩码
 *
 * @param mask 输入的二值掩码
 * @return 边缘平滑后的二值掩码
 */
cv::Mat MaskEMA::smoothEdges(const cv::Mat& mask) {
    // 空掩码直接返回，避免无效操作
    if (cv::countNonZero(mask) == 0) {
        return mask;
    }

    int kernel_size = blur_kernel_;
    cv::Mat mask_uint8;
    mask.convertTo(mask_uint8, CV_8U);
    mask_uint8 *= 255;  // 转换为0-255范围以便形态学操作

    // 形态学闭运算：先膨胀后腐蚀，填充小孔洞
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::Mat mask_closed;
    cv::morphologyEx(mask_uint8, mask_closed, cv::MORPH_CLOSE, kernel);

    // 高斯模糊平滑边缘
    cv::Mat mask_blurred;
    cv::GaussianBlur(mask_closed, mask_blurred, cv::Size(kernel_size, kernel_size), 0);

    // 重新二值化，阈值127（0-255范围的中值）
    cv::Mat mask_smooth;
    cv::threshold(mask_blurred, mask_smooth, 127, 1, cv::THRESH_BINARY);
    mask_smooth.convertTo(mask_smooth, CV_8U);

    return mask_smooth;
}

/**
 * @brief 重置EMA状态
 *
 * 业务逻辑说明：
 *   在场景切换或需要重新开始平滑时调用。
 *   例如：切换视频源、检测到场景变化、手动重置等场景。
 */
void MaskEMA::reset() {
    ema_mask_.release();
    has_ema_mask_ = false;
}

// =============================================================================
// ROIPipeline 类实现 - ROI提取处理管道
// =============================================================================

/**
 * @brief ROIPipeline构造函数
 *
 * 业务逻辑说明：
 *   ROIPipeline是ROI提取的主入口，封装了完整的处理流程。
 *   它将检测、过滤、掩码生成和图像处理组合成统一的接口。
 *
 *   参数说明：
 *     - detector: 检测回调函数，由外部提供（如YOLO推理器）
 *     - confidence_threshold: 过滤低置信度检测
 *     - use_ema: 是否启用掩码平滑，视频流建议开启
 *     - ema_alpha: EMA系数，控制平滑程度
 *     - bbox_padding: 烟雾边界框的扩展比例
 *
 * @param detector 检测回调函数
 * @param confidence_threshold 置信度阈值
 * @param use_ema 是否使用EMA平滑
 * @param ema_alpha EMA平滑系数
 * @param bbox_padding 边界框扩展比例
 */
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

/**
 * @brief 批量处理视频帧
 *
 * 业务逻辑说明：
 *   这是ROI提取的核心处理函数，对一批视频帧进行完整的ROI提取处理。
 *
 *   设计思路：
 *     - 批量处理可以提高GPU利用率（如果检测器支持批处理）
 *     - 烟雾检测计算整个批次的并集边界框，使ROI在时间上保持一致
 *     - EMA平滑在帧间传递状态，实现时序连续性
 *
 *   处理流程：
 *     1. 预处理：将所有帧统一缩放到推理尺寸
 *     2. 检测：对每帧运行检测器
 *     3. 过滤：根据检测模式分离火焰和烟雾检测
 *     4. 烟雾合并：计算批次内所有烟雾检测的并集边界框
 *     5. 掩码生成：为每帧生成火焰掩码和烟雾掩码
 *     6. 应用掩码：将掩码应用到原始帧，生成ROI图像
 *
 *   检测模式说明：
 *     - FIRE：只检测火焰，使用分割掩码
 *     - SMOKE：只检测烟雾，使用边界框
 *     - ALL：同时检测火焰和烟雾
 *
 * @param frames 输入视频帧列表
 * @param detect_mode 检测模式
 * @param skip_no_detection 是否跳过无检测结果的帧
 * @return 批处理结果，包含处理后的帧和统计信息
 */
BatchResult ROIPipeline::processBatch(
    const std::vector<cv::Mat>& frames,
    DetectMode detect_mode,
    bool skip_no_detection
) {
    if (frames.empty()) {
        return BatchResult{};
    }

    // 步骤1：预处理所有帧 - 统一缩放到推理尺寸
    std::vector<cv::Mat> resized_frames;
    resized_frames.reserve(frames.size());
    for (const auto& f : frames) {
        resized_frames.push_back(preprocessFrame(f));
    }
    cv::Size shape(resized_frames[0].cols, resized_frames[0].rows);

    // 步骤2：对所有帧运行检测器
    std::vector<std::vector<ROIDetection>> all_detections;
    all_detections.reserve(frames.size());
    for (const auto& f : resized_frames) {
        all_detections.push_back(detector_(f, confidence_threshold_));
    }

    // 步骤3：根据检测模式过滤和分类检测结果
    std::vector<std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>> all_filtered;
    all_filtered.reserve(frames.size());
    for (const auto& dets : all_detections) {
        all_filtered.push_back(filterDetections(dets, detect_mode));
    }

    // 步骤4：计算整个批次的烟雾并集边界框
    // 这确保了批次内所有帧使用相同的烟雾ROI区域，提高时序一致性
    auto smoke_union_bbox = computeBatchSmokeUnion(all_filtered, shape);

    // 步骤5：初始化EMA平滑器（如果启用）
    std::unique_ptr<MaskEMA> mask_ema;
    if (use_ema_) {
        mask_ema = std::make_unique<MaskEMA>(ema_alpha_);
    }

    // 步骤6：逐帧处理，生成掩码并应用到图像
    std::vector<ProcessedFrame> results;
    int frames_with_fire = 0;
    int frames_with_smoke = 0;

    for (size_t idx = 0; idx < resized_frames.size(); ++idx) {
        const auto& resized_frame = resized_frames[idx];
        const auto& [fire_dets, smoke_dets] = all_filtered[idx];

        bool has_fire = !fire_dets.empty();
        bool has_smoke = !smoke_dets.empty() && smoke_union_bbox.has_value();

        // 统计检测帧数
        if (has_fire) {
            frames_with_fire++;
        }
        if (has_smoke) {
            frames_with_smoke++;
        }

        // 构建火焰掩码（使用分割掩码）和烟雾掩码（使用边界框）
        cv::Mat fire_mask = has_fire ?
            buildFireMask(fire_dets, shape, mask_ema.get()) :
            cv::Mat::zeros(shape, CV_8U);
        cv::Mat smoke_mask = has_smoke ?
            buildSmokeMask(smoke_union_bbox, shape) :
            cv::Mat::zeros(shape, CV_8U);

        // 将掩码应用到帧，生成最终ROI图像
        auto [output_frame, roi_type] = applyMaskToFrame(resized_frame, fire_mask, smoke_mask);

        // 构建处理结果
        ProcessedFrame result;
        result.frame_idx = static_cast<int>(idx);
        result.data = output_frame;
        result.roi_type = roi_type;
        result.has_fire = has_fire;
        result.has_smoke = has_smoke;
        result.detections = buildDetectionInfo(fire_dets, smoke_dets);

        // 根据配置决定是否跳过无检测结果的帧
        if (!skip_no_detection || result.hasDetections()) {
            results.push_back(std::move(result));
        }
    }

    // 构建批处理结果
    BatchResult batch_result;
    batch_result.frames = std::move(results);
    batch_result.smoke_union_bbox = smoke_union_bbox;
    batch_result.frames_with_fire = frames_with_fire;
    batch_result.frames_with_smoke = frames_with_smoke;

    return batch_result;
}

/**
 * @brief 预处理单帧图像
 *
 * 业务逻辑说明：
 *   将输入帧统一缩放到推理所需的固定尺寸。
 *   使用双线性插值(INTER_LINEAR)，在速度和质量间取得平衡。
 *   INFERENCE_WIDTH和INFERENCE_HEIGHT是模型训练时的输入尺寸。
 *
 * @param frame 原始输入帧
 * @return 缩放后的帧
 */
cv::Mat ROIPipeline::preprocessFrame(const cv::Mat& frame) {
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(INFERENCE_WIDTH, INFERENCE_HEIGHT), 0, 0, cv::INTER_LINEAR);
    return resized;
}

/**
 * @brief 根据检测模式过滤和分类检测结果
 *
 * 业务逻辑说明：
 *   将检测结果按目标类型分为火焰和烟雾两类。
 *
 *   分类依据：
 *     - SEGMENT_CLASSES：需要实例分割的类别（如火焰），使用精确掩码
 *     - BBOX_CLASSES：仅需边界框的类别（如烟雾），使用矩形区域
 *
 *   模式说明：
 *     - FIRE模式：只保留火焰检测
 *     - SMOKE模式：只保留烟雾检测
 *     - ALL模式：同时保留两者
 *
 * @param detections 全部检测结果
 * @param mode 检测模式
 * @return pair<火焰检测列表, 烟雾检测列表>
 */
std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>
ROIPipeline::filterDetections(const std::vector<ROIDetection>& detections, DetectMode mode) {
    std::vector<ROIDetection> fire_dets;
    std::vector<ROIDetection> smoke_dets;

    for (const auto& det : detections) {
        // 根据模式判断是否收集火焰检测
        if (mode == DetectMode::FIRE || mode == DetectMode::ALL) {
            if (SEGMENT_CLASSES.count(det.class_id) > 0) {
                fire_dets.push_back(det);
            }
        }
        // 根据模式判断是否收集烟雾检测
        if (mode == DetectMode::SMOKE || mode == DetectMode::ALL) {
            if (BBOX_CLASSES.count(det.class_id) > 0) {
                smoke_dets.push_back(det);
            }
        }
    }

    // 确保单一模式下另一类型为空
    if (mode == DetectMode::FIRE) {
        smoke_dets.clear();
    } else if (mode == DetectMode::SMOKE) {
        fire_dets.clear();
    }

    return {fire_dets, smoke_dets};
}

/**
 * @brief 计算批次内所有烟雾检测的并集边界框
 *
 * 业务逻辑说明：
 *   对于视频处理，在整个批次内使用统一的烟雾ROI区域有以下优势：
 *     1. 时序一致性：避免ROI在帧间跳动
 *     2. 覆盖完整：包含批次内所有可能的烟雾位置
 *     3. 简化后处理：统一的ROI便于后续的时序分析
 *
 *   处理步骤：
 *     1. 收集批次内所有烟雾检测的边界框
 *     2. 计算所有边界框的并集
 *     3. 添加padding扩展ROI范围
 *
 * @param all_filtered 所有帧的过滤后检测结果
 * @param shape 帧尺寸，用于padding计算
 * @return 并集边界框（带padding），如果无烟雾检测返回nullopt
 */
std::optional<cv::Rect> ROIPipeline::computeBatchSmokeUnion(
    const std::vector<std::pair<std::vector<ROIDetection>, std::vector<ROIDetection>>>& all_filtered,
    const cv::Size& shape
) {
    std::vector<cv::Rect> all_smoke_bboxes;

    // 收集所有帧的烟雾边界框
    for (const auto& [fire_dets, smoke_dets] : all_filtered) {
        for (const auto& det : smoke_dets) {
            all_smoke_bboxes.push_back(det.getBBox());
        }
    }

    if (all_smoke_bboxes.empty()) {
        return std::nullopt;
    }

    // 计算并集并添加padding
    auto union_bbox = computeBBoxUnion(all_smoke_bboxes);
    if (union_bbox.has_value()) {
        return addBBoxPadding(union_bbox.value(), shape, bbox_padding_);
    }

    return std::nullopt;
}

/**
 * @brief 构建火焰掩码
 *
 * 业务逻辑说明：
 *   合并多个火焰检测的分割掩码，并可选地应用EMA平滑。
 *
 *   处理流程：
 *     1. 创建空掩码
 *     2. 遍历所有火焰检测，将分割掩码缩放到统一尺寸
 *     3. 使用max操作合并掩码（取并集）
 *     4. 如果启用EMA，应用时序平滑
 *
 *   EMA的作用：
 *     - 减少掩码边界的帧间抖动
 *     - 在火焰短暂消失时保持ROI连续性
 *
 * @param fire_dets 火焰检测列表
 * @param shape 目标掩码尺寸
 * @param mask_ema EMA平滑器指针，可为nullptr
 * @return 合并并平滑后的火焰掩码
 */
cv::Mat ROIPipeline::buildFireMask(
    const std::vector<ROIDetection>& fire_dets,
    const cv::Size& shape,
    MaskEMA* mask_ema
) {
    int h = shape.height;
    int w = shape.width;
    cv::Mat fire_mask = cv::Mat::zeros(h, w, CV_8U);

    // 合并所有火焰检测的分割掩码
    for (const auto& det : fire_dets) {
        if (!det.mask.empty()) {
            cv::Mat mask_resized;
            // 确保掩码尺寸一致
            if (det.mask.size() != cv::Size(w, h)) {
                cv::resize(det.mask, mask_resized, cv::Size(w, h), 0, 0, cv::INTER_NEAREST);
            } else {
                mask_resized = det.mask;
            }
            // 使用max操作实现掩码并集
            cv::max(fire_mask, mask_resized, fire_mask);
        }
    }

    // 应用EMA时序平滑（如果配置）
    if (mask_ema != nullptr) {
        fire_mask = mask_ema->update(fire_mask);
    }

    return fire_mask;
}

/**
 * @brief 构建烟雾掩码
 *
 * 业务逻辑说明：
 *   烟雾使用边界框而非分割掩码，因为：
 *     1. 烟雾边界不如火焰清晰，分割意义较小
 *     2. 边界框计算简单，效率高
 *     3. 烟雾通常是扩散的，边界框更适合表示其范围
 *
 *   实现：将边界框内的区域设为1，其余为0
 *
 * @param smoke_union_bbox 烟雾并集边界框
 * @param shape 目标掩码尺寸
 * @return 烟雾掩码
 */
cv::Mat ROIPipeline::buildSmokeMask(
    const std::optional<cv::Rect>& smoke_union_bbox,
    const cv::Size& shape
) {
    int h = shape.height;
    int w = shape.width;
    cv::Mat smoke_mask = cv::Mat::zeros(h, w, CV_8U);

    if (smoke_union_bbox.has_value()) {
        cv::Rect bbox = smoke_union_bbox.value();
        // 将边界框裁剪到图像边界内
        bbox = bbox & cv::Rect(0, 0, w, h);
        if (bbox.area() > 0) {
            smoke_mask(bbox).setTo(1);  // 将边界框区域设为1
        }
    }

    return smoke_mask;
}

/**
 * @brief 将掩码应用到图像帧
 *
 * 业务逻辑说明：
 *   合并火焰和烟雾掩码，并将掩码应用到原始图像。
 *   掩码外的区域设为黑色，只保留ROI区域的图像内容。
 *
 *   返回值说明：
 *     - roi_type标识检测类型："fire"、"smoke"、"fire+smoke"或"none"
 *     - 此信息用于后续处理和日志记录
 *
 * @param frame 原始图像帧
 * @param fire_mask 火焰掩码
 * @param smoke_mask 烟雾掩码
 * @return pair<处理后的图像, ROI类型字符串>
 */
std::pair<cv::Mat, std::string> ROIPipeline::applyMaskToFrame(
    const cv::Mat& frame,
    const cv::Mat& fire_mask,
    const cv::Mat& smoke_mask
) {
    // 合并火焰和烟雾掩码，取并集
    cv::Mat combined_mask;
    cv::max(fire_mask, smoke_mask, combined_mask);

    // 检查各掩码是否非空
    bool has_fire = cv::countNonZero(fire_mask) > 0;
    bool has_smoke = cv::countNonZero(smoke_mask) > 0;

    if (cv::countNonZero(combined_mask) > 0) {
        // 将掩码外的区域置为黑色
        cv::Mat output_frame = frame.clone();
        output_frame.setTo(cv::Scalar::all(0), combined_mask == 0);

        // 确定ROI类型标识
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
        // 无检测时返回全黑图像
        return {cv::Mat::zeros(frame.size(), frame.type()), "none"};
    }
}

/**
 * @brief 构建检测信息列表
 *
 * 业务逻辑说明：
 *   将内部的ROIDetection结构转换为对外输出的DetectionInfo结构。
 *   DetectionInfo包含更易读的信息，如类别名称，便于日志记录和可视化。
 *
 * @param fire_dets 火焰检测列表
 * @param smoke_dets 烟雾检测列表
 * @return 合并后的检测信息列表
 */
std::vector<DetectionInfo> ROIPipeline::buildDetectionInfo(
    const std::vector<ROIDetection>& fire_dets,
    const std::vector<ROIDetection>& smoke_dets
) {
    std::vector<DetectionInfo> result;
    result.reserve(fire_dets.size() + smoke_dets.size());

    // Lambda函数：将ROIDetection转换为DetectionInfo
    auto addDetections = [&result](const std::vector<ROIDetection>& dets) {
        for (const auto& det : dets) {
            DetectionInfo info;
            info.class_id = det.class_id;
            // 查找类别名称，未找到则标记为"unknown"
            auto it = CLASS_NAMES.find(det.class_id);
            info.class_name = (it != CLASS_NAMES.end()) ? it->second : "unknown";
            info.confidence = det.confidence;
            info.bbox = det.getBBox();
            result.push_back(info);
        }
    };

    // 添加火焰和烟雾检测信息
    addDetections(fire_dets);
    addDetections(smoke_dets);

    return result;
}

}  // namespace roi_extractor
