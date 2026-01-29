/**
 * @file segment_detector.cpp
 * @brief YOLOv8-seg 分割检测器实现
 *
 * 本模块实现了基于 YOLOv8-seg 模型的实例分割检测器，用于火焰/烟雾目标检测。
 *
 * 【模块功能概述】
 * 1. 图像预处理：将输入图像进行 letterbox 缩放，转换为模型所需的输入格式
 * 2. TensorRT 推理：调用 GPU 加速的 TensorRT 引擎进行目标检测
 * 3. 后处理整合：调用 YOLOv8Postprocessor 解析模型输出，生成检测结果
 * 4. 结果转换：将检测结果转换为 ROI 提取模块所需的格式
 *
 * 【YOLOv8-seg 模型输入输出规格】
 * - 输入：images (1, 3, 640, 640) - RGB 格式，归一化到 [0, 1]
 * - 输出0：output0 (1, 116, 8400) - 检测结果
 *   - 116 = 4(边界框) + 80(COCO类别数) + 32(掩膜系数)
 *   - 8400 = 特征图上的候选框数量 (80×80 + 40×40 + 20×20)
 * - 输出1：output1 (1, 32, 160, 160) - 掩膜原型
 *   - 32 个 160×160 的原型掩膜，通过线性组合生成实例掩膜
 *
 * 【处理流程】
 * 输入帧 -> letterbox预处理 -> HWC转CHW -> TensorRT推理 -> 后处理解码 -> ROI检测结果
 */

#include "detector/segment_detector.h"

#include <iostream>
#include <cstring>

namespace fire_detection {
namespace internal {

/**
 * @brief 构造函数 - 初始化检测器参数
 *
 * @param conf_threshold 置信度阈值，低于此值的检测框将被过滤
 *                       典型值：0.25-0.5，值越高检测越严格
 * @param iou_threshold  NMS 的 IoU 阈值，用于过滤重叠框
 *                       典型值：0.45-0.7，值越低过滤越激进
 *
 * 【业务逻辑说明】
 * - 构造时仅保存参数，不进行实际初始化
 * - postprocessor_ 在此处创建，后续与主检测器共享阈值配置
 * - 延迟初始化设计允许在不同时机配置 TensorRT 引擎
 */
SegmentDetector::SegmentDetector(float conf_threshold, float iou_threshold)
    : conf_threshold_(conf_threshold)
    , iou_threshold_(iou_threshold)
    , postprocessor_(NUM_CLASSES, conf_threshold, iou_threshold)
{
}

/**
 * @brief 初始化检测器 - 配置 TensorRT 引擎和分配张量内存
 *
 * @param engine TensorRT 多时间戳引擎的引用
 * @return 初始化是否成功
 *
 * 【初始化流程详解】
 *
 * 1. 防止重复初始化
 *    - 检查 initialized_ 标志，避免重复分配资源
 *
 * 2. 创建执行上下文
 *    - 指定输入张量名称 "images" 和维度 (1, 3, 640, 640)
 *    - 指定输出张量名称 "output0" 和 "output1"
 *    - TensorRT 根据这些信息创建推理上下文
 *
 * 3. 分配输入/输出张量
 *    - input_tensor_: 预处理后的图像数据 (1, 3, H, W)
 *    - output0_tensor_: 检测结果张量 (1, 116, 8400)
 *      - 116 = 4(xywh) + NUM_CLASSES(类别分数) + 32(掩膜系数)
 *    - output1_tensor_: 掩膜原型张量 (1, 32, 160, 160)
 *
 * 【内存管理说明】
 * - 使用 unique_ptr 管理张量生命周期，确保自动释放
 * - 张量内存在 GPU 上分配，推理时直接使用
 */
bool SegmentDetector::initialize(TrtEngineMultiTs& engine) {
    // 防止重复初始化，避免内存泄漏
    if (initialized_) {
        return true;
    }

    engine_ = &engine;

    // ========== 步骤1: 创建 TensorRT 执行上下文 ==========
    // 配置模型的输入输出绑定信息
    // YOLOv8-seg 模型结构：
    // - 输入 "images": (1, 3, 640, 640) NCHW 格式的 RGB 图像
    // - 输出 "output0": (1, 116, 8400) 检测结果
    //   - 前 4 个通道: 边界框坐标 (cx, cy, w, h)，中心点格式
    //   - 中间 NUM_CLASSES 个通道: 各类别的置信度分数
    //   - 最后 32 个通道: 掩膜系数，用于生成实例分割掩膜
    // - 输出 "output1": (1, 32, 160, 160) 掩膜原型
    //   - 32 个 160×160 的基础掩膜模板
    //   - 通过与掩膜系数线性组合得到最终实例掩膜
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_dims = {
        nvinfer1::Dims4{1, 3, INPUT_HEIGHT, INPUT_WIDTH}
    };
    std::vector<std::string> output_names = {"output0", "output1"};

    if (!engine_->createContext(input_names, input_dims, output_names)) {
        std::cerr << "SegmentDetector: Failed to create execution context" << std::endl;
        return false;
    }

    // ========== 步骤2: 分配输入张量 ==========
    // 输入图像张量: (batch=1, channels=3, height=640, width=640)
    // 数据类型: float32，像素值归一化到 [0, 1]
    input_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, 3, INPUT_HEIGHT, INPUT_WIDTH
    );

    // ========== 步骤3: 分配输出张量 ==========
    // 输出0 - 检测结果张量: (1, 4 + NUM_CLASSES + 32, 8400)
    // - 8400 个候选检测框，每个框包含:
    //   - 4 个坐标值 (cx, cy, w, h)
    //   - NUM_CLASSES 个类别置信度
    //   - 32 个掩膜系数
    output0_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, 4 + NUM_CLASSES + NUM_MASK_COEFFS, 8400
    );

    // 输出1 - 掩膜原型张量: (1, 32, 160, 160)
    // - 32 个原型掩膜，每个大小为 160×160
    // - 实例掩膜 = sigmoid(Σ(系数_i × 原型_i))
    output1_tensor_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32, 1, NUM_MASK_COEFFS, MASK_PROTO_H, MASK_PROTO_W
    );

    initialized_ = true;
    std::cout << "SegmentDetector: Initialized successfully" << std::endl;

    return true;
}

/**
 * @brief 预处理输入帧 - 执行 letterbox 缩放和格式转换
 *
 * @param frame 输入的 BGR 格式图像
 * @return 元组 (缩放比例, x方向填充, y方向填充)
 *
 * 【Letterbox 预处理原理】
 *
 * 1. 保持宽高比缩放
 *    - 计算使图像适应目标尺寸的最小缩放比例
 *    - scale = min(target_w/src_w, target_h/src_h)
 *    - 这样可以确保图像不会被拉伸变形
 *
 * 2. 居中填充
 *    - 缩放后的图像可能无法完全填满目标尺寸
 *    - 在周围填充灰色像素 (114, 114, 114)，使其达到目标尺寸
 *    - 灰色值 114 是 YOLO 训练时使用的标准填充值
 *    - pad_x/pad_y 记录填充偏移量，用于后处理时坐标还原
 *
 * 3. 色彩空间转换
 *    - OpenCV 默认 BGR 格式 -> 模型需要 RGB 格式
 *    - 同时归一化像素值到 [0, 1] 范围
 *
 * 4. 内存布局转换 (HWC -> CHW)
 *    - OpenCV: (H, W, 3) 交错存储，每个像素的 RGB 连续
 *    - TensorRT: (3, H, W) 分离存储，同通道的像素连续
 *    - 这种转换对 GPU 推理更高效（数据访问更连续）
 *
 * 【坐标映射关系】
 * - 模型输出坐标 (x_model, y_model) 是相对于 640×640 的 letterbox 图像
 * - 原始图像坐标 = (模型坐标 - pad) / scale
 */
std::tuple<float, int, int> SegmentDetector::preprocessFrame(const cv::Mat& frame) {
    int orig_h = frame.rows;
    int orig_w = frame.cols;

    // ========== 步骤1: 计算 letterbox 缩放参数 ==========
    // 选择较小的缩放比例，确保图像完全适应目标尺寸
    // 例如: 1920×1080 图像 -> 640×640
    //       scale_w = 640/1920 = 0.333
    //       scale_h = 640/1080 = 0.593
    //       scale = min(0.333, 0.593) = 0.333
    //       new_size = (640, 360)，高度方向需要填充
    float scale = std::min(
        static_cast<float>(INPUT_WIDTH) / orig_w,
        static_cast<float>(INPUT_HEIGHT) / orig_h
    );
    int new_w = static_cast<int>(orig_w * scale);
    int new_h = static_cast<int>(orig_h * scale);

    // 计算居中填充的偏移量
    // 填充量在两侧均匀分布，使缩放后的图像居中
    int pad_x = (INPUT_WIDTH - new_w) / 2;
    int pad_y = (INPUT_HEIGHT - new_h) / 2;

    // ========== 步骤2: 执行缩放和 letterbox 填充 ==========
    // 先将图像缩放到计算出的新尺寸
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(new_w, new_h));

    // 创建填充了灰色背景的目标图像，然后将缩放后的图像复制到中心
    // 灰色值 114 是 YOLO 系列模型训练时使用的标准填充颜色
    cv::Mat padded(INPUT_HEIGHT, INPUT_WIDTH, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

    // ========== 步骤3: 色彩空间转换和归一化 ==========
    // BGR -> RGB 转换（OpenCV 默认读取 BGR，但模型训练使用 RGB）
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    // 转换为 32 位浮点数并归一化到 [0, 1]
    // 神经网络通常期望输入在 [0, 1] 或 [-1, 1] 范围
    cv::Mat float_img;
    rgb.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // ========== 步骤4: HWC 转 CHW 格式 ==========
    // OpenCV 内存布局 (HWC): [R0,G0,B0, R1,G1,B1, R2,G2,B2, ...]
    // TensorRT 内存布局 (CHW): [R0,R1,R2,..., G0,G1,G2,..., B0,B1,B2,...]
    //
    // 转换公式:
    // - 源索引 src_idx = (h * W + w) * 3 + c  (HWC 格式)
    // - 目标索引 dst_idx = c * H * W + h * W + w  (CHW 格式)
    std::vector<float> input_data(3 * INPUT_HEIGHT * INPUT_WIDTH);
    const float* src = reinterpret_cast<const float*>(float_img.data);
    for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < INPUT_HEIGHT; ++h) {
            for (int w = 0; w < INPUT_WIDTH; ++w) {
                int src_idx = (h * INPUT_WIDTH + w) * 3 + c;
                int dst_idx = c * INPUT_HEIGHT * INPUT_WIDTH + h * INPUT_WIDTH + w;
                input_data[dst_idx] = src[src_idx];
            }
        }
    }

    // 将预处理后的数据复制到 GPU 张量
    input_tensor_->copyFromVector(input_data);

    // 返回缩放参数，供后处理时将检测框坐标还原到原始图像尺寸
    return {scale, pad_x, pad_y};
}

/**
 * @brief 执行目标检测 - 完整的检测流水线
 *
 * @param frame 输入的 BGR 格式图像
 * @return 检测到的 ROI 列表，每个 ROI 包含边界框、类别、置信度和分割掩膜
 *
 * 【检测流程详解】
 *
 * 1. 预处理阶段
 *    - 执行 letterbox 缩放和格式转换
 *    - 获取缩放参数用于后续坐标还原
 *
 * 2. TensorRT 推理阶段
 *    - 将预处理后的图像送入 GPU 进行推理
 *    - 获取检测结果张量 (output0) 和掩膜原型张量 (output1)
 *
 * 3. 后处理阶段
 *    - 解码检测框坐标和类别
 *    - 执行 NMS 去除重叠框
 *    - 生成实例分割掩膜
 *
 * 4. 结果转换阶段
 *    - 将检测结果从 Detection 格式转换为 ROIDetection 格式
 *    - 缩放掩膜到原始图像尺寸
 *    - 过滤非目标类别（只保留火焰/烟雾）
 *
 * 【掩膜处理说明】
 * - 后处理器返回的掩膜是 160×160 的概率图
 * - 需要缩放到原始帧大小后二值化 (阈值 0.5)
 * - 二值化后的掩膜用于 ROI 区域提取
 */
std::vector<roi_extractor::ROIDetection> SegmentDetector::detect(const cv::Mat& frame) {
    // 初始化检查
    if (!initialized_) {
        std::cerr << "SegmentDetector: Not initialized" << std::endl;
        return {};
    }

    // ========== 阶段1: 图像预处理 ==========
    // 执行 letterbox 缩放、BGR->RGB 转换、HWC->CHW 转换
    // 返回的参数用于后处理时坐标还原（当前未使用，因为后处理器内部处理）
    auto [scale, pad_x, pad_y] = preprocessFrame(frame);

    // ========== 阶段2: TensorRT 推理 ==========
    // 构建输入输出张量指针数组
    std::vector<Tensor<float>*> inputs = {input_tensor_.get()};
    std::vector<Tensor<float>*> outputs = {output0_tensor_.get(), output1_tensor_.get()};

    // 调用 TensorRT 引擎执行 GPU 推理
    // 推理过程完全在 GPU 上执行，利用 CUDA 核心并行计算
    if (!engine_->infer(inputs, outputs)) {
        std::cerr << "SegmentDetector: Inference failed" << std::endl;
        return {};
    }

    // ========== 阶段3: 数据传输和后处理 ==========
    // 将 GPU 上的输出张量复制到 CPU 内存
    // output0_data: 检测结果 (1, 116, 8400) -> 展平为 1D 向量
    // output1_data: 掩膜原型 (1, 32, 160, 160) -> 展平为 1D 向量
    std::vector<float> output0_data, output1_data;
    output0_tensor_->copyToVector(output0_data);
    output1_tensor_->copyToVector(output1_data);

    // 更新后处理器的阈值配置（支持运行时动态调整）
    postprocessor_.setConfThreshold(conf_threshold_);
    postprocessor_.setIoUThreshold(iou_threshold_);

    // 调用后处理器解析模型输出
    // 后处理器负责: 解码边界框、NMS、生成实例掩膜
    // 返回的检测结果已经转换到原始图像坐标系
    // 传递letterbox参数以正确还原坐标
    std::vector<Detection> detections = postprocessor_.process(
        output0_data, output1_data,
        frame.cols, frame.rows,      // 原始图像尺寸
        INPUT_WIDTH, INPUT_HEIGHT,   // 模型输入尺寸
        scale, pad_x, pad_y          // letterbox缩放参数
    );

    // ========== 阶段4: 结果格式转换 ==========
    // 将后处理器返回的 Detection 格式转换为 ROIDetection 格式
    //
    // 掩膜处理流程:
    // 1. 后处理器返回的掩膜尺寸可能与原图不同
    // 2. 需要将掩膜缩放到原始帧尺寸
    // 3. 二值化处理: 概率 > 0.5 的像素标记为前景
    std::vector<cv::Mat> masks;
    for (const auto& det : detections) {
        // 从 Detection 结构中恢复 2D 掩膜矩阵
        cv::Mat mask(det.mask_height, det.mask_width, CV_32FC1);
        std::memcpy(mask.data, det.mask.data(), det.mask.size() * sizeof(float));

        // 将掩膜双线性插值缩放到原始帧尺寸
        cv::Mat resized_mask;
        cv::resize(mask, resized_mask, cv::Size(frame.cols, frame.rows));

        // 二值化: 概率 > 0.5 转为 1，否则为 0
        // 结果用于标识实例分割区域
        masks.push_back(resized_mask > 0.5f);
    }

    // 调用 ROI 提取器将检测结果转换为 ROIDetection 格式
    // extractDetectionsFromYolo 函数会:
    // - 过滤非目标类别（只保留 TARGET_CLASSES 中定义的火焰/烟雾类别）
    // - 过滤低置信度检测
    // - 组装完整的 ROIDetection 结构
    return roi_extractor::extractDetectionsFromYolo(
        detections, masks,
        cv::Size(frame.cols, frame.rows),
        roi_extractor::TARGET_CLASSES,
        conf_threshold_
    );
}

/**
 * @brief 设置置信度阈值
 *
 * @param threshold 新的置信度阈值，范围 [0, 1]
 *
 * 【业务逻辑说明】
 * - 同时更新主检测器和后处理器的阈值
 * - 阈值越高，检测越严格，误检率降低但漏检率可能上升
 * - 典型值:
 *   - 0.25: 宽松检测，适合需要高召回率的场景
 *   - 0.5: 平衡检测，适合大多数场景
 *   - 0.7+: 严格检测，适合需要高精度的场景
 */
void SegmentDetector::setConfThreshold(float threshold) {
    conf_threshold_ = threshold;
    postprocessor_.setConfThreshold(threshold);
}

/**
 * @brief 设置 NMS 的 IoU 阈值
 *
 * @param threshold 新的 IoU 阈值，范围 [0, 1]
 *
 * 【业务逻辑说明】
 * - IoU (Intersection over Union) 阈值控制 NMS 的重叠框过滤强度
 * - 阈值越低，过滤越激进，保留的框越少
 * - 阈值越高，允许更多重叠框保留
 * - 典型值:
 *   - 0.45: 激进过滤，适合目标稀疏的场景
 *   - 0.5-0.6: 平衡过滤，适合大多数场景
 *   - 0.7+: 宽松过滤，适合密集目标场景
 */
void SegmentDetector::setIoUThreshold(float threshold) {
    iou_threshold_ = threshold;
    postprocessor_.setIoUThreshold(threshold);
}

}  // namespace internal
}  // namespace fire_detection
