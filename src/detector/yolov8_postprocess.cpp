/**
 * @file yolov8_postprocess.cpp
 * @brief YOLOv8 实例分割后处理模块实现
 *
 * 本模块负责处理YOLOv8-seg模型的原始输出，将其转换为可用的检测结果。
 * 主要功能包括：
 *   1. 解码模型输出，提取边界框和掩码系数
 *   2. 非极大值抑制（NMS）去除重叠检测
 *   3. 基于原型掩码生成实例分割掩码
 *
 * YOLOv8-seg模型输出说明：
 *   - output0: (1, 4+num_classes+32, 8400) 检测头输出
 *     - 4: 边界框参数 (cx, cy, w, h)
 *     - num_classes: 各类别置信度
 *     - 32: 掩码系数，用于与原型相乘生成实例掩码
 *   - output1: (1, 32, 160, 160) 原型掩码
 *     - 32个160×160的原型，通过线性组合生成实例掩码
 *
 * 业务流程说明：
 *   模型输出 → 解码边界框 → NMS去重 → 生成实例掩码 → 缩放到原图尺寸
 *
 * @author TrtEngineToolkits
 * @date 2025-04-22
 */

#include "detector/yolov8_postprocess.h"
#include <cstring>
#include <numeric>

/**
 * @brief YOLOv8后处理器构造函数
 *
 * @param num_classes 类别数量（如火焰检测为2类：火焰和烟雾）
 * @param conf_threshold 置信度阈值，低于此值的检测将被过滤
 * @param iou_threshold NMS的IoU阈值，高于此值的重叠框将被抑制
 */
YOLOv8PostProcessor::YOLOv8PostProcessor(int num_classes,
                                         float conf_threshold,
                                         float iou_threshold)
    : num_classes_(num_classes),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold) {}

/**
 * @brief 处理YOLOv8模型输出，生成检测结果
 *
 * 业务逻辑说明：
 *   这是后处理的主入口函数，完成从模型原始输出到最终检测结果的转换。
 *
 *   处理流程：
 *     1. 解码output0，提取所有候选框及其掩码系数
 *     2. 应用NMS去除高度重叠的冗余检测
 *     3. 为每个保留的检测生成实例分割掩码
 *     4. 将坐标从模型输入尺寸缩放到原始图像尺寸
 *
 *   坐标系统说明：
 *     - 模型坐标：相对于input_width × input_height（如640×640）
 *     - 原图坐标：相对于img_width × img_height（原始图像尺寸）
 *     - 需要进行缩放转换
 *
 * @param output0 检测头输出 (1, 4+num_classes+32, 8400)
 * @param output1 原型掩码输出 (1, 32, 160, 160)
 * @param img_width 原始图像宽度
 * @param img_height 原始图像高度
 * @param input_width 模型输入宽度（通常为640）
 * @param input_height 模型输入高度（通常为640）
 * @return 检测结果列表，包含边界框和实例掩码
 */
std::vector<Detection> YOLOv8PostProcessor::process(
    const std::vector<float>& output0,
    const std::vector<float>& output1,
    int img_width, int img_height,
    int input_width, int input_height)
{
    // YOLOv8输出格式说明：
    // output0: (1, 116, 8400)
    //   - 116 = 4(边界框) + num_classes(类别分数) + 32(掩码系数)
    //   - 8400 = 80×80 + 40×40 + 20×20 (三个检测头的锚点总数)
    // output1: (1, 32, 160, 160) 原型掩码

    const int num_anchors = 8400;   // 锚点总数
    const int proto_h = 160;        // 原型掩码高度
    const int proto_w = 160;        // 原型掩码宽度
    const int mask_coeffs = 32;     // 掩码系数数量

    // 步骤1：解码检测输出，提取边界框和掩码系数
    std::vector<BBox> boxes = decodeOutput(output0, input_width, input_height);

    // 步骤2：应用NMS去除重叠检测
    std::vector<BBox> filtered = nms(boxes);

    // 步骤3：为每个检测生成实例掩码
    std::vector<Detection> detections;
    detections.reserve(filtered.size());

    // 计算从模型坐标到原图坐标的缩放因子
    float scale_x = static_cast<float>(img_width) / input_width;
    float scale_y = static_cast<float>(img_height) / input_height;

    for (auto& bbox : filtered) {
        Detection det;

        // 将边界框坐标缩放到原始图像尺寸
        det.bbox = bbox;
        det.bbox.x1 *= scale_x;
        det.bbox.y1 *= scale_y;
        det.bbox.x2 *= scale_x;
        det.bbox.y2 *= scale_y;

        // 将边界框裁剪到图像边界内，防止越界
        det.bbox.x1 = std::max(0.0f, std::min(det.bbox.x1, static_cast<float>(img_width)));
        det.bbox.y1 = std::max(0.0f, std::min(det.bbox.y1, static_cast<float>(img_height)));
        det.bbox.x2 = std::max(0.0f, std::min(det.bbox.x2, static_cast<float>(img_width)));
        det.bbox.y2 = std::max(0.0f, std::min(det.bbox.y2, static_cast<float>(img_height)));

        // 使用原型掩码和掩码系数生成实例分割掩码
        det.mask = generateMask(bbox, output1, proto_h, proto_w);
        det.mask_width = proto_w;
        det.mask_height = proto_h;

        detections.push_back(std::move(det));
    }

    return detections;
}

/**
 * @brief 解码模型输出，提取边界框和掩码系数
 *
 * 业务逻辑说明：
 *   YOLOv8的输出是转置后的格式，需要按列读取每个锚点的信息。
 *
 *   输出张量布局 (channels, num_anchors)：
 *     - 第0-3行：边界框参数 (cx, cy, w, h)，中心点坐标和宽高
 *     - 第4到4+num_classes行：各类别的置信度分数
 *     - 最后32行：掩码系数，用于生成实例掩码
 *
 *   边界框格式转换：
 *     - 模型输出：中心点格式 (cx, cy, w, h)
 *     - 转换为：角点格式 (x1, y1, x2, y2)
 *     - 公式：x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2
 *
 *   置信度过滤：
 *     只保留最高类别分数超过conf_threshold的检测，
 *     这是第一步过滤，大幅减少后续NMS的计算量。
 *
 * @param output0 模型检测头输出
 * @param input_width 模型输入宽度
 * @param input_height 模型输入高度
 * @return 解码后的边界框列表（含掩码系数）
 */
std::vector<BBox> YOLOv8PostProcessor::decodeOutput(
    const std::vector<float>& output0,
    int input_width, int input_height)
{
    // YOLOv8输出格式: (1, 4+num_classes+32, 8400)
    // 转置后视为: (8400, 4+num_classes+32)，但内存中仍按列存储
    const int num_anchors = 8400;
    const int mask_coeffs = 32;
    const int channels = 4 + num_classes_ + mask_coeffs;  // 例如2类时: 4 + 2 + 32 = 38

    // 输出按(channels, num_anchors)存储，需要按列读取
    std::vector<BBox> boxes;
    boxes.reserve(num_anchors);

    // 遍历所有8400个锚点
    for (int i = 0; i < num_anchors; ++i) {
        // 提取边界框参数：中心点坐标和宽高
        // 索引计算：第row行第i列 = row * num_anchors + i
        float cx = output0[0 * num_anchors + i];  // 中心点x
        float cy = output0[1 * num_anchors + i];  // 中心点y
        float w = output0[2 * num_anchors + i];   // 宽度
        float h = output0[3 * num_anchors + i];   // 高度

        // 在所有类别中找到最高分数及其对应类别
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes_; ++c) {
            float score = output0[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }

        // 置信度过滤：跳过低于阈值的检测
        if (max_score < conf_threshold_) {
            continue;
        }

        BBox bbox;
        // 从中心点格式转换为角点格式
        bbox.x1 = cx - w / 2.0f;
        bbox.y1 = cy - h / 2.0f;
        bbox.x2 = cx + w / 2.0f;
        bbox.y2 = cy + h / 2.0f;
        bbox.confidence = max_score;
        bbox.class_id = max_class;

        // 提取32个掩码系数，用于后续生成实例掩码
        bbox.mask_coeffs.resize(mask_coeffs);
        for (int m = 0; m < mask_coeffs; ++m) {
            bbox.mask_coeffs[m] = output0[(4 + num_classes_ + m) * num_anchors + i];
        }

        boxes.push_back(bbox);
    }

    return boxes;
}

/**
 * @brief 非极大值抑制（NMS）
 *
 * 业务逻辑说明：
 *   目标检测中，同一个物体可能产生多个重叠的检测框。
 *   NMS通过抑制高度重叠的低置信度框来去除冗余检测。
 *
 *   算法流程：
 *     1. 按置信度降序排列所有候选框
 *     2. 从最高置信度开始，保留该框
 *     3. 计算该框与其他同类框的IoU
 *     4. 抑制IoU超过阈值的框（认为是同一物体的重复检测）
 *     5. 对下一个未被抑制的框重复步骤2-4
 *
 *   分类NMS说明：
 *     只对同一类别的框进行NMS，不同类别的框即使重叠也都保留。
 *     这允许不同类型的目标在空间上重叠（如火焰和烟雾）。
 *
 * @param boxes 输入候选框列表（会被排序修改）
 * @return NMS后保留的边界框列表
 */
std::vector<BBox> YOLOv8PostProcessor::nms(std::vector<BBox>& boxes) {
    // 按置信度降序排序
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    // 标记被抑制的框
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        // 跳过已被抑制的框
        if (suppressed[i]) continue;

        // 保留当前框（它是当前未抑制框中置信度最高的）
        result.push_back(boxes[i]);

        // 检查与后续框的重叠
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            // 只比较同一类别的框
            if (boxes[i].class_id != boxes[j].class_id) continue;

            // 计算IoU，超过阈值则抑制
            float iou = computeIoU(boxes[i], boxes[j]);
            if (iou > iou_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

/**
 * @brief 计算两个边界框的交并比（IoU）
 *
 * 业务逻辑说明：
 *   IoU（Intersection over Union）是衡量两个边界框重叠程度的标准指标。
 *
 *   计算公式：
 *     IoU = 交集面积 / 并集面积
 *     并集面积 = 面积A + 面积B - 交集面积
 *
 *   取值范围：
 *     - IoU = 0：两框完全不重叠
 *     - IoU = 1：两框完全重合
 *     - 典型NMS阈值：0.45-0.65
 *
 * @param a 第一个边界框
 * @param b 第二个边界框
 * @return IoU值，范围[0, 1]
 */
float YOLOv8PostProcessor::computeIoU(const BBox& a, const BBox& b) {
    // 计算交集区域的坐标
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    // 计算交集的宽高（可能为负，需要裁剪到0）
    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    // 计算各自的面积
    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);

    // 计算并集面积
    float union_area = area_a + area_b - inter_area;

    // 返回IoU，防止除零
    return union_area > 0 ? inter_area / union_area : 0.0f;
}

/**
 * @brief 生成实例分割掩码
 *
 * 业务逻辑说明：
 *   YOLOv8-seg使用原型掩码（prototypes）和掩码系数（coefficients）
 *   的线性组合来生成实例分割掩码，这是一种高效的掩码表示方法。
 *
 *   生成原理：
 *     - 模型学习32个160×160的原型掩码，作为"基底"
 *     - 每个检测有32个掩码系数，作为"权重"
 *     - 实例掩码 = sigmoid(sum(系数[i] × 原型[i]))
 *
 *   这种方法的优势：
 *     - 存储高效：只需存储32个原型 + 每个检测32个系数
 *     - 计算高效：简单的矩阵乘法
 *     - 质量较好：原型可以捕捉复杂的形状模式
 *
 *   边界框裁剪：
 *     生成的掩码会被裁剪到边界框范围内，框外像素置零。
 *     这确保掩码只在检测到的目标区域有效。
 *
 * @param bbox 边界框（含掩码系数）
 * @param prototypes 原型掩码 (1, 32, 160, 160)
 * @param proto_h 原型高度（160）
 * @param proto_w 原型宽度（160）
 * @return 实例分割掩码（160×160），值域[0,1]
 */
std::vector<float> YOLOv8PostProcessor::generateMask(
    const BBox& bbox,
    const std::vector<float>& prototypes,
    int proto_h, int proto_w)
{
    // 原型格式: (1, 32, 160, 160)
    // 掩码生成公式: mask = sigmoid(mask_coeffs @ prototypes)
    const int mask_coeffs = 32;
    const int proto_size = proto_h * proto_w;  // 160 × 160 = 25600

    // 初始化掩码为全零
    std::vector<float> mask(proto_size, 0.0f);

    // 计算掩码 = 各原型的加权和
    // 对每个像素位置：mask[j] = sum(coeffs[i] * prototype[i][j])
    for (int i = 0; i < mask_coeffs; ++i) {
        float coeff = bbox.mask_coeffs[i];
        for (int j = 0; j < proto_size; ++j) {
            // 原型按(32, 160*160)存储，第i个原型的第j个像素
            mask[j] += coeff * prototypes[i * proto_size + j];
        }
    }

    // 应用sigmoid激活函数，将值映射到[0,1]
    // sigmoid(x) = 1 / (1 + exp(-x))
    for (int j = 0; j < proto_size; ++j) {
        mask[j] = sigmoid(mask[j]);
    }

    // 将边界框外的掩码像素置零
    // 需要将边界框坐标从模型输入尺寸(640)缩放到原型尺寸(160)
    float scale = static_cast<float>(proto_w) / 640.0f;  // 160/640 = 0.25
    int bx1 = static_cast<int>(std::max(0.0f, bbox.x1 * scale));
    int by1 = static_cast<int>(std::max(0.0f, bbox.y1 * scale));
    int bx2 = static_cast<int>(std::min(static_cast<float>(proto_w), bbox.x2 * scale));
    int by2 = static_cast<int>(std::min(static_cast<float>(proto_h), bbox.y2 * scale));

    // 将边界框外的像素置零
    for (int y = 0; y < proto_h; ++y) {
        for (int x = 0; x < proto_w; ++x) {
            if (x < bx1 || x >= bx2 || y < by1 || y >= by2) {
                mask[y * proto_w + x] = 0.0f;
            }
        }
    }

    return mask;
}

/**
 * @brief 生成组合掩码
 *
 * 业务逻辑说明：
 *   将多个检测的实例掩码合并成一个组合掩码。
 *   这在需要获取某一类别所有实例的整体覆盖区域时很有用。
 *
 *   应用场景：
 *     - 获取所有火焰区域的总掩码
 *     - 生成用于ROI提取的二值掩码
 *     - 计算某类目标的总覆盖面积
 *
 *   实现方式：
 *     - 将各实例掩码缩放到目标尺寸
 *     - 使用最近邻插值（保持边缘锐利）
 *     - 阈值0.5二值化后取并集
 *
 * @param detections 检测结果列表
 * @param width 输出掩码宽度
 * @param height 输出掩码高度
 * @param class_id 要合并的类别ID，-1表示所有类别
 * @return 组合后的二值掩码（值为0或1）
 */
std::vector<float> YOLOv8PostProcessor::generateCombinedMask(
    const std::vector<Detection>& detections,
    int width, int height, int class_id)
{
    // 初始化组合掩码为全零
    std::vector<float> combined(width * height, 0.0f);

    for (const auto& det : detections) {
        // 根据class_id过滤：-1表示不过滤
        if (class_id >= 0 && det.bbox.class_id != class_id) {
            continue;
        }

        // 计算从原型尺寸到目标尺寸的缩放因子
        float scale_x = static_cast<float>(width) / det.mask_width;
        float scale_y = static_cast<float>(height) / det.mask_height;

        // 使用最近邻插值将掩码缩放到目标尺寸
        for (int y = 0; y < height; ++y) {
            // 计算源坐标
            int src_y = static_cast<int>(y / scale_y);
            src_y = std::min(src_y, det.mask_height - 1);

            for (int x = 0; x < width; ++x) {
                int src_x = static_cast<int>(x / scale_x);
                src_x = std::min(src_x, det.mask_width - 1);

                // 获取源掩码值
                float val = det.mask[src_y * det.mask_width + src_x];

                // 阈值0.5二值化后合并（取并集）
                if (val > 0.5f) {
                    combined[y * width + x] = 1.0f;
                }
            }
        }
    }

    return combined;
}
