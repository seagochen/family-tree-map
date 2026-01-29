/**
 * @file yolov8_postprocess.cpp
 * @brief YOLOv8 segmentation post-processing implementation
 *
 * @author TrtEngineToolkits
 * @date 2025-04-22
 */

#include "detector/yolov8_postprocess.h"
#include <cstring>
#include <numeric>

YOLOv8PostProcessor::YOLOv8PostProcessor(int num_classes,
                                         float conf_threshold,
                                         float iou_threshold)
    : num_classes_(num_classes),
      conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold) {}

std::vector<Detection> YOLOv8PostProcessor::process(
    const std::vector<float>& output0,
    const std::vector<float>& output1,
    int img_width, int img_height,
    int input_width, int input_height)
{
    // output0: (1, 116, 8400) -> 116 = 4(bbox) + num_classes + 32(mask coeffs)
    // For 2 classes: 116 = 4 + 80 + 32 (COCO) or custom
    // Actually for custom: 116 = 4 + 2 + 32 = 38? No, YOLO exports differ
    // Let's assume: 4 + num_classes + 32 mask coeffs
    // output1: (1, 32, 160, 160) mask prototypes

    const int num_anchors = 8400;
    const int proto_h = 160;
    const int proto_w = 160;
    const int mask_coeffs = 32;

    // Decode detections
    std::vector<BBox> boxes = decodeOutput(output0, input_width, input_height);

    // Apply NMS
    std::vector<BBox> filtered = nms(boxes);

    // Generate masks for each detection
    std::vector<Detection> detections;
    detections.reserve(filtered.size());

    // Scale factors for converting from model coordinates to original image
    float scale_x = static_cast<float>(img_width) / input_width;
    float scale_y = static_cast<float>(img_height) / input_height;

    for (auto& bbox : filtered) {
        Detection det;

        // Scale bbox to original image size
        det.bbox = bbox;
        det.bbox.x1 *= scale_x;
        det.bbox.y1 *= scale_y;
        det.bbox.x2 *= scale_x;
        det.bbox.y2 *= scale_y;

        // Clamp to image bounds
        det.bbox.x1 = std::max(0.0f, std::min(det.bbox.x1, static_cast<float>(img_width)));
        det.bbox.y1 = std::max(0.0f, std::min(det.bbox.y1, static_cast<float>(img_height)));
        det.bbox.x2 = std::max(0.0f, std::min(det.bbox.x2, static_cast<float>(img_width)));
        det.bbox.y2 = std::max(0.0f, std::min(det.bbox.y2, static_cast<float>(img_height)));

        // Generate mask from prototypes
        det.mask = generateMask(bbox, output1, proto_h, proto_w);
        det.mask_width = proto_w;
        det.mask_height = proto_h;

        detections.push_back(std::move(det));
    }

    return detections;
}

std::vector<BBox> YOLOv8PostProcessor::decodeOutput(
    const std::vector<float>& output0,
    int input_width, int input_height)
{
    // YOLOv8 output format: (1, 4+num_classes+32, 8400)
    // After transpose we treat it as (8400, 4+num_classes+32)
    const int num_anchors = 8400;
    const int mask_coeffs = 32;
    const int channels = 4 + num_classes_ + mask_coeffs;  // 4 + 2 + 32 = 38 for 2 classes

    // Output is stored as (channels, num_anchors), need to read column-wise
    std::vector<BBox> boxes;
    boxes.reserve(num_anchors);

    for (int i = 0; i < num_anchors; ++i) {
        // Extract bbox: x_center, y_center, width, height
        float cx = output0[0 * num_anchors + i];
        float cy = output0[1 * num_anchors + i];
        float w = output0[2 * num_anchors + i];
        float h = output0[3 * num_anchors + i];

        // Find max class score
        float max_score = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes_; ++c) {
            float score = output0[(4 + c) * num_anchors + i];
            if (score > max_score) {
                max_score = score;
                max_class = c;
            }
        }

        // Filter by confidence
        if (max_score < conf_threshold_) {
            continue;
        }

        BBox bbox;
        // Convert from center format to corner format
        bbox.x1 = cx - w / 2.0f;
        bbox.y1 = cy - h / 2.0f;
        bbox.x2 = cx + w / 2.0f;
        bbox.y2 = cy + h / 2.0f;
        bbox.confidence = max_score;
        bbox.class_id = max_class;

        // Extract mask coefficients
        bbox.mask_coeffs.resize(mask_coeffs);
        for (int m = 0; m < mask_coeffs; ++m) {
            bbox.mask_coeffs[m] = output0[(4 + num_classes_ + m) * num_anchors + i];
        }

        boxes.push_back(bbox);
    }

    return boxes;
}

std::vector<BBox> YOLOv8PostProcessor::nms(std::vector<BBox>& boxes) {
    // Sort by confidence (descending)
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;

        result.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            // Only compare same class
            if (boxes[i].class_id != boxes[j].class_id) continue;

            float iou = computeIoU(boxes[i], boxes[j]);
            if (iou > iou_threshold_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
}

float YOLOv8PostProcessor::computeIoU(const BBox& a, const BBox& b) {
    float inter_x1 = std::max(a.x1, b.x1);
    float inter_y1 = std::max(a.y1, b.y1);
    float inter_x2 = std::min(a.x2, b.x2);
    float inter_y2 = std::min(a.y2, b.y2);

    float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    float inter_area = inter_w * inter_h;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - inter_area;

    return union_area > 0 ? inter_area / union_area : 0.0f;
}

std::vector<float> YOLOv8PostProcessor::generateMask(
    const BBox& bbox,
    const std::vector<float>& prototypes,
    int proto_h, int proto_w)
{
    // prototypes: (1, 32, 160, 160)
    // mask = sigmoid(mask_coeffs @ prototypes)
    const int mask_coeffs = 32;
    const int proto_size = proto_h * proto_w;

    std::vector<float> mask(proto_size, 0.0f);

    // Compute mask = sum(coeffs[i] * prototype[i]) for each pixel
    for (int i = 0; i < mask_coeffs; ++i) {
        float coeff = bbox.mask_coeffs[i];
        for (int j = 0; j < proto_size; ++j) {
            mask[j] += coeff * prototypes[i * proto_size + j];
        }
    }

    // Apply sigmoid
    for (int j = 0; j < proto_size; ++j) {
        mask[j] = sigmoid(mask[j]);
    }

    // Crop to bbox region and zero out outside
    // Scale bbox to prototype size (160x160 for 640x640 input)
    float scale = static_cast<float>(proto_w) / 640.0f;
    int bx1 = static_cast<int>(std::max(0.0f, bbox.x1 * scale));
    int by1 = static_cast<int>(std::max(0.0f, bbox.y1 * scale));
    int bx2 = static_cast<int>(std::min(static_cast<float>(proto_w), bbox.x2 * scale));
    int by2 = static_cast<int>(std::min(static_cast<float>(proto_h), bbox.y2 * scale));

    // Zero out pixels outside bbox
    for (int y = 0; y < proto_h; ++y) {
        for (int x = 0; x < proto_w; ++x) {
            if (x < bx1 || x >= bx2 || y < by1 || y >= by2) {
                mask[y * proto_w + x] = 0.0f;
            }
        }
    }

    return mask;
}

std::vector<float> YOLOv8PostProcessor::generateCombinedMask(
    const std::vector<Detection>& detections,
    int width, int height, int class_id)
{
    std::vector<float> combined(width * height, 0.0f);

    for (const auto& det : detections) {
        // Filter by class if specified
        if (class_id >= 0 && det.bbox.class_id != class_id) {
            continue;
        }

        // Scale mask to output size
        float scale_x = static_cast<float>(width) / det.mask_width;
        float scale_y = static_cast<float>(height) / det.mask_height;

        for (int y = 0; y < height; ++y) {
            int src_y = static_cast<int>(y / scale_y);
            src_y = std::min(src_y, det.mask_height - 1);

            for (int x = 0; x < width; ++x) {
                int src_x = static_cast<int>(x / scale_x);
                src_x = std::min(src_x, det.mask_width - 1);

                float val = det.mask[src_y * det.mask_width + src_x];
                if (val > 0.5f) {
                    combined[y * width + x] = 1.0f;
                }
            }
        }
    }

    return combined;
}
