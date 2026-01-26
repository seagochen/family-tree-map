/**
 * @file yolov8_postprocess.h
 * @brief YOLOv8 segmentation post-processing utilities
 *
 * Provides post-processing for YOLOv8 instance segmentation model:
 * - Non-maximum suppression (NMS)
 * - Detection decoding
 * - Mask generation from prototype coefficients
 *
 * @author TrtEngineToolkits
 * @date 2025-04-22
 */

#ifndef TRT_ENGINE_YOLOV8_POSTPROCESS_H
#define TRT_ENGINE_YOLOV8_POSTPROCESS_H

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

/**
 * @brief Bounding box structure
 */
struct BBox {
    float x1, y1, x2, y2;  // Top-left and bottom-right coordinates
    float confidence;       // Detection confidence
    int class_id;           // Class index
    std::vector<float> mask_coeffs;  // 32 mask prototype coefficients
};

/**
 * @brief Detection result with mask
 */
struct Detection {
    BBox bbox;
    std::vector<float> mask;  // Instance mask (160x160 or resized)
    int mask_width;
    int mask_height;
};

/**
 * @brief YOLOv8 segmentation post-processor
 */
class YOLOv8PostProcessor {
public:
    /**
     * @brief Constructor
     * @param num_classes Number of classes (default: 2 for fire/smoke)
     * @param conf_threshold Confidence threshold for detection
     * @param iou_threshold IoU threshold for NMS
     */
    YOLOv8PostProcessor(int num_classes = 2,
                        float conf_threshold = 0.5f,
                        float iou_threshold = 0.45f);

    /**
     * @brief Process YOLOv8 segmentation output
     *
     * @param output0 Detection output (1, 116, 8400) - [x,y,w,h + num_classes + 32 mask coeffs]
     * @param output1 Mask prototypes (1, 32, 160, 160)
     * @param img_width Original image width
     * @param img_height Original image height
     * @param input_width Model input width (640)
     * @param input_height Model input height (640)
     * @return Vector of detections with masks
     */
    std::vector<Detection> process(
        const std::vector<float>& output0,
        const std::vector<float>& output1,
        int img_width, int img_height,
        int input_width = 640, int input_height = 640);

    /**
     * @brief Generate combined fire mask from detections
     * @param detections Vector of detections
     * @param width Output mask width
     * @param height Output mask height
     * @param class_id Class ID to filter (0=fire, 1=smoke, -1=all)
     * @return Binary mask as float vector
     */
    std::vector<float> generateCombinedMask(
        const std::vector<Detection>& detections,
        int width, int height, int class_id = -1);

    // Setters for thresholds
    void setConfThreshold(float threshold) { conf_threshold_ = threshold; }
    void setIoUThreshold(float threshold) { iou_threshold_ = threshold; }

private:
    int num_classes_;
    float conf_threshold_;
    float iou_threshold_;

    // Internal methods
    std::vector<BBox> decodeOutput(const std::vector<float>& output0,
                                   int input_width, int input_height);
    std::vector<BBox> nms(std::vector<BBox>& boxes);
    float computeIoU(const BBox& a, const BBox& b);
    std::vector<float> generateMask(const BBox& bbox,
                                    const std::vector<float>& prototypes,
                                    int proto_h, int proto_w);
    float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }
};

#endif // TRT_ENGINE_YOLOV8_POSTPROCESS_H
