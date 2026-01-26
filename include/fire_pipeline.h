/**
 * @file fire_pipeline.h
 * @brief Fire detection pipeline combining YOLO segmentation and ConvLSTM temporal classifier
 *
 * This class provides a complete fire detection pipeline:
 * 1. YOLO segmentation for fire/smoke detection in individual frames
 * 2. ConvLSTM temporal classification for distinguishing static vs dynamic patterns
 *
 * @author TrtEngineToolkits
 * @date 2025-04-22
 */

#ifndef TRT_ENGINE_FIRE_PIPELINE_H
#define TRT_ENGINE_FIRE_PIPELINE_H

#include <string>
#include <vector>
#include <memory>

#include <trt_engine/trt_engine.h>
#include "yolov8_postprocess.h"

/**
 * @brief Classification result from temporal classifier
 */
enum class FireClass {
    STATIC = 0,     // Static fire pattern (e.g., still image of fire)
    DYNAMIC = 1,    // Dynamic fire (real fire with movement)
    NEGATIVE = 2    // No fire
};

/**
 * @brief Detection result for a single frame
 */
struct FrameResult {
    std::vector<Detection> detections;  // YOLO detections
    bool has_fire;
    bool has_smoke;
};

/**
 * @brief Complete pipeline result
 */
struct PipelineResult {
    // Temporal classification
    FireClass predicted_class;
    std::vector<float> class_probs;  // Probabilities for each class (3,)
    std::vector<float> heatmap;      // Raw heatmap output (3, 20, 20)

    // Per-frame detection results
    std::vector<FrameResult> frame_results;

    // Summary
    int frames_with_fire;
    int frames_with_smoke;
    bool has_detection;

    // Convenience method to get class name
    static std::string getClassName(FireClass cls) {
        switch (cls) {
            case FireClass::STATIC: return "static";
            case FireClass::DYNAMIC: return "dynamic";
            case FireClass::NEGATIVE: return "negative";
            default: return "unknown";
        }
    }
};

/**
 * @brief Fire detection pipeline class
 *
 * Combines YOLOv8 segmentation and ConvLSTM temporal classification
 * for robust fire detection in video sequences.
 */
class FireDetectionPipeline {
public:
    /**
     * @brief Constructor
     * @param segment_engine_path Path to YOLOv8 segmentation TensorRT engine
     * @param convlstm_engine_path Path to ConvLSTM TensorRT engine
     */
    FireDetectionPipeline(const std::string& segment_engine_path,
                          const std::string& convlstm_engine_path);

    ~FireDetectionPipeline();

    // Non-copyable
    FireDetectionPipeline(const FireDetectionPipeline&) = delete;
    FireDetectionPipeline& operator=(const FireDetectionPipeline&) = delete;

    /**
     * @brief Initialize the pipeline (load engines, create contexts)
     * @return true on success
     */
    bool initialize();

    /**
     * @brief Check if pipeline is ready
     */
    bool isReady() const { return initialized_; }

    /**
     * @brief Process a sequence of frames
     *
     * @param frames Vector of image data (each frame: HWC format, BGR, uint8)
     * @param width Frame width
     * @param height Frame height
     * @param channels Number of channels (3)
     * @return Pipeline result with classification and detections
     */
    PipelineResult process(
        const std::vector<std::vector<uint8_t>>& frames,
        int width, int height, int channels = 3);

    /**
     * @brief Process pre-normalized float frames
     *
     * @param frames Vector of normalized float frames (each: CHW format, RGB, [0,1])
     * @param width Frame width
     * @param height Frame height
     * @return Pipeline result
     */
    PipelineResult processNormalized(
        const std::vector<std::vector<float>>& frames,
        int width, int height);

    /**
     * @brief Run YOLO segmentation on a single frame
     *
     * @param frame_data Frame data (HWC, BGR, uint8)
     * @param width Frame width
     * @param height Frame height
     * @return Frame detection result
     */
    FrameResult detectSingleFrame(
        const std::vector<uint8_t>& frame_data,
        int width, int height);

    // Configuration setters
    void setConfidenceThreshold(float threshold);
    void setIoUThreshold(float threshold);
    void setNumClasses(int num_classes);

    // Getters
    int getInputWidth() const { return input_width_; }
    int getInputHeight() const { return input_height_; }
    int getSequenceLength() const { return sequence_length_; }

private:
    // Engine paths
    std::string segment_engine_path_;
    std::string convlstm_engine_path_;

    // TensorRT engines
    std::unique_ptr<TrtEngineMultiTs> segment_engine_;
    std::unique_ptr<TrtEngineMultiTs> convlstm_engine_;

    // Post-processor
    std::unique_ptr<YOLOv8PostProcessor> postprocessor_;

    // GPU tensors
    std::unique_ptr<Tensor<float>> segment_input_;
    std::unique_ptr<Tensor<float>> segment_output0_;
    std::unique_ptr<Tensor<float>> segment_output1_;
    std::unique_ptr<Tensor<float>> convlstm_input_;
    std::unique_ptr<Tensor<float>> convlstm_output_;

    // Configuration
    int input_width_ = 640;
    int input_height_ = 640;
    int sequence_length_ = 10;
    int num_classes_ = 2;  // fire, smoke
    int temporal_classes_ = 3;  // static, dynamic, negative
    bool initialized_ = false;

    // Internal methods
    bool createSegmentContext();
    bool createConvLSTMContext();

    void preprocessFrame(const std::vector<uint8_t>& frame,
                        int src_width, int src_height,
                        std::vector<float>& output);

    void resizeAndNormalize(const std::vector<uint8_t>& src,
                           int src_w, int src_h,
                           std::vector<float>& dst,
                           int dst_w, int dst_h);

    std::vector<float> softmax(const std::vector<float>& logits);
};

#endif // TRT_ENGINE_FIRE_PIPELINE_H
