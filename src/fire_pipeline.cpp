/**
 * @file fire_pipeline.cpp
 * @brief Fire detection pipeline implementation
 *
 * @author TrtEngineToolkits
 * @date 2025-04-22
 */

#include "fire_pipeline.h"
#include <trt_engine/logger.h>

#include <cmath>
#include <algorithm>
#include <numeric>

FireDetectionPipeline::FireDetectionPipeline(
    const std::string& segment_engine_path,
    const std::string& convlstm_engine_path)
    : segment_engine_path_(segment_engine_path),
      convlstm_engine_path_(convlstm_engine_path)
{
    segment_engine_ = std::make_unique<TrtEngineMultiTs>();
    convlstm_engine_ = std::make_unique<TrtEngineMultiTs>();
    postprocessor_ = std::make_unique<YOLOv8PostProcessor>(num_classes_);
}

FireDetectionPipeline::~FireDetectionPipeline() = default;

bool FireDetectionPipeline::initialize() {
    LOG_INFO("FirePipeline", "Initializing fire detection pipeline...");

    // Load segmentation engine
    LOG_INFO("FirePipeline", "Loading segmentation engine: " + segment_engine_path_);
    if (!segment_engine_->loadFromFile(segment_engine_path_)) {
        LOG_ERROR("FirePipeline", "Failed to load segmentation engine");
        return false;
    }

    // Load ConvLSTM engine
    LOG_INFO("FirePipeline", "Loading ConvLSTM engine: " + convlstm_engine_path_);
    if (!convlstm_engine_->loadFromFile(convlstm_engine_path_)) {
        LOG_ERROR("FirePipeline", "Failed to load ConvLSTM engine");
        return false;
    }

    // Create execution contexts
    if (!createSegmentContext()) {
        LOG_ERROR("FirePipeline", "Failed to create segmentation context");
        return false;
    }

    if (!createConvLSTMContext()) {
        LOG_ERROR("FirePipeline", "Failed to create ConvLSTM context");
        return false;
    }

    initialized_ = true;
    LOG_INFO("FirePipeline", "Pipeline initialized successfully");
    return true;
}

bool FireDetectionPipeline::createSegmentContext() {
    // Segment model: input (1, 3, 640, 640), output0 (1, 116, 8400), output1 (1, 32, 160, 160)
    std::vector<std::string> input_names = {"images"};
    std::vector<nvinfer1::Dims4> input_dims = {
        nvinfer1::Dims4{1, 3, input_height_, input_width_}
    };
    std::vector<std::string> output_names = {"output0", "output1"};

    if (!segment_engine_->createContext(input_names, input_dims, output_names)) {
        return false;
    }

    // Allocate tensors
    // Note: For YOLOv8-seg with 2 classes: output0 is (1, 4+2+32, 8400) = (1, 38, 8400)
    // But YOLO exports may vary, using standard 116 channels for COCO
    // Actual: 4 bbox + num_classes(80 for COCO or 2 for custom) + 32 mask = varies
    // We'll use 116 as per config, actual format determined by model export
    segment_input_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, 1, 3, input_height_, input_width_);
    segment_output0_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, 1, 116, 8400);
    segment_output1_ = std::make_unique<Tensor<float>>(TensorType::FLOAT32, 1, 32, 160, 160);

    return true;
}

bool FireDetectionPipeline::createConvLSTMContext() {
    // ConvLSTM: input (1, 10, 3, 640, 640), output (1, 3, 20, 20)
    std::vector<std::string> input_names = {"input"};
    std::vector<std::string> output_names = {"output"};

    // Create 5D dims for ConvLSTM input
    nvinfer1::Dims input_dim;
    input_dim.nbDims = 5;
    input_dim.d[0] = 1;
    input_dim.d[1] = sequence_length_;
    input_dim.d[2] = 3;
    input_dim.d[3] = input_height_;
    input_dim.d[4] = input_width_;

    std::vector<nvinfer1::Dims> input_dims = {input_dim};

    if (!convlstm_engine_->createContext(input_names, input_dims, output_names)) {
        return false;
    }

    // Allocate tensors (5D tensor)
    convlstm_input_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32,
        std::vector<int>{1, sequence_length_, 3, input_height_, input_width_}
    );
    convlstm_output_ = std::make_unique<Tensor<float>>(
        TensorType::FLOAT32,
        std::vector<int>{1, temporal_classes_, 20, 20}
    );

    return true;
}

void FireDetectionPipeline::preprocessFrame(
    const std::vector<uint8_t>& frame,
    int src_width, int src_height,
    std::vector<float>& output)
{
    // Resize and normalize: BGR HWC uint8 -> RGB CHW float [0,1]
    resizeAndNormalize(frame, src_width, src_height, output, input_width_, input_height_);
}

void FireDetectionPipeline::resizeAndNormalize(
    const std::vector<uint8_t>& src,
    int src_w, int src_h,
    std::vector<float>& dst,
    int dst_w, int dst_h)
{
    // Simple bilinear resize + BGR to RGB + normalize to [0,1]
    dst.resize(3 * dst_h * dst_w);

    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    for (int y = 0; y < dst_h; ++y) {
        float src_y = y * scale_y;
        int y0 = static_cast<int>(src_y);
        int y1 = std::min(y0 + 1, src_h - 1);
        float fy = src_y - y0;

        for (int x = 0; x < dst_w; ++x) {
            float src_x = x * scale_x;
            int x0 = static_cast<int>(src_x);
            int x1 = std::min(x0 + 1, src_w - 1);
            float fx = src_x - x0;

            // Bilinear interpolation for each channel
            for (int c = 0; c < 3; ++c) {
                // Source is BGR HWC
                int src_c = 2 - c;  // Convert BGR to RGB
                float v00 = src[(y0 * src_w + x0) * 3 + src_c];
                float v01 = src[(y0 * src_w + x1) * 3 + src_c];
                float v10 = src[(y1 * src_w + x0) * 3 + src_c];
                float v11 = src[(y1 * src_w + x1) * 3 + src_c];

                float val = (1 - fy) * ((1 - fx) * v00 + fx * v01) +
                           fy * ((1 - fx) * v10 + fx * v11);

                // Normalize to [0, 1] and store in CHW format
                dst[c * dst_h * dst_w + y * dst_w + x] = val / 255.0f;
            }
        }
    }
}

FrameResult FireDetectionPipeline::detectSingleFrame(
    const std::vector<uint8_t>& frame_data,
    int width, int height)
{
    FrameResult result;
    result.has_fire = false;
    result.has_smoke = false;

    if (!initialized_) {
        LOG_ERROR("FirePipeline", "Pipeline not initialized");
        return result;
    }

    // Preprocess frame
    std::vector<float> preprocessed;
    preprocessFrame(frame_data, width, height, preprocessed);

    // Upload to GPU
    segment_input_->copyFromVector(preprocessed);

    // Run inference
    std::vector<Tensor<float>*> inputs = {segment_input_.get()};
    std::vector<Tensor<float>*> outputs = {segment_output0_.get(), segment_output1_.get()};

    if (!segment_engine_->infer(inputs, outputs)) {
        LOG_ERROR("FirePipeline", "Segmentation inference failed");
        return result;
    }

    // Download results
    std::vector<float> output0, output1;
    segment_output0_->copyToVector(output0);
    segment_output1_->copyToVector(output1);

    // Post-process
    result.detections = postprocessor_->process(
        output0, output1, width, height, input_width_, input_height_);

    // Check for fire/smoke
    for (const auto& det : result.detections) {
        if (det.bbox.class_id == 0) result.has_fire = true;
        if (det.bbox.class_id == 1) result.has_smoke = true;
    }

    return result;
}

PipelineResult FireDetectionPipeline::process(
    const std::vector<std::vector<uint8_t>>& frames,
    int width, int height, int channels)
{
    PipelineResult result;
    result.frames_with_fire = 0;
    result.frames_with_smoke = 0;
    result.has_detection = false;
    result.predicted_class = FireClass::NEGATIVE;

    if (!initialized_) {
        LOG_ERROR("FirePipeline", "Pipeline not initialized");
        return result;
    }

    if (frames.size() != static_cast<size_t>(sequence_length_)) {
        LOG_ERROR("FirePipeline",
            "Expected " + std::to_string(sequence_length_) + " frames, got " +
            std::to_string(frames.size()));
        return result;
    }

    // Process each frame with YOLO and prepare ConvLSTM input
    std::vector<float> convlstm_data(sequence_length_ * 3 * input_height_ * input_width_);

    for (size_t i = 0; i < frames.size(); ++i) {
        // Run YOLO detection
        FrameResult frame_result = detectSingleFrame(frames[i], width, height);
        result.frame_results.push_back(frame_result);

        if (frame_result.has_fire) {
            result.frames_with_fire++;
            result.has_detection = true;
        }
        if (frame_result.has_smoke) {
            result.frames_with_smoke++;
            result.has_detection = true;
        }

        // Preprocess for ConvLSTM (already done in detectSingleFrame, reuse)
        std::vector<float> preprocessed;
        preprocessFrame(frames[i], width, height, preprocessed);

        // Copy to ConvLSTM input buffer
        size_t frame_size = 3 * input_height_ * input_width_;
        std::copy(preprocessed.begin(), preprocessed.end(),
                 convlstm_data.begin() + i * frame_size);
    }

    // Run ConvLSTM classification
    convlstm_input_->copyFromVector(convlstm_data);

    std::vector<Tensor<float>*> convlstm_inputs = {convlstm_input_.get()};
    std::vector<Tensor<float>*> convlstm_outputs = {convlstm_output_.get()};

    if (!convlstm_engine_->infer(convlstm_inputs, convlstm_outputs)) {
        LOG_ERROR("FirePipeline", "ConvLSTM inference failed");
        return result;
    }

    // Download and process ConvLSTM output
    std::vector<float> heatmap;
    convlstm_output_->copyToVector(heatmap);
    result.heatmap = heatmap;

    // Compute class probabilities from heatmap
    // heatmap: (1, 3, 20, 20) -> global average pooling then softmax
    int spatial_size = 20 * 20;
    std::vector<float> logits(temporal_classes_, 0.0f);

    for (int c = 0; c < temporal_classes_; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; ++i) {
            sum += heatmap[c * spatial_size + i];
        }
        logits[c] = sum / spatial_size;
    }

    result.class_probs = softmax(logits);

    // Find predicted class
    int max_idx = std::distance(result.class_probs.begin(),
        std::max_element(result.class_probs.begin(), result.class_probs.end()));
    result.predicted_class = static_cast<FireClass>(max_idx);

    return result;
}

PipelineResult FireDetectionPipeline::processNormalized(
    const std::vector<std::vector<float>>& frames,
    int width, int height)
{
    PipelineResult result;
    result.frames_with_fire = 0;
    result.frames_with_smoke = 0;
    result.has_detection = false;
    result.predicted_class = FireClass::NEGATIVE;

    if (!initialized_) {
        LOG_ERROR("FirePipeline", "Pipeline not initialized");
        return result;
    }

    if (frames.size() != static_cast<size_t>(sequence_length_)) {
        LOG_ERROR("FirePipeline",
            "Expected " + std::to_string(sequence_length_) + " frames, got " +
            std::to_string(frames.size()));
        return result;
    }

    // Prepare ConvLSTM input from pre-normalized frames
    std::vector<float> convlstm_data;
    convlstm_data.reserve(sequence_length_ * 3 * input_height_ * input_width_);

    for (const auto& frame : frames) {
        convlstm_data.insert(convlstm_data.end(), frame.begin(), frame.end());
    }

    // Run ConvLSTM classification
    convlstm_input_->copyFromVector(convlstm_data);

    std::vector<Tensor<float>*> convlstm_inputs = {convlstm_input_.get()};
    std::vector<Tensor<float>*> convlstm_outputs = {convlstm_output_.get()};

    if (!convlstm_engine_->infer(convlstm_inputs, convlstm_outputs)) {
        LOG_ERROR("FirePipeline", "ConvLSTM inference failed");
        return result;
    }

    // Download and process output
    std::vector<float> heatmap;
    convlstm_output_->copyToVector(heatmap);
    result.heatmap = heatmap;

    // Compute class probabilities
    int spatial_size = 20 * 20;
    std::vector<float> logits(temporal_classes_, 0.0f);

    for (int c = 0; c < temporal_classes_; ++c) {
        float sum = 0.0f;
        for (int i = 0; i < spatial_size; ++i) {
            sum += heatmap[c * spatial_size + i];
        }
        logits[c] = sum / spatial_size;
    }

    result.class_probs = softmax(logits);

    int max_idx = std::distance(result.class_probs.begin(),
        std::max_element(result.class_probs.begin(), result.class_probs.end()));
    result.predicted_class = static_cast<FireClass>(max_idx);

    return result;
}

std::vector<float> FireDetectionPipeline::softmax(const std::vector<float>& logits) {
    std::vector<float> probs(logits.size());
    float max_val = *std::max_element(logits.begin(), logits.end());

    float sum = 0.0f;
    for (size_t i = 0; i < logits.size(); ++i) {
        probs[i] = std::exp(logits[i] - max_val);
        sum += probs[i];
    }

    for (auto& p : probs) {
        p /= sum;
    }

    return probs;
}

void FireDetectionPipeline::setConfidenceThreshold(float threshold) {
    postprocessor_->setConfThreshold(threshold);
}

void FireDetectionPipeline::setIoUThreshold(float threshold) {
    postprocessor_->setIoUThreshold(threshold);
}

void FireDetectionPipeline::setNumClasses(int num_classes) {
    num_classes_ = num_classes;
    postprocessor_ = std::make_unique<YOLOv8PostProcessor>(num_classes);
}
