/**
 * @file scene_test.cpp
 * @brief SceneController 单元测试程序
 *
 * 测试内容：
 *   1. 功能检测 - 场景管理、帧处理、多场景隔离、重置操作
 *   2. 内存泄漏检测 - 反复创建/销毁/处理循环，监测RSS增长
 *   3. 精度检测 - 处理多个视频，评估检测率和时序分类准确性
 *
 * 处理后的视频输出带有标注（bbox、分类标签）。
 *
 * 使用方法:
 *   ./scene_controller_test <yolo_engine> <convlstm_engine> [video_dir]
 *
 * 示例:
 *   ./scene_controller_test segment.engine convlstm.engine .
 */

#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <numeric>
#include <cmath>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "scene/scene_controller.h"
#include "fire_detection_api.h"

//=============================================================================
// Test infrastructure
//=============================================================================

struct TestResult {
    std::string test_name;
    bool passed;
    std::string detail;
};

static std::vector<TestResult> g_test_results;

static void ASSERT_TRUE(bool condition, const std::string& test_name,
                         const std::string& detail = "") {
    g_test_results.push_back({test_name, condition, detail});
    if (condition) {
        std::cout << "  [PASS] " << test_name << "\n";
    } else {
        std::cout << "  [FAIL] " << test_name;
        if (!detail.empty()) std::cout << " -- " << detail;
        std::cout << "\n";
    }
}

static bool hasFailures() {
    for (const auto& r : g_test_results) {
        if (!r.passed) return true;
    }
    return false;
}

static void printFinalSummary() {
    int passed = 0, failed = 0;
    for (const auto& r : g_test_results) {
        if (r.passed) passed++; else failed++;
    }
    int total = passed + failed;

    std::cout << "\n========================================\n";
    std::cout << "  Final Summary\n";
    std::cout << "========================================\n";
    std::cout << "  Total tests:  " << total << "\n";
    std::cout << "  Passed:       " << passed << "\n";
    std::cout << "  Failed:       " << failed << "\n";
    if (total > 0) {
        std::cout << "  Pass rate:    "
                  << std::fixed << std::setprecision(1)
                  << (100.0 * passed / total) << "%\n";
    }
    std::cout << "========================================\n";

    if (failed > 0) {
        std::cout << "\n  Failed tests:\n";
        for (const auto& r : g_test_results) {
            if (!r.passed) {
                std::cout << "    - " << r.test_name;
                if (!r.detail.empty()) std::cout << " (" << r.detail << ")";
                std::cout << "\n";
            }
        }
    }
}

//=============================================================================
// Memory monitoring (Linux)
//=============================================================================

static long getRSSKb() {
    std::ifstream status_file("/proc/self/status");
    std::string line;
    while (std::getline(status_file, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line.substr(6));
            long rss_kb;
            iss >> rss_kb;
            return rss_kb;
        }
    }
    return -1;
}

//=============================================================================
// Ground truth & statistics
//=============================================================================

struct VideoGroundTruth {
    std::string filename;
    bool expect_fire;
    bool expect_smoke;
    fire_detection::TemporalClass expected_temporal;
    std::string description;
};

static std::vector<VideoGroundTruth> buildGroundTruthTable() {
    using TC = fire_detection::TemporalClass;
    return {
        {"fire1.mp4",         true,  false, TC::DYNAMIC,  "Real fire"},
        {"flower1.mp4",       false, false, TC::NEGATIVE, "Flower (negative)"},
        {"flower2.mp4",       false, false, TC::NEGATIVE, "Flower (negative)"},
        {"real1.mp4",         true,  true,  TC::DYNAMIC,  "Real fire scene"},
        {"real2.mp4",         true,  true,  TC::DYNAMIC,  "Real fire scene"},
        {"real3.mp4",         true,  true,  TC::DYNAMIC,  "Real fire scene"},
        {"smoke1.mp4",        true,  true,  TC::DYNAMIC,  "Smoke/fire scene"},
        {"smoke2.mp4",        true,  true,  TC::DYNAMIC,  "Smoke/fire scene"},
        {"static_flame.mp4",  true,  false, TC::STATIC,   "Static flame image"},
    };
}

struct VideoStats {
    std::string filename;
    int total_frames = 0;
    int fire_detected_frames = 0;
    int smoke_detected_frames = 0;
    int temporal_valid_frames = 0;
    int fire_alert_frames = 0;
    int temporal_static_frames = 0;
    int temporal_dynamic_frames = 0;
    int temporal_negative_frames = 0;
    float max_temporal_confidence = 0.0f;
    float sum_temporal_confidence = 0.0f;
    int total_fire_boxes = 0;
    int total_smoke_boxes = 0;

    float fireDetectionRate() const {
        return total_frames > 0 ? static_cast<float>(fire_detected_frames) / total_frames : 0;
    }
    float smokeDetectionRate() const {
        return total_frames > 0 ? static_cast<float>(smoke_detected_frames) / total_frames : 0;
    }
    float fireAlertRate() const {
        return total_frames > 0 ? static_cast<float>(fire_alert_frames) / total_frames : 0;
    }
    float avgTemporalConfidence() const {
        return temporal_valid_frames > 0 ? sum_temporal_confidence / temporal_valid_frames : 0;
    }
    std::string dominantTemporalClass() const {
        int max_count = std::max({temporal_static_frames, temporal_dynamic_frames,
                                  temporal_negative_frames});
        if (max_count == 0) return "N/A";
        if (max_count == temporal_dynamic_frames) return "DYNAMIC";
        if (max_count == temporal_static_frames) return "STATIC";
        return "NEGATIVE";
    }
};

//=============================================================================
// Video annotation drawing
//=============================================================================

static void drawAnnotatedFrame(
    cv::Mat& display,
    const fire_detection::SceneDetectionResult& result,
    int frame_idx, int total_frames,
    const std::string& video_name)
{
    // Fire bounding boxes (red)
    for (const auto& bbox : result.fire_boxes) {
        cv::Rect rect(
            static_cast<int>(bbox.x1), static_cast<int>(bbox.y1),
            static_cast<int>(bbox.width()), static_cast<int>(bbox.height())
        );
        // Clamp to frame bounds
        rect &= cv::Rect(0, 0, display.cols, display.rows);
        if (rect.area() <= 0) continue;

        cv::Scalar color(0, 0, 255); // Red BGR
        cv::rectangle(display, rect, color, 2);

        std::string label = "Fire " +
            std::to_string(static_cast<int>(bbox.confidence * 100)) + "%";
        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(display,
            cv::Point(rect.x, rect.y - text_sz.height - 5),
            cv::Point(rect.x + text_sz.width, rect.y),
            color, cv::FILLED);
        cv::putText(display, label,
            cv::Point(rect.x, rect.y - 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }

    // Smoke bounding boxes (yellow)
    for (const auto& bbox : result.smoke_boxes) {
        cv::Rect rect(
            static_cast<int>(bbox.x1), static_cast<int>(bbox.y1),
            static_cast<int>(bbox.width()), static_cast<int>(bbox.height())
        );
        rect &= cv::Rect(0, 0, display.cols, display.rows);
        if (rect.area() <= 0) continue;

        cv::Scalar color(0, 255, 255); // Yellow BGR
        cv::rectangle(display, rect, color, 2);

        std::string label = "Smoke " +
            std::to_string(static_cast<int>(bbox.confidence * 100)) + "%";
        int baseline = 0;
        cv::Size text_sz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
        cv::rectangle(display,
            cv::Point(rect.x, rect.y - text_sz.height - 5),
            cv::Point(rect.x + text_sz.width, rect.y),
            color, cv::FILLED);
        cv::putText(display, label,
            cv::Point(rect.x, rect.y - 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }

    // Frame info overlay (top-left)
    int y_offset = 30;
    cv::putText(display,
        video_name + "  Frame: " + std::to_string(frame_idx) + "/" + std::to_string(total_frames),
        cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);

    // Detection status
    y_offset += 28;
    std::string status;
    if (result.has_fire && result.has_smoke) status = "FIRE + SMOKE";
    else if (result.has_fire) status = "FIRE";
    else if (result.has_smoke) status = "SMOKE";
    else status = "No detection";
    cv::Scalar status_color = (result.has_fire || result.has_smoke)
        ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(display, "Detection: " + status,
        cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2);

    // Queue / buffer info
    y_offset += 28;
    cv::putText(display,
        "Queue: " + std::to_string(result.queue_size) +
        "  Buffer: " + std::to_string(result.convlstm_buffer_size),
        cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 0), 2);

    // Temporal analysis
    if (result.temporal_valid) {
        y_offset += 28;
        cv::Scalar cls_color = fire_detection::temporalClassToColor(result.temporal_class);
        std::string cls_text = std::string("Temporal: ") +
            fire_detection::temporalClassToString(result.temporal_class) +
            " (" + std::to_string(static_cast<int>(result.temporal_confidence * 100)) + "%)";
        cv::putText(display, cls_text,
            cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.6, cls_color, 2);

        // Class scores
        y_offset += 22;
        std::ostringstream scores_ss;
        scores_ss << std::fixed << std::setprecision(2)
                  << "S:" << result.class_scores[0]
                  << " D:" << result.class_scores[1]
                  << " N:" << result.class_scores[2];
        cv::putText(display, scores_ss.str(),
            cv::Point(10, y_offset), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(200, 200, 200), 1);
    }

    // Temporal class label (top-right)
    if (result.temporal_valid) {
        std::string temporal_label;
        switch (result.temporal_class) {
            case fire_detection::TemporalClass::STATIC:
                temporal_label = "Static Image"; break;
            case fire_detection::TemporalClass::DYNAMIC:
                temporal_label = "Real Fire"; break;
            case fire_detection::TemporalClass::NEGATIVE:
                temporal_label = "Negative"; break;
        }
        int baseline = 0;
        cv::Size sz = cv::getTextSize(temporal_label, cv::FONT_HERSHEY_SIMPLEX, 0.8, 2, &baseline);
        // Background for readability
        cv::rectangle(display,
            cv::Point(display.cols - sz.width - 15, 10),
            cv::Point(display.cols - 5, 10 + sz.height + 10),
            cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(display, temporal_label,
            cv::Point(display.cols - sz.width - 10, 10 + sz.height + 3),
            cv::FONT_HERSHEY_SIMPLEX, 0.8,
            fire_detection::temporalClassToColor(result.temporal_class), 2);
    }

    // Fire alert banner (center)
    if (result.isFireAlert()) {
        std::string alert = "!!! FIRE ALERT !!!";
        int baseline = 0;
        cv::Size sz = cv::getTextSize(alert, cv::FONT_HERSHEY_SIMPLEX, 1.2, 3, &baseline);
        int x = (display.cols - sz.width) / 2;
        // Background
        cv::rectangle(display,
            cv::Point(x - 5, 40),
            cv::Point(x + sz.width + 5, 40 + sz.height + 10),
            cv::Scalar(0, 0, 100), cv::FILLED);
        cv::putText(display, alert,
            cv::Point(x, 40 + sz.height + 3),
            cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
    }
}

//=============================================================================
// Test 1: Scene Management
//=============================================================================

static void testSceneManagement(fire_detection::SceneController& controller) {
    // isReady
    ASSERT_TRUE(controller.isReady(), "isReady returns true after init");

    // Register scenes
    ASSERT_TRUE(controller.registerScene(1), "registerScene(1) succeeds");
    ASSERT_TRUE(controller.registerScene(2), "registerScene(2) succeeds");
    ASSERT_TRUE(!controller.registerScene(1), "registerScene(1) duplicate returns false");

    // hasScene
    ASSERT_TRUE(controller.hasScene(1), "hasScene(1) returns true");
    ASSERT_TRUE(controller.hasScene(2), "hasScene(2) returns true");
    ASSERT_TRUE(!controller.hasScene(999), "hasScene(999) returns false");

    // Count and IDs
    ASSERT_TRUE(controller.getSceneCount() == 2, "getSceneCount returns 2",
                "got " + std::to_string(controller.getSceneCount()));
    auto ids = controller.getSceneIds();
    bool has_both = (std::find(ids.begin(), ids.end(), 1) != ids.end()) &&
                    (std::find(ids.begin(), ids.end(), 2) != ids.end());
    ASSERT_TRUE(has_both, "getSceneIds contains 1 and 2");

    // State on empty scene
    ASSERT_TRUE(controller.getSceneQueueSize(1) == 0, "empty scene queue size is 0");
    ASSERT_TRUE(controller.getSceneQueueSize(999) == -1, "nonexistent scene queue returns -1");
    ASSERT_TRUE(controller.getSceneBufferSize(1) == 0, "empty scene buffer size is 0");
    ASSERT_TRUE(controller.getSceneBufferSize(999) == -1, "nonexistent scene buffer returns -1");
    ASSERT_TRUE(!controller.isSceneReadyForInference(1), "empty scene not ready for inference");
    ASSERT_TRUE(!controller.isSceneReadyForInference(999), "nonexistent scene not ready");
    ASSERT_TRUE(controller.getLastRoiFrame(1).empty(), "empty scene ROI frame is empty");
    ASSERT_TRUE(controller.getLastRoiFrame(999).empty(), "nonexistent scene ROI is empty");

    // Unregister
    ASSERT_TRUE(controller.unregisterScene(1), "unregisterScene(1) succeeds");
    ASSERT_TRUE(!controller.hasScene(1), "hasScene(1) false after unregister");
    ASSERT_TRUE(controller.getSceneCount() == 1, "count is 1 after unregister");
    ASSERT_TRUE(!controller.unregisterScene(1), "unregister already removed returns false");

    // Re-register
    ASSERT_TRUE(controller.registerScene(1), "re-register scene(1) succeeds");
    ASSERT_TRUE(controller.hasScene(1), "hasScene(1) true after re-register");

    // Negative and zero IDs
    ASSERT_TRUE(controller.registerScene(-1), "registerScene(-1) succeeds");
    ASSERT_TRUE(controller.registerScene(0), "registerScene(0) succeeds");
    ASSERT_TRUE(controller.hasScene(-1), "hasScene(-1) returns true");
    ASSERT_TRUE(controller.hasScene(0), "hasScene(0) returns true");

    // Reset
    controller.resetScene(2); // no crash
    controller.resetAllScenes(); // no crash

    // Cleanup
    controller.unregisterScene(1);
    controller.unregisterScene(2);
    controller.unregisterScene(-1);
    controller.unregisterScene(0);
    ASSERT_TRUE(controller.getSceneCount() == 0, "all scenes removed");
}

//=============================================================================
// Test 2: Frame Processing
//=============================================================================

static void testFrameProcessing(fire_detection::SceneController& controller,
                                 const std::string& video_dir) {
    std::string video_path = video_dir + "/fire1.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "  Warning: Cannot open " << video_path << ", skipping test\n";
        ASSERT_TRUE(false, "open fire1.mp4", "file not found: " + video_path);
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    controller.registerScene(1);

    cv::Mat frame;
    int frame_idx = 0;
    bool fire_ever_detected = false;
    bool valid_boxes = true;
    int max_frames = 30;

    while (cap.read(frame) && frame_idx < max_frames) {
        int64_t ts = static_cast<int64_t>(frame_idx * 1000.0 / fps);
        auto result = controller.processFrame(1, frame, ts);

        if (frame_idx == 0) {
            ASSERT_TRUE(result.scene_id == 1, "processFrame returns scene_id 1");
        }

        if (result.has_fire) {
            fire_ever_detected = true;
            for (const auto& b : result.fire_boxes) {
                if (b.x1 >= b.x2 || b.y1 >= b.y2 || b.confidence <= 0) {
                    valid_boxes = false;
                }
            }
        }
        frame_idx++;
    }

    ASSERT_TRUE(controller.getSceneQueueSize(1) > 0, "queue size > 0 after processing");
    ASSERT_TRUE(fire_ever_detected, "fire detected in fire1.mp4");
    ASSERT_TRUE(valid_boxes, "fire_boxes have valid coordinates");

    // processFrameRaw auto-registers scene 2
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    if (cap.read(frame)) {
        auto result = controller.processFrameRaw(
            2, frame.data, frame.cols, frame.rows, frame.channels(), 0);
        ASSERT_TRUE(result.scene_id == 2, "processFrameRaw returns scene_id 2");
        ASSERT_TRUE(controller.hasScene(2), "processFrameRaw auto-registers scene");
    }

    // Empty frame
    {
        cv::Mat empty;
        auto result = controller.processFrame(1, empty, 999999);
        // Should not crash; result should have default values
        ASSERT_TRUE(true, "empty frame does not crash");
    }

    // ROI frame after processing
    cv::Mat roi = controller.getLastRoiFrame(1);
    ASSERT_TRUE(!roi.empty(), "getLastRoiFrame non-empty after processing");

    // Cleanup
    controller.unregisterScene(1);
    controller.unregisterScene(2);
    cap.release();
}

//=============================================================================
// Test 3: Multi-Scene Isolation
//=============================================================================

static void testMultiSceneIsolation(fire_detection::SceneController& controller,
                                     const std::string& video_dir) {
    cv::VideoCapture cap_fire(video_dir + "/fire1.mp4");
    cv::VideoCapture cap_flower(video_dir + "/flower1.mp4");
    cv::VideoCapture cap_smoke(video_dir + "/smoke1.mp4");

    if (!cap_fire.isOpened() || !cap_flower.isOpened() || !cap_smoke.isOpened()) {
        std::cerr << "  Warning: Cannot open required videos, skipping test\n";
        ASSERT_TRUE(false, "open multi-scene videos");
        return;
    }

    controller.registerScene(10);
    controller.registerScene(20);
    controller.registerScene(30);

    double fps_fire = cap_fire.get(cv::CAP_PROP_FPS);
    double fps_flower = cap_flower.get(cv::CAP_PROP_FPS);
    double fps_smoke = cap_smoke.get(cv::CAP_PROP_FPS);
    if (fps_fire <= 0) fps_fire = 30.0;
    if (fps_flower <= 0) fps_flower = 30.0;
    if (fps_smoke <= 0) fps_smoke = 30.0;

    cv::Mat f_fire, f_flower, f_smoke;
    bool fire_scene_has_fire = false;
    bool flower_scene_has_fire = false;
    bool smoke_scene_has_smoke = false;
    int frames_to_process = 20;

    for (int i = 0; i < frames_to_process; i++) {
        if (cap_fire.read(f_fire)) {
            int64_t ts = static_cast<int64_t>(i * 1000.0 / fps_fire);
            auto r = controller.processFrame(10, f_fire, ts);
            if (r.has_fire) fire_scene_has_fire = true;
        }
        if (cap_flower.read(f_flower)) {
            int64_t ts = static_cast<int64_t>(i * 1000.0 / fps_flower);
            auto r = controller.processFrame(20, f_flower, ts);
            if (r.has_fire) flower_scene_has_fire = true;
        }
        if (cap_smoke.read(f_smoke)) {
            int64_t ts = static_cast<int64_t>(i * 1000.0 / fps_smoke);
            auto r = controller.processFrame(30, f_smoke, ts);
            if (r.has_smoke) smoke_scene_has_smoke = true;
        }
    }

    // Independent queues
    int q10 = controller.getSceneQueueSize(10);
    int q20 = controller.getSceneQueueSize(20);
    int q30 = controller.getSceneQueueSize(30);
    ASSERT_TRUE(q10 > 0 && q20 > 0 && q30 > 0, "all scene queues have data");

    ASSERT_TRUE(fire_scene_has_fire, "fire scene detects fire");
    // flower may or may not trigger false fire, so we just note it
    ASSERT_TRUE(smoke_scene_has_smoke, "smoke scene detects smoke");

    // Reset scene 10, verify others unaffected
    controller.resetScene(10);
    ASSERT_TRUE(controller.getSceneQueueSize(10) == 0, "reset scene 10 queue cleared");
    ASSERT_TRUE(controller.getSceneQueueSize(20) == q20, "scene 20 unaffected by reset 10");
    ASSERT_TRUE(controller.getSceneQueueSize(30) == q30, "scene 30 unaffected by reset 10");

    // resetAllScenes
    controller.resetAllScenes();
    ASSERT_TRUE(controller.getSceneQueueSize(10) == 0, "resetAll clears scene 10");
    ASSERT_TRUE(controller.getSceneQueueSize(20) == 0, "resetAll clears scene 20");
    ASSERT_TRUE(controller.getSceneQueueSize(30) == 0, "resetAll clears scene 30");

    // Cleanup
    controller.unregisterScene(10);
    controller.unregisterScene(20);
    controller.unregisterScene(30);
    cap_fire.release();
    cap_flower.release();
    cap_smoke.release();
}

//=============================================================================
// Test 4: Reset Operations
//=============================================================================

static void testResetOperations(fire_detection::SceneController& controller,
                                 const std::string& video_dir) {
    std::string video_path = video_dir + "/fire1.mp4";
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "  Warning: Cannot open " << video_path << ", skipping test\n";
        ASSERT_TRUE(false, "open fire1.mp4 for reset test");
        return;
    }

    double fps = cap.get(cv::CAP_PROP_FPS);
    if (fps <= 0) fps = 30.0;

    controller.registerScene(100);

    // Process 15 frames
    cv::Mat frame;
    for (int i = 0; i < 15 && cap.read(frame); i++) {
        int64_t ts = static_cast<int64_t>(i * 1000.0 / fps);
        controller.processFrame(100, frame, ts);
    }

    ASSERT_TRUE(controller.getSceneQueueSize(100) > 0, "queue populated before reset");

    // resetScene
    controller.resetScene(100);
    ASSERT_TRUE(controller.getSceneQueueSize(100) == 0, "resetScene clears queue");
    ASSERT_TRUE(controller.getSceneBufferSize(100) == 0, "resetScene clears buffer");
    ASSERT_TRUE(!controller.isSceneReadyForInference(100), "not ready after reset");
    ASSERT_TRUE(controller.getLastRoiFrame(100).empty(), "ROI frame empty after reset");

    // Re-accumulate
    cap.set(cv::CAP_PROP_POS_FRAMES, 0);
    for (int i = 0; i < 15 && cap.read(frame); i++) {
        int64_t ts = static_cast<int64_t>(i * 1000.0 / fps);
        controller.processFrame(100, frame, ts);
    }
    ASSERT_TRUE(controller.getSceneQueueSize(100) > 0, "scene re-accumulates after reset");

    // resetAllScenes
    controller.resetAllScenes();
    ASSERT_TRUE(controller.getSceneQueueSize(100) == 0, "resetAll clears scene 100");

    // Cleanup
    controller.unregisterScene(100);
    cap.release();
}

//=============================================================================
// Test 5: Memory Leak Detection
//=============================================================================

static void testMemoryLeaks(fire_detection::SceneController& controller,
                             const std::string& video_dir) {
    long baseline_rss = getRSSKb();
    if (baseline_rss < 0) {
        std::cerr << "  Warning: Cannot read /proc/self/status, skipping memory test\n";
        ASSERT_TRUE(false, "read RSS baseline", "/proc/self/status not available");
        return;
    }
    std::cout << "  Baseline RSS: " << baseline_rss << " KB\n";

    // Cycle 1: Repeated register/unregister
    for (int i = 0; i < 100; i++) {
        controller.registerScene(5000 + i);
        controller.unregisterScene(5000 + i);
    }
    long rss_after_cycle1 = getRSSKb();
    std::cout << "  After 100 register/unregister cycles: " << rss_after_cycle1
              << " KB (+" << (rss_after_cycle1 - baseline_rss) << " KB)\n";

    // Cycle 2: Repeated video process cycles with smoke2.mp4 (small file)
    std::string video_path = video_dir + "/smoke2.mp4";
    for (int cycle = 0; cycle < 5; cycle++) {
        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cerr << "  Warning: Cannot open " << video_path << ", reducing cycles\n";
            break;
        }
        double fps = cap.get(cv::CAP_PROP_FPS);
        if (fps <= 0) fps = 30.0;

        int scene_id = 6000 + cycle;
        controller.registerScene(scene_id);

        cv::Mat frame;
        int idx = 0;
        while (cap.read(frame)) {
            int64_t ts = static_cast<int64_t>(idx * 1000.0 / fps);
            controller.processFrame(scene_id, frame, ts);
            idx++;
        }
        controller.unregisterScene(scene_id);
        cap.release();
    }
    long rss_after_cycle2 = getRSSKb();
    std::cout << "  After 5 video process cycles: " << rss_after_cycle2
              << " KB (+" << (rss_after_cycle2 - baseline_rss) << " KB)\n";

    // Cycle 3: Process + reset + process again
    {
        cv::VideoCapture cap(video_path);
        if (cap.isOpened()) {
            double fps = cap.get(cv::CAP_PROP_FPS);
            if (fps <= 0) fps = 30.0;

            controller.registerScene(500);
            cv::Mat frame;
            int idx = 0;

            // First 200 frames (or all if less)
            while (cap.read(frame) && idx < 200) {
                int64_t ts = static_cast<int64_t>(idx * 1000.0 / fps);
                controller.processFrame(500, frame, ts);
                idx++;
            }
            controller.resetScene(500);

            // Restart video
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            idx = 0;
            while (cap.read(frame) && idx < 200) {
                int64_t ts = static_cast<int64_t>(idx * 1000.0 / fps);
                controller.processFrame(500, frame, ts);
                idx++;
            }
            controller.unregisterScene(500);
            cap.release();
        }
    }

    long final_rss = getRSSKb();
    long growth = final_rss - baseline_rss;
    std::cout << "  Final RSS: " << final_rss << " KB (+" << growth << " KB total growth)\n";

    // Pass if growth < 50MB (51200 KB)
    const long threshold_kb = 51200;
    ASSERT_TRUE(growth < threshold_kb, "RSS growth within threshold",
                "growth=" + std::to_string(growth) + "KB, threshold=" +
                std::to_string(threshold_kb) + "KB");
}

//=============================================================================
// Test 6: Precision/Accuracy + Output Video
//=============================================================================

static cv::VideoWriter createVideoWriter(const std::string& path, double fps,
                                          cv::Size frame_size) {
    // Try mp4v codec first
    cv::VideoWriter writer(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                           fps, frame_size);
    if (writer.isOpened()) return writer;

    // Fallback: XVID with .avi
    std::string avi_path = path.substr(0, path.rfind('.')) + ".avi";
    writer.open(avi_path, cv::VideoWriter::fourcc('X', 'V', 'I', 'D'),
                fps, frame_size);
    if (writer.isOpened()) {
        std::cout << "    (using XVID fallback: " << avi_path << ")\n";
    }
    return writer;
}

static void testPrecisionAccuracy(fire_detection::SceneController& controller,
                                   const std::string& video_dir,
                                   const std::vector<VideoGroundTruth>& ground_truth) {
    std::vector<VideoStats> all_stats;

    for (size_t vi = 0; vi < ground_truth.size(); vi++) {
        const auto& gt = ground_truth[vi];
        std::string video_path = video_dir + "/" + gt.filename;

        cv::VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            std::cout << "  Skipping " << gt.filename << " (not found)\n";
            continue;
        }

        int total_frames = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
        double fps = cap.get(cv::CAP_PROP_FPS);
        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        if (fps <= 0) fps = 30.0;

        std::string output_path = video_dir + "/output_" + gt.filename;
        cv::VideoWriter writer = createVideoWriter(output_path, fps, cv::Size(width, height));
        if (!writer.isOpened()) {
            std::cerr << "  Warning: Cannot create output video " << output_path << "\n";
        }

        int32_t scene_id = static_cast<int32_t>(1000 + vi);
        controller.registerScene(scene_id);

        VideoStats stats;
        stats.filename = gt.filename;

        cv::Mat frame;
        int frame_idx = 0;

        std::cout << "  Processing " << gt.filename
                  << " [" << total_frames << " frames]... " << std::flush;

        while (cap.read(frame)) {
            int64_t ts = static_cast<int64_t>(frame_idx * 1000.0 / fps);
            auto result = controller.processFrame(scene_id, frame, ts);

            // Update stats
            stats.total_frames++;
            if (result.has_fire) stats.fire_detected_frames++;
            if (result.has_smoke) stats.smoke_detected_frames++;
            stats.total_fire_boxes += static_cast<int>(result.fire_boxes.size());
            stats.total_smoke_boxes += static_cast<int>(result.smoke_boxes.size());

            if (result.temporal_valid) {
                stats.temporal_valid_frames++;
                if (result.isFireAlert()) stats.fire_alert_frames++;
                switch (result.temporal_class) {
                    case fire_detection::TemporalClass::STATIC:
                        stats.temporal_static_frames++; break;
                    case fire_detection::TemporalClass::DYNAMIC:
                        stats.temporal_dynamic_frames++; break;
                    case fire_detection::TemporalClass::NEGATIVE:
                        stats.temporal_negative_frames++; break;
                }
                stats.sum_temporal_confidence += result.temporal_confidence;
                stats.max_temporal_confidence = std::max(stats.max_temporal_confidence,
                                                         result.temporal_confidence);
            }

            // Write annotated frame
            if (writer.isOpened()) {
                cv::Mat display = frame.clone();
                drawAnnotatedFrame(display, result, frame_idx, total_frames, gt.filename);
                writer.write(display);
            }

            frame_idx++;
        }

        cap.release();
        if (writer.isOpened()) writer.release();
        controller.resetScene(scene_id);
        controller.unregisterScene(scene_id);

        std::cout << "done (" << output_path << ")\n";
        all_stats.push_back(stats);
    }

    // Print precision summary table
    std::cout << "\n  Precision Summary:\n";
    std::cout << "  " << std::left
              << std::setw(22) << "Video" << "| "
              << std::setw(7) << "Frames" << "| "
              << std::setw(9) << "FireDet%" << "| "
              << std::setw(10) << "SmokeDet%" << "| "
              << std::setw(10) << "Dominant" << "| "
              << std::setw(8) << "AvgConf" << "| "
              << std::setw(11) << "Boxes(F/S)" << "| "
              << "Result\n";
    std::cout << "  " << std::string(22, '-') << "|" << std::string(8, '-')
              << "|" << std::string(10, '-') << "|" << std::string(11, '-')
              << "|" << std::string(11, '-') << "|" << std::string(9, '-')
              << "|" << std::string(12, '-') << "|" << std::string(7, '-') << "\n";

    for (size_t i = 0; i < all_stats.size(); i++) {
        const auto& s = all_stats[i];

        // Find corresponding ground truth
        const VideoGroundTruth* gt_ptr = nullptr;
        for (const auto& gt : ground_truth) {
            if (gt.filename == s.filename) { gt_ptr = &gt; break; }
        }

        std::string boxes_str = std::to_string(s.total_fire_boxes) + "/" +
                                std::to_string(s.total_smoke_boxes);

        std::cout << "  " << std::left
                  << std::setw(22) << s.filename << "| "
                  << std::setw(7) << s.total_frames << "| "
                  << std::setw(8) << std::fixed << std::setprecision(1)
                  << (s.fireDetectionRate() * 100) << "%| "
                  << std::setw(9) << (s.smokeDetectionRate() * 100) << "%| "
                  << std::setw(10) << s.dominantTemporalClass() << "| "
                  << std::setw(8) << std::setprecision(2)
                  << s.avgTemporalConfidence() << "| "
                  << std::setw(11) << boxes_str << "| ";

        // Evaluate accuracy
        if (gt_ptr) {
            bool pass = true;
            std::string reason;

            if (gt_ptr->expect_fire) {
                if (s.fireDetectionRate() < 0.10f) {
                    pass = false;
                    reason = "low fire detection rate";
                }
            } else {
                if (s.fireDetectionRate() > 0.05f) {
                    pass = false;
                    reason = "false fire positive";
                }
            }

            if (gt_ptr->expect_smoke && !gt_ptr->expect_fire) {
                // Pure smoke video
                if (s.smokeDetectionRate() < 0.10f) {
                    pass = false;
                    reason = "low smoke detection rate";
                }
            }

            if (s.temporal_valid_frames > 0) {
                std::string dominant = s.dominantTemporalClass();
                std::string expected = fire_detection::temporalClassToString(
                    gt_ptr->expected_temporal);
                if (dominant != expected && dominant != "N/A") {
                    pass = false;
                    reason = "temporal mismatch: " + dominant + " vs " + expected;
                }
            }

            std::cout << (pass ? "PASS" : "FAIL") << "\n";
            ASSERT_TRUE(pass, s.filename + " accuracy", reason);
        } else {
            std::cout << "SKIP\n";
        }
    }
}

//=============================================================================
// Main
//=============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "SceneController Test v" << fire_detection::getVersion() << "\n\n";
        std::cout << "Usage: " << argv[0]
                  << " <yolo_engine> <convlstm_engine> [video_dir]\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " segment.engine convlstm.engine .\n";
        return 1;
    }

    const char* yolo_engine = argv[1];
    const char* convlstm_engine = argv[2];
    std::string video_dir = (argc >= 4) ? argv[3] : ".";

    std::cout << "========================================\n";
    std::cout << "  SceneController Test v" << fire_detection::getVersion() << "\n";
    std::cout << "========================================\n\n";

    // Configuration
    fire_detection::SceneConfig config;
    config.detector_config.yolo_engine_path = yolo_engine;
    config.detector_config.convlstm_engine_path = convlstm_engine;
    config.detector_config.confidence_threshold = 0.5f;
    config.detector_config.convlstm_seq_len = 10;
    config.max_frame_gap_ms = 30000;
    config.min_sample_interval_ms = 100;
    config.max_queue_size = 30;

    std::cout << "Configuration:\n";
    std::cout << "  YOLO engine: " << config.detector_config.yolo_engine_path << "\n";
    std::cout << "  ConvLSTM engine: " << config.detector_config.convlstm_engine_path << "\n";
    std::cout << "  Video directory: " << video_dir << "\n";
    std::cout << "  Confidence: " << config.detector_config.confidence_threshold << "\n";
    std::cout << "  Sequence length: " << config.detector_config.convlstm_seq_len << "\n\n";

    // Initialize
    fire_detection::SceneController controller;
    if (!controller.initialize(config)) {
        std::cerr << "Error: Failed to initialize SceneController\n";
        return 1;
    }
    std::cout << "SceneController initialized successfully.\n";

    // Test suites
    std::cout << "\n[Test 1/6] Scene Management\n";
    std::cout << "----------------------------------------\n";
    testSceneManagement(controller);

    std::cout << "\n[Test 2/6] Frame Processing\n";
    std::cout << "----------------------------------------\n";
    testFrameProcessing(controller, video_dir);

    std::cout << "\n[Test 3/6] Multi-Scene Isolation\n";
    std::cout << "----------------------------------------\n";
    testMultiSceneIsolation(controller, video_dir);

    std::cout << "\n[Test 4/6] Reset Operations\n";
    std::cout << "----------------------------------------\n";
    testResetOperations(controller, video_dir);

    std::cout << "\n[Test 5/6] Memory Leak Detection\n";
    std::cout << "----------------------------------------\n";
    testMemoryLeaks(controller, video_dir);

    std::cout << "\n[Test 6/6] Precision/Accuracy + Output Video\n";
    std::cout << "----------------------------------------\n";
    testPrecisionAccuracy(controller, video_dir, buildGroundTruthTable());

    // Final summary
    printFinalSummary();

    return hasFailures() ? 1 : 0;
}
