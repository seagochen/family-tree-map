# Fire Detection API 故障排查与注意事项

本文档详细说明使用 Fire Detection API 时可能遇到的问题、错误原因及解决方案。重点关注程序逻辑层面的问题（模型相关问题除外）。

---

## 目录

1. [初始化失败场景](#初始化失败场景)
2. [运行时失败场景](#运行时失败场景)
3. [结果异常场景](#结果异常场景)
4. [资源管理问题](#资源管理问题)
5. [常见错误代码](#常见错误代码)
6. [调试建议](#调试建议)

---

## 初始化失败场景

### 1. 配置参数无效

**症状**: `initialize()` 返回 `false`，错误信息："Invalid configuration"

**可能原因**:

| 参数 | 无效条件 | 说明 |
|------|----------|------|
| `yolo_engine_path` | 为空字符串 | 必须指定 YOLOv8-seg 引擎路径 |
| `convlstm_engine_path` | 为空字符串 | 必须指定 ConvLSTM 引擎路径 |
| `confidence_threshold` | ≤ 0 或 > 1 | 置信度阈值必须在 (0, 1] 范围内 |
| `iou_threshold` | ≤ 0 或 > 1 | IoU 阈值必须在 (0, 1] 范围内 |
| `convlstm_seq_len` | ≤ 0 | 序列长度必须为正整数 |
| `ema_alpha` | ≤ 0 或 > 1 | EMA 系数必须在 (0, 1] 范围内 |
| `bbox_padding` | < 0 或 > 1 | 填充比例必须在 [0, 1] 范围内 |

**解决方案**:

```cpp
fire_detection::Config config;
config.yolo_engine_path = "segment.engine";       // 不能为空
config.convlstm_engine_path = "convlstm.engine";  // 不能为空
config.confidence_threshold = 0.5f;  // (0, 1]
config.iou_threshold = 0.45f;        // (0, 1]
config.convlstm_seq_len = 10;        // > 0
config.ema_alpha = 0.2f;             // (0, 1]
config.bbox_padding = 0.05f;         // [0, 1]

// 初始化前检查
if (!config.isValid()) {
    std::cerr << "配置参数无效" << std::endl;
    // 逐项检查哪个参数有问题
}
```

### 2. 引擎文件路径问题

**症状**: `initialize()` 返回 `false`，错误信息："Failed to load YOLO/ConvLSTM engine"

**可能原因**:

- 路径不存在
- 路径为相对路径，但工作目录不正确
- 文件权限不足
- 文件损坏或格式不正确

**解决方案**:

```cpp
#include <filesystem>

// 检查文件是否存在
if (!std::filesystem::exists(config.yolo_engine_path)) {
    std::cerr << "YOLOv8 引擎文件不存在: " << config.yolo_engine_path << std::endl;
}

// 使用绝对路径
config.yolo_engine_path = "/absolute/path/to/segment.engine";

// 或获取绝对路径
config.yolo_engine_path = std::filesystem::absolute("segment.engine").string();
```

### 3. 重复初始化

**症状**: 第二次调用 `initialize()` 返回 `false`，错误信息："Already initialized"

**说明**: `FireDetector` 实例只能初始化一次。

**解决方案**:

```cpp
// 方法 1: 检查是否已初始化
if (!detector.isReady()) {
    detector.initialize(config);
}

// 方法 2: 需要重新初始化时，创建新实例
fire_detection::FireDetector new_detector;
new_detector.initialize(new_config);
```

### 4. TensorRT 执行上下文创建失败

**症状**: `initialize()` 返回 `false`，错误信息："Failed to create execution context"

**可能原因**:

- GPU 显存不足
- CUDA 设备不可用
- TensorRT 版本与引擎文件不兼容

**解决方案**:

```bash
# 检查 GPU 状态
nvidia-smi

# 检查 CUDA 版本
nvcc --version

# 检查 TensorRT 版本
dpkg -l | grep tensorrt
```

```cpp
// 在初始化前检查 CUDA 设备
int device_count = 0;
cudaGetDeviceCount(&device_count);
if (device_count == 0) {
    std::cerr << "没有可用的 CUDA 设备" << std::endl;
}
```

---

## 运行时失败场景

### 1. 检测器未初始化

**症状**: `processFrame()` 返回空结果，错误信息："Not initialized"

**代码位置**: [src/api/fire_detection_api.cpp:106](src/api/fire_detection_api.cpp#L106)

**解决方案**:

```cpp
// 必须先初始化
fire_detection::FireDetector detector;
if (!detector.initialize(config)) {
    std::cerr << "初始化失败" << std::endl;
    return;
}

// 检查就绪状态
if (!detector.isReady()) {
    std::cerr << "检测器未就绪" << std::endl;
    return;
}

auto result = detector.processFrame(frame);
```

### 2. 输入帧为空

**症状**: `processFrame()` 返回空结果，错误信息："Empty frame"

**代码位置**: [src/api/fire_detection_api.cpp:110](src/api/fire_detection_api.cpp#L110)

**可能原因**:

- 视频解码失败
- 摄像头连接断开
- 图像加载失败

**解决方案**:

```cpp
cv::Mat frame;

// 检查读取是否成功
if (!cap.read(frame)) {
    std::cerr << "读取帧失败，视频可能已结束" << std::endl;
    break;
}

// 检查帧是否为空
if (frame.empty()) {
    std::cerr << "帧为空" << std::endl;
    continue;
}

auto result = detector.processFrame(frame);
```

### 3. 图像通道数错误

**症状**: `processFrameRaw()` 返回空结果，错误信息："Only 3-channel images are supported"

**代码位置**: [src/api/fire_detection_api.cpp:182](src/api/fire_detection_api.cpp#L182)

**说明**: API 只支持 3 通道 BGR 图像。

**解决方案**:

```cpp
// 检查通道数
if (frame.channels() != 3) {
    // 灰度图转 BGR
    if (frame.channels() == 1) {
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);
    }
    // BGRA 转 BGR
    else if (frame.channels() == 4) {
        cv::cvtColor(frame, frame, cv::COLOR_BGRA2BGR);
    }
}

auto result = detector.processFrame(frame);
```

### 4. C 接口空指针

**症状**: C 接口函数返回 0（失败）

**可能原因**:

- `handle` 为 NULL
- `data` 为 NULL
- `result` 为 NULL
- `config` 为 NULL

**解决方案**:

```c
FireDetectorHandle detector = fire_detector_create();
if (!detector) {
    fprintf(stderr, "创建检测器失败\n");
    return;
}

FireDetectionResultC result;
unsigned char* frame_data = get_frame_data();  // 确保非空

if (!frame_data) {
    fprintf(stderr, "帧数据为空\n");
    return;
}

if (!fire_detector_process_frame(detector, frame_data, width, height, 3, &result)) {
    fprintf(stderr, "处理帧失败\n");
}
```

---

## 结果异常场景

### 1. temporal_valid 始终为 false

**症状**: 处理多帧后，`result.temporal_valid` 仍为 `false`

**原因**: 滑动窗口缓冲区未满。需要处理至少 `seq_len` 帧后才能进行时序分类。

**代码位置**: [src/api/fire_detection_api.cpp:162](src/api/fire_detection_api.cpp#L162)

**验证方法**:

```cpp
auto result = detector.processFrame(frame);

// 检查缓冲区状态
std::cout << "Buffer: " << detector.getBufferSize()
          << "/" << detector.getBufferCapacity() << std::endl;

// 只有缓冲区满时 temporal_valid 才为 true
if (detector.isBufferFull()) {
    // 此时 result.temporal_valid 应为 true
    assert(result.temporal_valid);
}
```

**典型时间线**:

```
帧 1:  Buffer 1/10,  temporal_valid = false
帧 2:  Buffer 2/10,  temporal_valid = false
...
帧 9:  Buffer 9/10,  temporal_valid = false
帧 10: Buffer 10/10, temporal_valid = true  ← 首次有效
帧 11: Buffer 10/10, temporal_valid = true  ← 滑动窗口
```

### 2. 切换视频后时序结果异常

**症状**: 切换视频源后，前几帧的时序分类结果不准确

**原因**: 滑动窗口中仍保留上一个视频的帧数据

**解决方案**:

```cpp
// 处理完一个视频后
detector.reset();  // 清空滑动窗口

// 再处理下一个视频
processVideo(detector, "next_video.mp4");
```

**适用场景**:

- 切换摄像头
- 切换视频文件
- 场景跳转（如广告插入）
- 检测暂停后恢复

### 3. ROI 帧显示异常

**症状**: `result.frame.roi_frame` 全黑或显示不正确

**可能原因**:

| 现象 | 原因 |
|------|------|
| 全黑 | 没有检测到火焰/烟雾，ROI 区域为空 |
| 与原帧相同 | `enable_roi_extraction = false` |
| 部分显示 | 只显示检测到的 ROI 区域，其余置黑 |

**验证方法**:

```cpp
auto result = detector.processFrame(frame);

// 检查是否有检测
if (!result.frame.hasDetection()) {
    // 无检测时，ROI 帧应该全黑（或原帧，取决于配置）
}

// 检查 ROI 帧是否为空
if (result.frame.roi_frame.empty()) {
    std::cerr << "ROI 帧为空" << std::endl;
}

// 检查 ROI 帧尺寸
std::cout << "ROI 帧尺寸: " << result.frame.roi_frame.cols
          << "x" << result.frame.roi_frame.rows << std::endl;
```

### 4. 检测框坐标越界

**症状**: `BoundingBox` 的坐标超出图像范围

**说明**: 理论上不应发生，后处理已进行 clamp 操作

**代码位置**: [src/detector/yolov8_postprocess.cpp:62-65](src/detector/yolov8_postprocess.cpp#L62-L65)

**防御性编程**:

```cpp
for (const auto& det : result.frame.detections) {
    // 安全 clamp
    int x1 = std::max(0, std::min(static_cast<int>(det.x1), frame.cols - 1));
    int y1 = std::max(0, std::min(static_cast<int>(det.y1), frame.rows - 1));
    int x2 = std::max(0, std::min(static_cast<int>(det.x2), frame.cols));
    int y2 = std::max(0, std::min(static_cast<int>(det.y2), frame.rows));

    cv::Rect safe_rect(x1, y1, x2 - x1, y2 - y1);
    cv::rectangle(frame, safe_rect, cv::Scalar(0, 0, 255), 2);
}
```

---

## 资源管理问题

### 1. 内存泄漏（C 接口）

**症状**: 长时间运行后内存持续增长

**原因**: 未调用 `fire_detector_destroy()`

**解决方案**:

```c
FireDetectorHandle detector = fire_detector_create();

// 使用...

// 务必销毁
fire_detector_destroy(detector);
detector = NULL;  // 避免悬空指针
```

### 2. GPU 内存未释放

**症状**: 程序退出后 `nvidia-smi` 仍显示 GPU 内存占用

**原因**: `FireDetector` 未正确析构

**解决方案**:

```cpp
// C++: 确保 FireDetector 在作用域结束时析构
{
    fire_detection::FireDetector detector;
    detector.initialize(config);
    // 处理...
}  // 自动析构

// 或使用智能指针
auto detector = std::make_unique<fire_detection::FireDetector>();
detector->initialize(config);
// 处理...
detector.reset();  // 显式释放
```

### 3. 多线程资源竞争

**症状**: 崩溃、结果不一致、数据损坏

**原因**: 多线程共享同一个 `FireDetector` 实例

**解决方案**:

```cpp
// 错误示例
fire_detection::FireDetector shared_detector;  // 共享实例
// 多线程访问 shared_detector ← 危险！

// 正确示例：每个线程独立实例
void workerThread(const std::string& video) {
    fire_detection::FireDetector detector;  // 线程局部
    detector.initialize(config);

    cv::VideoCapture cap(video);
    cv::Mat frame;
    while (cap.read(frame)) {
        auto result = detector.processFrame(frame);
        // 处理...
    }
}
```

### 4. ROI 帧数据复制

**症状**: 使用 `result.frame.roi_frame` 后原数据被覆盖

**说明**: `roi_frame` 是 `clone()` 后的副本，可安全保存

**代码位置**: [src/api/fire_detection_api.cpp:134](src/api/fire_detection_api.cpp#L134)

```cpp
// 安全：roi_frame 是独立副本
std::vector<cv::Mat> saved_frames;
while (cap.read(frame)) {
    auto result = detector.processFrame(frame);
    saved_frames.push_back(result.frame.roi_frame);  // 安全保存
}
```

---

## 常见错误代码

### C++ 接口

| 情况 | 返回值 | 说明 |
|------|--------|------|
| 初始化成功 | `true` | - |
| 初始化失败 | `false` | 检查错误输出 |
| 处理成功 | 有效的 `DetectionResult` | - |
| 处理失败 | 空的 `DetectionResult` | `temporal_valid = false` |

### C 接口

| 函数 | 成功 | 失败 |
|------|------|------|
| `fire_detector_create()` | 有效句柄 | `NULL` |
| `fire_detector_initialize()` | `1` | `0` |
| `fire_detector_process_frame()` | `1` | `0` |
| `fire_detector_get_roi_frame()` | `1` | `0` |
| `fire_detector_is_ready()` | `1` | `0` |
| `fire_detector_is_buffer_full()` | `1` | `0` |

---

## 调试建议

### 1. 启用详细日志

当前实现使用 `std::cerr` 输出错误信息。在调试时注意观察标准错误输出：

```bash
./my_app 2>&1 | tee debug.log
```

### 2. 逐步验证

```cpp
// 1. 验证配置
std::cout << "配置:" << std::endl;
std::cout << "  YOLO: " << config.yolo_engine_path << std::endl;
std::cout << "  ConvLSTM: " << config.convlstm_engine_path << std::endl;
std::cout << "  有效: " << (config.isValid() ? "是" : "否") << std::endl;

// 2. 验证初始化
if (!detector.initialize(config)) {
    std::cerr << "初始化失败" << std::endl;
    return;
}
std::cout << "初始化成功" << std::endl;

// 3. 验证帧处理
cv::Mat frame = cv::imread("test.jpg");
std::cout << "帧尺寸: " << frame.cols << "x" << frame.rows << std::endl;
std::cout << "帧类型: " << frame.type() << std::endl;
std::cout << "帧通道: " << frame.channels() << std::endl;

// 4. 验证结果
auto result = detector.processFrame(frame);
std::cout << "结果:" << std::endl;
std::cout << "  火焰: " << result.frame.has_fire << std::endl;
std::cout << "  烟雾: " << result.frame.has_smoke << std::endl;
std::cout << "  检测数: " << result.frame.detections.size() << std::endl;
std::cout << "  时序有效: " << result.temporal_valid << std::endl;
std::cout << "  缓冲区: " << detector.getBufferSize()
          << "/" << detector.getBufferCapacity() << std::endl;
```

### 3. 最小复现

创建最小测试程序隔离问题：

```cpp
#include "fire_detection_api.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    fire_detection::Config config;
    config.yolo_engine_path = "segment.engine";
    config.convlstm_engine_path = "convlstm.engine";

    fire_detection::FireDetector detector;
    if (!detector.initialize(config)) {
        std::cerr << "初始化失败" << std::endl;
        return 1;
    }

    // 创建测试帧
    cv::Mat test_frame(1080, 1920, CV_8UC3, cv::Scalar(0, 0, 0));

    for (int i = 0; i < 15; i++) {
        auto result = detector.processFrame(test_frame);
        std::cout << "帧 " << i << ": "
                  << "buffer=" << detector.getBufferSize()
                  << ", valid=" << result.temporal_valid
                  << std::endl;
    }

    return 0;
}
```

### 4. 性能分析

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
auto result = detector.processFrame(frame);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "处理耗时: " << duration.count() << " ms" << std::endl;
```

---

## 问题检查清单

在报告问题前，请检查以下项目：

- [ ] 引擎文件路径是否正确且文件存在
- [ ] 配置参数是否在有效范围内
- [ ] 输入帧是否为有效的 3 通道 BGR 图像
- [ ] 是否正确调用了 `initialize()`
- [ ] 是否处理了足够多的帧（至少 `seq_len` 帧）
- [ ] 切换视频源时是否调用了 `reset()`
- [ ] C 接口是否正确调用了 `fire_detector_destroy()`
- [ ] GPU 是否有足够的显存
- [ ] TensorRT 版本是否与引擎文件兼容
