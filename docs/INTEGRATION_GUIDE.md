# Fire Detection API 集成指南

本文档介绍如何将 Fire Detection API 集成到第三方 C/C++ 项目中。

---

## 快速开始

### 依赖项

在使用本库之前，请确保系统已安装以下依赖：

| 依赖 | 最低版本 | 说明 |
|------|----------|------|
| CUDA | 11.x | GPU 计算支持 |
| TensorRT | 8.x | 深度学习推理引擎 |
| OpenCV | 4.x | 图像处理 |
| cudatt | - | CUDA 张量运算与 TensorRT 引擎封装库 |

### 库文件

编译后会生成以下库文件：

```
lib/
├── libfire_detection.a      # 静态库
└── libfire_detection.so     # 共享库 (动态库)
    └── libfire_detection.so.1
    └── libfire_detection.so.1.0.0

include/
└── fire_detection_api.h     # 公共头文件（仅需此文件）
```

### 模型文件

需要准备两个 TensorRT 引擎文件：

| 文件 | 说明 | 输入尺寸 |
|------|------|----------|
| `segment.engine` | YOLOv8-seg 实例分割模型 | (1, 3, 640, 640) |
| `convlstm.engine` | ConvLSTM 时序分类模型 | (1, 10, 3, 640, 640) |

---

## C++ 集成

### 头文件包含

```cpp
#include "fire_detection_api.h"
```

### 基础用法

```cpp
#include <iostream>
#include <opencv2/opencv.hpp>
#include "fire_detection_api.h"

int main() {
    // 1. 配置检测器
    fire_detection::Config config;
    config.yolo_engine_path = "/path/to/segment.engine";
    config.convlstm_engine_path = "/path/to/convlstm.engine";
    config.confidence_threshold = 0.5f;
    config.iou_threshold = 0.45f;
    config.convlstm_seq_len = 10;

    // 2. 创建并初始化检测器
    fire_detection::FireDetector detector;
    if (!detector.initialize(config)) {
        std::cerr << "初始化失败" << std::endl;
        return 1;
    }

    // 3. 打开视频源
    cv::VideoCapture cap("video.mp4");
    // 或使用摄像头: cv::VideoCapture cap(0);

    cv::Mat frame;
    while (cap.read(frame)) {
        // 4. 处理每一帧
        auto result = detector.processFrame(frame);

        // 5. 检查单帧检测结果（不依赖时序分析，立即可用）
        if (result.hasFire()) {
            std::cout << "检测到火焰！" << std::endl;
            // 获取火焰检测框
            auto fire_boxes = result.getFireBoxes();
            for (const auto& box : fire_boxes) {
                std::cout << "  位置: (" << box.x1 << ", " << box.y1 << ") - ("
                          << box.x2 << ", " << box.y2 << "), 置信度: "
                          << box.confidence << std::endl;
            }
        }
        if (result.hasSmoke()) {
            std::cout << "检测到烟雾！" << std::endl;
            // 获取烟雾检测框（可指定置信度阈值）
            auto smoke_boxes = result.getSmokeBoxes(0.6f);
            for (const auto& box : smoke_boxes) {
                std::cout << "  位置: (" << box.x1 << ", " << box.y1 << ") - ("
                          << box.x2 << ", " << box.y2 << ")" << std::endl;
            }
        }

        // 6. 检查时序分类结果（缓冲区满后有效）
        if (result.temporal_valid) {
            std::cout << "时序分类: "
                      << fire_detection::temporalClassToString(result.temporal_class)
                      << " (置信度: " << result.temporal_confidence * 100 << "%)"
                      << std::endl;

            // 7. 判断是否为真实火灾
            if (result.isFireAlert()) {
                std::cout << "!!! 火灾警报 !!!" << std::endl;
                // 触发报警逻辑...
            }
        }

        // 8. 可选：显示 ROI 帧
        cv::imshow("ROI", result.frame.roi_frame);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
```

### 完整示例：带状态显示

```cpp
#include <iostream>
#include <iomanip>
#include <opencv2/opencv.hpp>
#include "fire_detection_api.h"

void drawInfo(cv::Mat& frame, const fire_detection::DetectionResult& result,
              const fire_detection::FireDetector& detector) {
    int y = 30;
    const int line_height = 30;

    // 缓冲区状态
    std::string buffer_info = "Buffer: " +
        std::to_string(detector.getBufferSize()) + "/" +
        std::to_string(detector.getBufferCapacity());
    cv::putText(frame, buffer_info, cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(255, 255, 0), 2);
    y += line_height;

    // 检测状态
    std::string detect_status = "Detection: ";
    if (result.frame.has_fire && result.frame.has_smoke) {
        detect_status += "FIRE + SMOKE";
    } else if (result.frame.has_fire) {
        detect_status += "FIRE";
    } else if (result.frame.has_smoke) {
        detect_status += "SMOKE";
    } else {
        detect_status += "None";
    }
    cv::Scalar detect_color = (result.hasFire() || result.hasSmoke())
        ? cv::Scalar(0, 0, 255) : cv::Scalar(0, 255, 0);
    cv::putText(frame, detect_status, cv::Point(10, y),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, detect_color, 2);
    y += line_height;

    // 时序分类
    if (result.temporal_valid) {
        std::ostringstream oss;
        oss << "Temporal: "
            << fire_detection::temporalClassToString(result.temporal_class)
            << " (" << std::fixed << std::setprecision(1)
            << result.temporal_confidence * 100 << "%)";
        cv::Scalar cls_color = fire_detection::temporalClassToColor(result.temporal_class);
        cv::putText(frame, oss.str(), cv::Point(10, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, cls_color, 2);
        y += line_height;

        // 火灾警报
        if (result.isFireAlert()) {
            cv::putText(frame, "!!! FIRE ALERT !!!",
                        cv::Point(frame.cols / 2 - 150, 50),
                        cv::FONT_HERSHEY_SIMPLEX, 1.2, cv::Scalar(0, 0, 255), 3);
        }
    }

    // 绘制检测框
    for (const auto& det : result.frame.detections) {
        cv::Rect rect(
            static_cast<int>(det.x1),
            static_cast<int>(det.y1),
            static_cast<int>(det.width()),
            static_cast<int>(det.height())
        );
        cv::Scalar color = (det.class_id == fire_detection::DetectionClass::FIRE)
            ? cv::Scalar(0, 0, 255) : cv::Scalar(128, 128, 128);
        cv::rectangle(frame, rect, color, 2);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <yolo_engine> <convlstm_engine> <video>" << std::endl;
        return 1;
    }

    // 配置
    fire_detection::Config config;
    config.yolo_engine_path = argv[1];
    config.convlstm_engine_path = argv[2];

    // 初始化
    fire_detection::FireDetector detector;
    if (!detector.initialize(config)) {
        return 1;
    }

    // 处理视频
    cv::VideoCapture cap(argv[3]);
    cv::Mat frame;

    while (cap.read(frame)) {
        auto result = detector.processFrame(frame);

        cv::Mat display = result.frame.roi_frame.clone();
        drawInfo(display, result, detector);

        cv::imshow("Fire Detection", display);
        if (cv::waitKey(1) == 'q') break;
    }

    return 0;
}
```

### 切换视频源时重置状态

当需要切换视频源或重新开始检测时，务必调用 `reset()` 清空滑动窗口：

```cpp
// 处理第一个视频
processVideo(detector, "video1.mp4");

// 切换视频前重置状态
detector.reset();

// 处理第二个视频
processVideo(detector, "video2.mp4");
```

### CMake 集成

```cmake
cmake_minimum_required(VERSION 3.18)
project(my_fire_app)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# Fire Detection 库路径
set(FIRE_DETECTION_ROOT "/path/to/fire_detection")
set(FIRE_DETECTION_INCLUDE "${FIRE_DETECTION_ROOT}/include")
set(FIRE_DETECTION_LIB_DIR "${FIRE_DETECTION_ROOT}/lib")

# 查找库文件
find_library(FIRE_DETECTION_LIB fire_detection
    HINTS ${FIRE_DETECTION_LIB_DIR})

# 包含目录
include_directories(
    ${FIRE_DETECTION_INCLUDE}
    ${OpenCV_INCLUDE_DIRS}
)

# 可执行文件
add_executable(my_fire_app main.cpp)

# 链接库
target_link_libraries(my_fire_app
    ${FIRE_DETECTION_LIB}
    ${OpenCV_LIBS}
)
```

---

## C 语言集成

### 头文件包含

```c
#include "fire_detection_api.h"
```

### 基础用法

```c
#include <stdio.h>
#include <stdlib.h>
#include "fire_detection_api.h"

int main() {
    // 1. 创建检测器
    FireDetectorHandle detector = fire_detector_create();
    if (!detector) {
        fprintf(stderr, "创建检测器失败\n");
        return 1;
    }

    // 2. 配置
    FireDetectorConfig config = {
        .yolo_engine_path = "/path/to/segment.engine",
        .convlstm_engine_path = "/path/to/convlstm.engine",
        .confidence_threshold = 0.5f,
        .iou_threshold = 0.45f,
        .convlstm_seq_len = 10,
        .enable_roi_extraction = 1,
        .ema_alpha = 0.2f,
        .bbox_padding = 0.05f
    };

    // 3. 初始化
    if (!fire_detector_initialize(detector, &config)) {
        fprintf(stderr, "初始化失败\n");
        fire_detector_destroy(detector);
        return 1;
    }

    printf("初始化成功，版本: %s\n", fire_detector_get_version());

    // 4. 处理帧（假设已有图像数据）
    unsigned char* frame_data = /* 从视频解码器获取 BGR 数据 */;
    int width = 1920;
    int height = 1080;
    int channels = 3;

    FireDetectionResultC result;

    // 处理多帧
    for (int i = 0; i < frame_count; i++) {
        if (fire_detector_process_frame(detector, frame_data, width, height, channels, &result)) {
            // 检查单帧检测
            if (result.has_fire) {
                printf("帧 %d: 检测到火焰\n", i);
            }
            if (result.has_smoke) {
                printf("帧 %d: 检测到烟雾\n", i);
            }

            // 检查时序分类
            if (result.temporal_valid) {
                const char* class_names[] = {"STATIC", "DYNAMIC", "NEGATIVE"};
                printf("帧 %d: 时序分类 = %s (%.1f%%)\n",
                       i, class_names[result.temporal_class],
                       result.temporal_confidence * 100);

                // 真实火灾警报
                if (result.temporal_class == 1) {  // DYNAMIC
                    printf("!!! 火灾警报 !!!\n");
                }
            }
        }

        // 获取下一帧数据...
    }

    // 5. 清理
    fire_detector_destroy(detector);

    return 0;
}
```

### 获取 ROI 帧数据

```c
// 首先获取 ROI 帧尺寸
int roi_width = 0, roi_height = 0;
if (fire_detector_get_roi_frame(detector, NULL, &roi_width, &roi_height)) {
    // 分配缓冲区
    size_t buffer_size = roi_width * roi_height * 3;
    unsigned char* roi_data = (unsigned char*)malloc(buffer_size);

    // 获取 ROI 帧数据
    if (fire_detector_get_roi_frame(detector, roi_data, &roi_width, &roi_height)) {
        // 使用 roi_data...
        // 数据格式: BGR, HWC, uint8
    }

    free(roi_data);
}
```

### C API 完整参考

```c
// 创建/销毁
FireDetectorHandle fire_detector_create(void);
void fire_detector_destroy(FireDetectorHandle handle);

// 初始化
int fire_detector_initialize(FireDetectorHandle handle, const FireDetectorConfig* config);
int fire_detector_is_ready(FireDetectorHandle handle);

// 处理
int fire_detector_process_frame(FireDetectorHandle handle,
                                const unsigned char* data,
                                int width, int height, int channels,
                                FireDetectionResultC* result);

// ROI 帧
int fire_detector_get_roi_frame(FireDetectorHandle handle,
                                unsigned char* output_data,
                                int* width, int* height);

// 状态
void fire_detector_reset(FireDetectorHandle handle);
int fire_detector_get_buffer_size(FireDetectorHandle handle);
int fire_detector_is_buffer_full(FireDetectorHandle handle);

// 结果查询辅助函数
int fire_detection_result_get_fire_boxes(const FireDetectionResultC* result,
                                          float confidence_threshold,
                                          BoundingBoxC* output_boxes,
                                          int max_boxes);
int fire_detection_result_get_smoke_boxes(const FireDetectionResultC* result,
                                           float confidence_threshold,
                                           BoundingBoxC* output_boxes,
                                           int max_boxes);

// 版本
const char* fire_detector_get_version(void);
```

### 获取特定类别检测框

```c
FireDetectionResultC result;
if (fire_detector_process_frame(detector, frame_data, width, height, 3, &result)) {
    // 获取火焰检测框
    if (result.has_fire) {
        BoundingBoxC fire_boxes[16];
        int fire_count = fire_detection_result_get_fire_boxes(&result, 0.5f, fire_boxes, 16);
        for (int i = 0; i < fire_count; i++) {
            printf("火焰 %d: (%.0f, %.0f) - (%.0f, %.0f), 置信度: %.2f\n",
                   i, fire_boxes[i].x1, fire_boxes[i].y1,
                   fire_boxes[i].x2, fire_boxes[i].y2,
                   fire_boxes[i].confidence);
        }
    }

    // 获取烟雾检测框
    if (result.has_smoke) {
        BoundingBoxC smoke_boxes[16];
        int smoke_count = fire_detection_result_get_smoke_boxes(&result, 0.5f, smoke_boxes, 16);
        for (int i = 0; i < smoke_count; i++) {
            printf("烟雾 %d: (%.0f, %.0f) - (%.0f, %.0f)\n",
                   i, smoke_boxes[i].x1, smoke_boxes[i].y1,
                   smoke_boxes[i].x2, smoke_boxes[i].y2);
        }
    }
}
```

---

## FFI 绑定（其他语言）

### Python (ctypes)

```python
import ctypes
from ctypes import c_void_p, c_char_p, c_int, c_float, POINTER, Structure

# 加载库
lib = ctypes.CDLL("libfire_detection.so")

# 定义结构体
class FireDetectorConfig(Structure):
    _fields_ = [
        ("yolo_engine_path", c_char_p),
        ("convlstm_engine_path", c_char_p),
        ("confidence_threshold", c_float),
        ("iou_threshold", c_float),
        ("convlstm_seq_len", c_int),
        ("enable_roi_extraction", c_int),
        ("ema_alpha", c_float),
        ("bbox_padding", c_float),
    ]

class FireDetectionResultC(Structure):
    _fields_ = [
        ("has_fire", c_int),
        ("has_smoke", c_int),
        ("temporal_valid", c_int),
        ("temporal_class", c_int),
        ("temporal_confidence", c_float),
        ("class_scores", c_float * 3),
    ]

# 设置函数签名
lib.fire_detector_create.restype = c_void_p
lib.fire_detector_destroy.argtypes = [c_void_p]
lib.fire_detector_initialize.argtypes = [c_void_p, POINTER(FireDetectorConfig)]
lib.fire_detector_initialize.restype = c_int
lib.fire_detector_process_frame.argtypes = [
    c_void_p, POINTER(ctypes.c_ubyte), c_int, c_int, c_int,
    POINTER(FireDetectionResultC)
]
lib.fire_detector_process_frame.restype = c_int

# 使用示例
detector = lib.fire_detector_create()

config = FireDetectorConfig()
config.yolo_engine_path = b"/path/to/segment.engine"
config.convlstm_engine_path = b"/path/to/convlstm.engine"
config.confidence_threshold = 0.5
config.iou_threshold = 0.45
config.convlstm_seq_len = 10
config.enable_roi_extraction = 1
config.ema_alpha = 0.2
config.bbox_padding = 0.05

if lib.fire_detector_initialize(detector, ctypes.byref(config)):
    print("初始化成功")

    # 处理帧
    import numpy as np
    import cv2

    frame = cv2.imread("test.jpg")
    frame_data = frame.ctypes.data_as(POINTER(ctypes.c_ubyte))

    result = FireDetectionResultC()
    if lib.fire_detector_process_frame(detector, frame_data,
                                        frame.shape[1], frame.shape[0], 3,
                                        ctypes.byref(result)):
        print(f"Fire: {result.has_fire}, Smoke: {result.has_smoke}")
        if result.temporal_valid:
            classes = ["STATIC", "DYNAMIC", "NEGATIVE"]
            print(f"Temporal: {classes[result.temporal_class]}")

lib.fire_detector_destroy(detector)
```

### Rust (bindgen)

```rust
// 需要先使用 bindgen 生成绑定
// build.rs:
// bindgen::Builder::default()
//     .header("fire_detection_api.h")
//     .generate()

use std::ffi::CString;
use std::ptr;

fn main() {
    unsafe {
        let detector = fire_detector_create();
        if detector.is_null() {
            panic!("Failed to create detector");
        }

        let yolo_path = CString::new("/path/to/segment.engine").unwrap();
        let convlstm_path = CString::new("/path/to/convlstm.engine").unwrap();

        let config = FireDetectorConfig {
            yolo_engine_path: yolo_path.as_ptr(),
            convlstm_engine_path: convlstm_path.as_ptr(),
            confidence_threshold: 0.5,
            iou_threshold: 0.45,
            convlstm_seq_len: 10,
            enable_roi_extraction: 1,
            ema_alpha: 0.2,
            bbox_padding: 0.05,
        };

        if fire_detector_initialize(detector, &config) != 0 {
            println!("Initialized successfully");

            // 处理帧...

        }

        fire_detector_destroy(detector);
    }
}
```

---

## 编译选项

### 链接静态库

```bash
g++ -std=c++17 main.cpp -o app \
    -I/path/to/include \
    -L/path/to/lib \
    -lfire_detection \
    $(pkg-config --cflags --libs opencv4) \
    -lcudart -lnvinfer
```

### 链接动态库

```bash
g++ -std=c++17 main.cpp -o app \
    -I/path/to/include \
    -L/path/to/lib \
    -lfire_detection \
    $(pkg-config --cflags --libs opencv4) \
    -Wl,-rpath,/path/to/lib
```

### 运行时库路径

```bash
# 方法 1: 环境变量
export LD_LIBRARY_PATH=/path/to/lib:$LD_LIBRARY_PATH
./app

# 方法 2: ldconfig
sudo echo "/path/to/lib" > /etc/ld.so.conf.d/fire_detection.conf
sudo ldconfig

# 方法 3: 编译时指定 rpath
g++ ... -Wl,-rpath,/path/to/lib
```

---

## 最佳实践

### 1. 初始化时机

在程序启动时尽早初始化检测器，因为加载 TensorRT 引擎需要一定时间：

```cpp
// 程序启动时
fire_detection::FireDetector detector;
detector.initialize(config);  // 可能需要几秒钟

// 后续使用时已就绪
while (running) {
    auto result = detector.processFrame(frame);
}
```

### 2. 错误处理

```cpp
fire_detection::FireDetector detector;

// 检查配置有效性
if (!config.isValid()) {
    std::cerr << "配置无效" << std::endl;
    return 1;
}

// 检查初始化结果
if (!detector.initialize(config)) {
    std::cerr << "初始化失败，请检查引擎文件路径" << std::endl;
    return 1;
}

// 检查帧有效性
if (frame.empty()) {
    std::cerr << "帧为空" << std::endl;
    continue;
}

// 处理结果
auto result = detector.processFrame(frame);
if (!result.temporal_valid) {
    // 缓冲区未满，等待更多帧
}
```

### 3. 性能优化

```cpp
// 避免频繁 clone
const cv::Mat& roi_frame = result.frame.roi_frame;  // 使用引用

// 按需处理时序结果
if (detector.isBufferFull()) {
    // 只在缓冲区满时检查时序结果
    if (result.isFireAlert()) {
        handleAlert();
    }
}
```

### 4. 资源管理

```cpp
// RAII 风格：FireDetector 析构时自动释放资源
{
    fire_detection::FireDetector detector;
    detector.initialize(config);
    // 使用 detector...
}  // 自动清理

// C 接口：手动管理
FireDetectorHandle detector = fire_detector_create();
// 使用...
fire_detector_destroy(detector);  // 必须调用
```
