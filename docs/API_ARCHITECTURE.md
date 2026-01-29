# Fire Detection API 技术架构文档

## 概述

Fire Detection API 是一个集成了深度学习的火灾检测库，提供从视频帧到火灾警报的完整处理流程。该库使用 YOLOv8-seg 进行实例分割检测，并通过 ConvLSTM 进行时序分析，以区分真实火灾与静态误报（如红色物体、灯光等）。

### 核心特性

- **双阶段检测**: YOLOv8-seg 空间检测 + ConvLSTM 时序分析
- **实时流式处理**: 单帧输入，内部自动管理滑动窗口
- **ROI 区域提取**: 仅关注火焰/烟雾区域，减少误报干扰
- **EMA 掩膜平滑**: 消除检测框抖动
- **双接口支持**: C++ 和 C 接口（便于 FFI 绑定）
- **PIMPL 模式**: ABI 稳定，实现细节隐藏

---

## 系统架构

### 模块层次结构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Public API Layer                            │
│   fire_detection_api.h - C++/C 统一接口                         │
│   FireDetector 类 / fire_detector_* C 函数                       │
├─────────────────────────────────────────────────────────────────┤
│                    Implementation Layer                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ SegmentDetector │  │   ROIPipeline   │  │ConvLSTMClassifier│ │
│  │ (YOLOv8-seg)    │  │  (ROI 区域提取) │  │ (时序分类)       │ │
│  │ detector/       │  │  roi/           │  │ temporal/        │ │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘  │
│           │                    │                    │           │
├───────────┴────────────────────┴────────────────────┴───────────┤
│                     Foundation Layer                             │
│  ┌───────────────────┐  ┌──────────────────────────────────────┐│
│  │YOLOv8PostProcessor│  │       TrtEngineMultiTs (cudatt)      ││
│  │ (NMS + 掩膜生成)  │  │       TensorRT 推理引擎封装          ││
│  └───────────────────┘  └──────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

### 目录结构

```
include/
├── fire_detection_api.h          # 公共 API（用户只需包含此文件）
├── detector/
│   ├── segment_detector.h        # YOLOv8-seg 检测器
│   └── yolov8_postprocess.h      # 后处理（NMS、掩膜）
├── roi/
│   └── roi_extractor.h           # ROI 区域提取与掩膜平滑
└── temporal/
    └── convlstm_classifier.h     # ConvLSTM 时序分类器

src/
├── api/
│   ├── fire_detection_api.cpp    # C++ API 实现（PIMPL）
│   └── fire_detection_api_c.cpp  # C 接口实现
├── detector/
│   ├── segment_detector.cpp
│   └── yolov8_postprocess.cpp
├── roi/
│   └── roi_extractor.cpp
├── temporal/
│   └── convlstm_classifier.cpp
└── demo/
    └── main.cpp                  # 示例程序
```

---

## 计算流程

### 整体数据流

```mermaid
flowchart TB
    subgraph Input["输入"]
        Frame["视频帧<br/>(cv::Mat BGR)"]
    end

    subgraph Stage1["阶段 1: 空间检测"]
        Preprocess1["预处理<br/>Letterbox + 归一化<br/>BGR→RGB, HWC→CHW"]
        YOLO["YOLOv8-seg 推理<br/>TensorRT Engine"]
        Postprocess["后处理<br/>解码 + NMS + 掩膜生成"]
    end

    subgraph Stage2["阶段 2: ROI 提取"]
        MaskEMA["掩膜 EMA 平滑<br/>α=0.2"]
        ROIExtract["ROI 区域提取<br/>非检测区域置黑"]
    end

    subgraph Stage3["阶段 3: 时序分类"]
        SlidingWindow["滑动窗口<br/>缓冲 seq_len 帧"]
        Preprocess2["预处理<br/>Letterbox + 归一化"]
        ConvLSTM["ConvLSTM 推理<br/>TensorRT Engine"]
        Softmax["Softmax + ArgMax"]
    end

    subgraph Output["输出"]
        Result["DetectionResult<br/>• 检测框列表<br/>• ROI 帧<br/>• 时序分类<br/>• 置信度"]
    end

    Frame --> Preprocess1
    Preprocess1 --> YOLO
    YOLO --> Postprocess
    Postprocess --> MaskEMA
    MaskEMA --> ROIExtract
    ROIExtract --> SlidingWindow
    SlidingWindow -->|"缓冲区满"| Preprocess2
    Preprocess2 --> ConvLSTM
    ConvLSTM --> Softmax

    Postprocess --> Result
    ROIExtract --> Result
    Softmax --> Result

    style Stage1 fill:#e1f5fe
    style Stage2 fill:#fff3e0
    style Stage3 fill:#f3e5f5
```

### 详细计算流程

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant API as FireDetector
    participant SD as SegmentDetector
    participant ROI as ROIPipeline
    participant LSTM as ConvLSTMClassifier
    participant TRT as TensorRT

    User->>API: processFrame(frame)

    Note over API: 检查初始化状态

    rect rgb(225, 245, 254)
        Note over SD,TRT: 阶段 1: YOLOv8-seg 检测
        API->>ROI: processBatch([frame])
        ROI->>SD: detect(frame)
        SD->>SD: preprocessFrame()<br/>640x640, RGB, [0,1], CHW
        SD->>TRT: infer(input_tensor)
        TRT-->>SD: output0 (1,38,8400)<br/>output1 (1,32,160,160)
        SD->>SD: postprocess<br/>decodeOutput + NMS
        SD->>SD: generateMask<br/>系数 × 原型
        SD-->>ROI: vector<ROIDetection>
    end

    rect rgb(255, 243, 224)
        Note over ROI: 阶段 2: ROI 提取
        ROI->>ROI: updateMaskEMA()<br/>mask = α*new + (1-α)*old
        ROI->>ROI: applyMaskToFrame()<br/>非 ROI 区域置黑
        ROI-->>API: BatchResult<br/>{roi_frame, detections}
    end

    rect rgb(243, 229, 245)
        Note over LSTM,TRT: 阶段 3: 时序分类
        API->>LSTM: addFrame(roi_frame)
        LSTM->>LSTM: preprocessFrame()<br/>640x640, RGB, [0,1]
        LSTM->>LSTM: 存入滑动窗口

        alt 缓冲区已满 (size >= seq_len)
            API->>LSTM: infer()
            LSTM->>LSTM: assembleInputTensor()<br/>(1, seq_len, 3, 640, 640)
            LSTM->>TRT: infer(input_tensor)
            TRT-->>LSTM: output (1, 3, 20, 20)
            LSTM->>LSTM: parseOutput()<br/>max pooling + softmax
            LSTM-->>API: classification, confidence
        end
    end

    API-->>User: DetectionResult
```

---

## 各阶段详解

### 阶段 1: YOLOv8-seg 空间检测

#### 输入预处理

```mermaid
flowchart LR
    subgraph Input["输入帧"]
        BGR["BGR 图像<br/>(H, W, 3)"]
    end

    subgraph Letterbox["Letterbox 缩放"]
        Scale["计算缩放比例<br/>scale = min(640/W, 640/H)"]
        Resize["双线性插值缩放"]
        Pad["边缘填充<br/>灰色 (114,114,114)"]
    end

    subgraph Convert["格式转换"]
        RGB["BGR → RGB"]
        Norm["归一化 [0,1]<br/>pixel / 255.0"]
        CHW["HWC → CHW<br/>转置维度"]
    end

    subgraph Output["输出张量"]
        Tensor["(1, 3, 640, 640)<br/>float32"]
    end

    BGR --> Scale --> Resize --> Pad --> RGB --> Norm --> CHW --> Tensor
```

#### YOLOv8-seg 输出解码

| 输出 | 形状 | 说明 |
|------|------|------|
| output0 | (1, 38, 8400) | 检测结果：4(bbox) + 3(类别) + 32(掩膜系数) × 8400 个锚点 |
| output1 | (1, 32, 160, 160) | 掩膜原型矩阵 |

```mermaid
flowchart TB
    subgraph Decode["解码 output0"]
        Anchors["遍历 8400 个锚点"]
        BBox["提取 bbox<br/>(cx, cy, w, h)"]
        Class["提取类别得分<br/>3 个类别"]
        Coeffs["提取掩膜系数<br/>32 维向量"]
    end

    subgraph Filter["过滤与 NMS"]
        ConfFilter["置信度过滤<br/>score > threshold"]
        NMS["非极大值抑制<br/>IoU 阈值"]
    end

    subgraph Mask["掩膜生成"]
        MatMul["系数 × 原型<br/>(32,) × (32, 160, 160)"]
        Sigmoid["Sigmoid 激活"]
        Crop["裁剪到 bbox 区域"]
    end

    subgraph Output["输出"]
        Dets["Detection 列表<br/>{bbox, mask, class_id}"]
    end

    Anchors --> BBox & Class & Coeffs
    BBox & Class --> ConfFilter
    ConfFilter --> NMS
    NMS --> MatMul
    Coeffs --> MatMul
    MatMul --> Sigmoid --> Crop --> Dets
```

### 阶段 2: ROI 区域提取

#### EMA 掩膜平滑

为了消除检测框的帧间抖动，使用指数移动平均（EMA）对掩膜进行时序平滑：

```
mask_ema = α × mask_new + (1 - α) × mask_old
```

默认 `α = 0.2`，表示新帧贡献 20%，历史帧贡献 80%。

#### ROI 帧生成

```mermaid
flowchart LR
    subgraph Input["输入"]
        Frame["原始帧"]
        Masks["检测掩膜"]
    end

    subgraph Process["处理"]
        Combine["合并所有掩膜<br/>fire + smoke"]
        EMA["EMA 平滑"]
        Binary["二值化<br/>threshold = 0.5"]
    end

    subgraph Apply["应用"]
        Multiply["ROI = 原帧 × 掩膜"]
        Black["非 ROI 区域 = 黑色"]
    end

    subgraph Output["输出"]
        ROIFrame["ROI 帧<br/>仅显示火焰/烟雾区域"]
    end

    Frame --> Multiply
    Masks --> Combine --> EMA --> Binary --> Multiply --> ROIFrame
```

### 阶段 3: ConvLSTM 时序分类

#### 滑动窗口机制

```mermaid
flowchart TB
    subgraph Buffer["滑动窗口缓冲区 (seq_len=10)"]
        direction LR
        F1["帧 1"] --> F2["帧 2"] --> F3["帧 3"] --> F4["..."] --> F10["帧 10"]
    end

    subgraph NewFrame["新帧到达"]
        New["帧 11"]
    end

    subgraph Updated["更新后"]
        direction LR
        F2_new["帧 2"] --> F3_new["帧 3"] --> F4_new["..."] --> F10_new["帧 10"] --> F11["帧 11"]
    end

    New --> Updated
    Buffer -.->|"移除最旧帧"| Updated

    style F1 fill:#ffcdd2
    style F11 fill:#c8e6c9
```

#### ConvLSTM 推理流程

```mermaid
flowchart TB
    subgraph Input["输入张量"]
        Tensor["(1, 10, 3, 640, 640)<br/>10 帧 RGB 图像序列"]
    end

    subgraph Model["ConvLSTM 模型"]
        Conv["卷积特征提取"]
        LSTM["LSTM 时序建模"]
        Head["分类头"]
    end

    subgraph Output["输出张量"]
        Heatmap["(1, 3, 20, 20)<br/>3 个类别的热力图"]
    end

    subgraph Postprocess["后处理"]
        MaxPool["每类取最大值<br/>20×20 → 1"]
        Softmax["Softmax 归一化"]
        ArgMax["取最大类别"]
    end

    subgraph Result["分类结果"]
        Class["STATIC / DYNAMIC / NEGATIVE"]
        Conf["置信度"]
    end

    Tensor --> Conv --> LSTM --> Head --> Heatmap
    Heatmap --> MaxPool --> Softmax --> ArgMax --> Class
    Softmax --> Conf
```

#### 分类类别说明

| 类别 | 枚举值 | 说明 | 颜色标记 |
|------|--------|------|----------|
| STATIC | 0 | 静态误报（图片、红色物体、灯光等） | 橙色 |
| DYNAMIC | 1 | 动态火焰（真实火灾，具有闪烁、扩散特征） | 红色 |
| NEGATIVE | 2 | 无检测（未检测到火焰/烟雾） | 绿色 |

---

## 数据结构

### 配置结构 (Config)

```cpp
struct Config {
    std::string yolo_engine_path;       // YOLOv8-seg 引擎路径（必需）
    std::string convlstm_engine_path;   // ConvLSTM 引擎路径（必需）
    float confidence_threshold = 0.5f;  // 检测置信度阈值
    float iou_threshold = 0.45f;        // NMS IoU 阈值
    int convlstm_seq_len = 10;          // 滑动窗口大小
    bool enable_roi_extraction = true;  // 启用 ROI 提取
    float ema_alpha = 0.2f;             // EMA 平滑系数
    float bbox_padding = 0.05f;         // 边界框填充比例
};
```

### 检测结果 (DetectionResult)

```cpp
struct DetectionResult {
    FrameResult frame;                  // 单帧检测结果
    bool temporal_valid = false;        // 时序结果是否有效
    TemporalClass temporal_class;       // STATIC / DYNAMIC / NEGATIVE
    float temporal_confidence;          // 时序分类置信度
    float class_scores[3];              // 各类别得分
};

struct FrameResult {
    bool has_fire;                      // 是否检测到火焰
    bool has_smoke;                     // 是否检测到烟雾
    std::vector<BoundingBox> detections; // 检测框列表
    cv::Mat roi_frame;                  // ROI 处理后的帧
};
```

---

## 性能参数

### 模型规格

| 模型 | 输入尺寸 | 输出尺寸 | 用途 |
|------|----------|----------|------|
| YOLOv8-seg | (1, 3, 640, 640) | (1, 38, 8400) + (1, 32, 160, 160) | 实例分割 |
| ConvLSTM | (1, 10, 3, 640, 640) | (1, 3, 20, 20) | 时序分类 |

### 检测类别

| 类别 ID | 名称 | 说明 |
|---------|------|------|
| 0 | FIRE | 火焰 |
| 1 | PERSON | 人（负样本，用于排除） |
| 2 | SMOKE | 烟雾 |

---

## 状态机

```mermaid
stateDiagram-v2
    [*] --> Created: 构造 FireDetector
    Created --> Initializing: initialize(config)

    Initializing --> Ready: 成功
    Initializing --> Error: 失败

    Ready --> Processing: processFrame()
    Processing --> Buffering: 缓冲区未满
    Processing --> Classifying: 缓冲区已满

    Buffering --> Ready: 返回结果<br/>(temporal_valid=false)
    Classifying --> Ready: 返回结果<br/>(temporal_valid=true)

    Ready --> Ready: reset()

    Error --> [*]: 销毁
    Ready --> [*]: 销毁
```

---

## 线程安全性

- **单实例单线程**: `FireDetector` 实例 **不是** 线程安全的
- **多实例**: 可以在不同线程中使用不同的 `FireDetector` 实例
- **GPU 资源**: TensorRT 上下文绑定到创建时的 CUDA 设备

推荐用法：

```cpp
// 正确：每个线程一个实例
void processThread(const std::string& video_path) {
    fire_detection::FireDetector detector;
    detector.initialize(config);
    // ... 处理视频
}

// 错误：多线程共享实例
fire_detection::FireDetector shared_detector;  // 危险！
```

---

## 参考资料

- [YOLOv8 文档](https://docs.ultralytics.com/)
- [TensorRT 开发者指南](https://docs.nvidia.com/deeplearning/tensorrt/)
- [ConvLSTM 论文](https://arxiv.org/abs/1506.04214)
