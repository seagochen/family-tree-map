# 火焰检测 API 使用指南

## 完整调用流程

```python
import cv2
import torch
from ultralytics import YOLO

from detect import FireDetectionPipeline
from convlstm.models.temporal_classifier import create_model

# ============ 1. 加载模型 ============

# YOLO 分割模型（用于检测火焰/烟雾区域）
yolo_model = YOLO("path/to/yolo_seg.pt")

# ConvLSTM 时序分类模型
classifier = create_model(
    checkpoint_path="path/to/convlstm.pt",  # 预训练权重路径
    num_classes=3                            # 类别数: 0=static, 1=dynamic, 2=negative
)

# ============ 2. 创建 Pipeline ============

pipeline = FireDetectionPipeline(
    yolo_model=yolo_model,
    classifier=classifier,
    device='cuda',                # 'cuda' 或 'cpu'
    confidence_threshold=0.5,     # YOLO 检测置信度
    use_ema=True,                 # 是否对 fire mask 使用 EMA 平滑
    ema_alpha=0.3,                # EMA 平滑因子
    bbox_padding=0.1,             # smoke bbox 内边距比例
)

# ============ 3. 准备输入数据 ============

# 读取连续帧（例如 10 帧）
frames = []
cap = cv2.VideoCapture("path/to/video.mp4")
for _ in range(10):
    ret, frame = cap.read()
    if ret:
        frames.append(frame)  # BGR 格式，任意尺寸
cap.release()

# ============ 4. 执行检测 ============

result = pipeline.predict(
    frames=frames,
    detect_mode='all'  # 'all' | 'fire' | 'smoke'
)

# ============ 5. 获取结果 ============

# 预测类别
print(f"预测类别: {result.predicted_class}")
print(f"类别名称: {pipeline.get_class_name(result.predicted_class)}")

# 各类别概率
print(f"类别概率: {result.class_probs}")  # numpy array (num_classes,)

# 分类热力图
print(f"热力图形状: {result.heatmap.shape}")  # (num_classes, 20, 20)

# 是否检测到火焰/烟雾
print(f"检测到目标: {result.has_detection}")

# ROI 提取详情
print(f"检测到火焰的帧数: {result.roi_result.frames_with_fire}")
print(f"检测到烟雾的帧数: {result.roi_result.frames_with_smoke}")
print(f"烟雾并集 bbox: {result.roi_result.smoke_union_bbox}")
```

## 单独使用 ROIPipeline

如果只需要提取 ROI 区域，不需要分类：

```python
from ultralytics import YOLO
from roi_extractor.video_processor import ROIPipeline

# 加载 YOLO 模型
yolo_model = YOLO("path/to/yolo_seg.pt")

# 创建 ROI Pipeline
roi_pipeline = ROIPipeline(
    model=yolo_model,
    confidence_threshold=0.5,
    use_ema=True,
    ema_alpha=0.3,
    bbox_padding=0.1,
)

# 处理一批连续图片
frames = [frame1, frame2, ..., frame10]
result = roi_pipeline.process_batch(
    frames=frames,
    detect_mode='all',
    skip_no_detection=False
)

# 获取处理后的帧（不需要的像素已置 0）
for processed_frame in result.frames:
    print(f"帧 {processed_frame.frame_idx}: {processed_frame.roi_type}")
    cv2.imshow("ROI", processed_frame.data)
```

## 单独使用 TemporalClassifier

如果已有预处理好的数据，直接使用分类器：

```python
import torch
from convlstm.models.temporal_classifier import (
    create_model,
    heatmap_to_prob,
    heatmap_to_pred
)

# 加载模型
classifier = create_model("path/to/convlstm.pt", num_classes=3)
classifier = classifier.to('cuda')
classifier.eval()

# 准备输入: (Batch, Time, 3, 640, 640) RGB normalized [0, 1]
x = torch.randn(1, 10, 3, 640, 640).cuda()

# 推理
with torch.no_grad():
    heatmap = classifier(x)           # (1, num_classes, 20, 20)
    probs = heatmap_to_prob(heatmap)  # (1, num_classes)
    pred = heatmap_to_pred(heatmap)   # (1,)

print(f"预测类别: {pred[0].item()}")
print(f"类别概率: {probs[0].cpu().numpy()}")
```

## 数据结构

### DetectionResult

```python
@dataclass
class DetectionResult:
    predicted_class: int        # 预测的类别索引
    class_probs: np.ndarray     # 各类别概率 (num_classes,)
    heatmap: np.ndarray         # 分类热力图 (num_classes, 20, 20)
    roi_result: BatchResult     # ROI 提取结果
    has_detection: bool         # 是否检测到火焰/烟雾
```

### BatchResult

```python
@dataclass
class BatchResult:
    frames: List[ProcessedFrame]  # 处理后的帧列表
    smoke_union_bbox: Optional[Tuple[int, int, int, int]]  # smoke 并集 bbox
    frames_with_fire: int         # 包含 fire 的帧数
    frames_with_smoke: int        # 包含 smoke 的帧数
```

### ProcessedFrame

```python
@dataclass
class ProcessedFrame:
    frame_idx: int              # 帧索引
    data: np.ndarray            # 处理后的帧 (H, W, 3)，不需要的像素为 0
    roi_type: str               # 'fire' | 'smoke' | 'fire+smoke' | 'none'
    has_fire: bool
    has_smoke: bool
    detections: List[Dict]      # 检测信息列表
```
