"""
ROI Extractor - 火焰/烟雾 ROI 提取模块

从视频中提取火焰/烟雾的感兴趣区域 (ROI)：
- fire (class 0): 使用分割 mask + EMA 平滑 + 边缘平滑
- smoke (class 2): 使用并集 bbox（计算视频中所有 smoke 的并集）
- person (class 1): 忽略（负样本）

使用示例:
    from roi_extractor import VideoROIProcessor

    processor = VideoROIProcessor("segment.pt", use_ema=True)

    # 处理视频
    for frame in processor.process_video("video.mp4"):
        print(f"Frame {frame.frame_idx}: {frame.roi_type}")
        # frame.data 是处理后的帧（ROI 提取后）
"""

from .constants import (
    CLASS_NAMES,
    CLASS_COLORS,
    TARGET_CLASSES,
    SEGMENT_CLASSES,
    BBOX_CLASSES,
    NEGATIVE_CLASSES,
    INFERENCE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
    MASK_BLUR_KERNEL_SIZE,
    BBOX_PADDING_RATIO,
    DEFAULT_SAMPLE_RATE,
)

from .detection import (
    Detection,
    extract_detections_from_yolo,
    compute_bbox_union,
    add_bbox_padding,
    apply_bbox_mask,
    apply_segmentation_mask,
    get_class_color,
)

from .mask_ema import MaskEMA

from .video_processor import (
    VideoROIProcessor,
    ProcessedFrame,
    VideoAnalysisResult,
    DetectMode,
)

__all__ = [
    # 主要类
    'VideoROIProcessor',
    'ProcessedFrame',
    'VideoAnalysisResult',
    'MaskEMA',
    'Detection',
    # 类型
    'DetectMode',
    # 常量
    'CLASS_NAMES',
    'CLASS_COLORS',
    'TARGET_CLASSES',
    'SEGMENT_CLASSES',
    'BBOX_CLASSES',
    'NEGATIVE_CLASSES',
    'INFERENCE_SIZE',
    'DEFAULT_CONFIDENCE_THRESHOLD',
    'EMA_ALPHA',
    'MASK_BLUR_KERNEL_SIZE',
    'BBOX_PADDING_RATIO',
    'DEFAULT_SAMPLE_RATE',
    # 工具函数
    'extract_detections_from_yolo',
    'compute_bbox_union',
    'add_bbox_padding',
    'apply_bbox_mask',
    'apply_segmentation_mask',
    'get_class_color',
]
