"""
Detect module for video detection processing.

本模块提供使用 YOLO Detect 模型处理视频的 API，
支持 BBox EMA 平滑以减少检测框抖动。

主要类:
    - VideoDetectProcessor: 视频处理主 API
    - ProcessedFrame: 包含处理后帧数据的结果对象
    - Detection: 检测结果（bbox、置信度、类别）
    - BBoxEMA: 边界框 EMA 平滑器

示例:
    from detect import VideoDetectProcessor

    # 使用类 API
    processor = VideoDetectProcessor("model.pt")
    for frame in processor.process_video("video.mp4"):
        if frame is not None:
            print(f"Detections: {len(frame.detections)}")

    # 使用函数 API
    from detect import process_video_api
    for frame in process_video_api("video.mp4", "model.pt"):
        if frame is not None:
            for det in frame.detections:
                print(f"  {det['class_name']}: {det['bbox']}")
"""

from .detection import Detection
from .bbox_ema import BBoxEMA
from .video_processor import (
    VideoDetectProcessor,
    ProcessedFrame,
    process_video_api,
    process_frames_api,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_CLASS_NAMES,
    INFERENCE_SIZE,
)

__all__ = [
    # Main API
    "VideoDetectProcessor",
    "process_video_api",
    "process_frames_api",
    # Data classes
    "ProcessedFrame",
    "Detection",
    # Utilities
    "BBoxEMA",
    # Constants
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "DEFAULT_CLASS_NAMES",
    "INFERENCE_SIZE",
]
