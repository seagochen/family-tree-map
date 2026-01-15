"""
Segment module for video segmentation processing.

This module provides APIs for processing videos using YOLO segmentation,
automatically determining mask vs bbox mode based on detection patterns.

Main classes:
    - VideoSegmentProcessor: Main API for video processing
    - ProcessedFrame: Result object containing processed frame data
    - ROIMode: Enum for mask/bbox processing modes
    - Detection: Detection result with bbox and optional mask

Example:
    from segment import VideoSegmentProcessor, process_video_api

    # Using class API
    processor = VideoSegmentProcessor("model.pt")
    for frame in processor.process_video("video.mp4"):
        if frame is not None:
            print(f"Mode: {frame.mode.value}, Detections: {len(frame.detections)}")

    # Using function API
    for frame in process_video_api("video.mp4", "model.pt"):
        if frame is not None:
            cv2.imshow("Result", frame.data)
"""

from .detection import Detection
from .mask_ema import MaskEMA
from .video_processor import (
    VideoSegmentProcessor,
    ProcessedFrame,
    VideoAnalysisResult,
    ROIMode,
    process_video_api,
    process_frames_api,
    DEFAULT_CONFIDENCE_THRESHOLD,
    MASK_CLASSES,
    BBOX_CLASSES,
)

__all__ = [
    # Main API
    "VideoSegmentProcessor",
    "process_video_api",
    "process_frames_api",
    # Data classes
    "ProcessedFrame",
    "VideoAnalysisResult",
    "Detection",
    "ROIMode",
    # Utilities
    "MaskEMA",
    # Constants
    "DEFAULT_CONFIDENCE_THRESHOLD",
    "MASK_CLASSES",
    "BBOX_CLASSES",
]
