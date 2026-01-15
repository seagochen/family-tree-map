"""
Video Segmentation Processor API

This module provides an API to process videos using YOLO segmentation models,
automatically determining whether to use mask mode or bbox mode based on
detection patterns.

Supports both video files and frame lists (list[np.ndarray]).

Usage:
    from segment.video_processor import VideoSegmentProcessor, ProcessedFrame

    processor = VideoSegmentProcessor(model_path="path/to/model.pt")

    # Process video file
    for frame in processor.process_video("path/to/video.mp4"):
        if frame is not None:
            # frame.data: processed frame data (np.ndarray)
            # frame.mode: 'mask' or 'bbox'
            # frame.frame_idx: frame index
            # frame.detections: list of detection info
            pass

    # Process frame list
    frames = [frame1, frame2, frame3]  # list[np.ndarray]
    for frame in processor.process_frames(frames):
        if frame is not None:
            print(f"Frame {frame.frame_idx}: mode={frame.mode.value}")
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Union, Sequence
from dataclasses import dataclass, field
from enum import Enum

from .detection import Detection
from .mask_ema import MaskEMA


class ROIMode(Enum):
    """ROI (Region of Interest) processing mode."""
    MASK = "mask"   # Use pixel-level mask for ROI
    BBOX = "bbox"   # Use full frame (bbox indicates regions but doesn't crop)


@dataclass
class ProcessedFrame:
    """Result of processing a single video frame."""
    frame_idx: int
    data: np.ndarray
    mode: ROIMode
    detections: List[Dict] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


@dataclass
class VideoAnalysisResult:
    """Result of video analysis for ROI strategy determination."""
    recommended_mode: ROIMode
    total_frames: int
    sampled_frames: int
    mask_mode_frames: int
    bbox_mode_frames: int
    no_detection_frames: int
    bbox_ratio: float

    @property
    def has_targets(self) -> bool:
        """Returns True if video contains detectable targets."""
        return (self.mask_mode_frames + self.bbox_mode_frames) > 0


# Default configuration
DEFAULT_CLASS_NAMES = {
    0: "fire_core",
    1: "fire_obscured",
    2: "smoke_obscured",
    3: "smoke_source"
}

# Classes that use mask mode (pixel-level segmentation)
MASK_CLASSES = {0, 3}  # fire_core, smoke_source
# Classes that use bbox mode (bounding box only)
BBOX_CLASSES = {1, 2}  # fire_obscured, smoke_obscured

# Processing parameters
INFERENCE_SIZE = (640, 640)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MASK_AREA_THRESHOLD = 0.25  # Max mask area ratio for mask mode (25% of frame)
BBOX_RATIO_THRESHOLD = 0.10  # If >10% frames need bbox mode, use bbox for entire video
EMA_ALPHA = 0.2  # EMA smoothing factor


class VideoSegmentProcessor:
    """
    Video segmentation processor API.

    Analyzes video content and automatically determines optimal processing mode
    (mask or bbox) based on detection patterns.

    Example:
        processor = VideoSegmentProcessor("model.pt")

        # Process video and get frames with detections
        for frame in processor.process_video("video.mp4"):
            if frame is not None:
                cv2.imshow("Result", frame.data)
                print(f"Frame {frame.frame_idx}: mode={frame.mode.value}")
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_ema: bool = True,
        ema_alpha: float = EMA_ALPHA,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize the video segment processor.

        Args:
            model_path: Path to YOLO segmentation model (.pt file)
            confidence_threshold: Minimum confidence for detections (default: 0.5)
            use_ema: Enable EMA smoothing for masks (default: True)
            ema_alpha: EMA smoothing factor (default: 0.2)
            class_names: Custom class name mapping (default: fire/smoke classes)
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(str(self.model_path))
        self.confidence_threshold = confidence_threshold
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self._mask_ema: Optional[MaskEMA] = None

    def analyze_video(
        self,
        video_path: Union[str, Path],
        sample_rate: int = 10
    ) -> VideoAnalysisResult:
        """
        Analyze video to determine optimal ROI strategy.

        Samples frames from the video and determines whether mask mode or
        bbox mode is more appropriate for the entire video.

        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame (default: 10)

        Returns:
            VideoAnalysisResult with recommended mode and statistics
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        sampled_frames = 0
        mask_mode_count = 0
        bbox_mode_count = 0
        no_detection_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_rate == 0:
                resized = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
                h, w = resized.shape[:2]

                results = self.model(resized, conf=self.confidence_threshold, verbose=False)[0]
                detections = self._extract_detections(results, (h, w))

                if not detections:
                    no_detection_count += 1
                else:
                    mask_dets = [d for d in detections if d.class_id in MASK_CLASSES]
                    frame_mode = self._decide_frame_mode(mask_dets, (h, w))

                    if frame_mode == ROIMode.MASK:
                        mask_mode_count += 1
                    else:
                        bbox_mode_count += 1

                sampled_frames += 1

            frame_idx += 1

        cap.release()

        # Decision: if >10% of frames need bbox mode, use bbox for entire video
        detection_frames = mask_mode_count + bbox_mode_count
        bbox_ratio = bbox_mode_count / sampled_frames if sampled_frames > 0 else 0.0

        if detection_frames == 0:
            recommended_mode = ROIMode.MASK  # No detections, but use mask mode as default
        elif bbox_ratio > BBOX_RATIO_THRESHOLD:
            recommended_mode = ROIMode.BBOX
        else:
            recommended_mode = ROIMode.MASK

        return VideoAnalysisResult(
            recommended_mode=recommended_mode,
            total_frames=total_frames,
            sampled_frames=sampled_frames,
            mask_mode_frames=mask_mode_count,
            bbox_mode_frames=bbox_mode_count,
            no_detection_frames=no_detection_count,
            bbox_ratio=bbox_ratio
        )

    def process_video(
        self,
        video_path: Union[str, Path],
        skip_no_detection: bool = True,
        force_mode: Optional[ROIMode] = None
    ) -> Generator[Optional[ProcessedFrame], None, None]:
        """
        Process video and yield processed frames.

        This is the main API method. It analyzes the video to determine
        optimal processing mode, then yields processed frames.

        Args:
            video_path: Path to video file
            skip_no_detection: If True, yields None for frames without detections
                               If False, yields empty frame data
            force_mode: Force a specific mode instead of auto-detecting

        Yields:
            ProcessedFrame objects for frames with detections,
            None for frames without detections (if skip_no_detection=True)
        """
        video_path = Path(video_path)

        # Analyze video to determine mode (unless forced)
        if force_mode is None:
            analysis = self.analyze_video(video_path)
            video_mode = analysis.recommended_mode

            # If no targets detected in entire video, skip processing
            if not analysis.has_targets:
                return
        else:
            video_mode = force_mode

        # Reset EMA for new video
        if self.use_ema:
            self._mask_ema = MaskEMA(alpha=self.ema_alpha)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self._process_frame(frame, video_mode)

            if processed is None:
                if skip_no_detection:
                    yield None
                else:
                    yield ProcessedFrame(
                        frame_idx=frame_idx,
                        data=np.zeros((*INFERENCE_SIZE, 3), dtype=np.uint8),
                        mode=video_mode,
                        detections=[]
                    )
            else:
                processed.frame_idx = frame_idx
                yield processed

            frame_idx += 1

        cap.release()

    def analyze_frames(
        self,
        frames: Sequence[np.ndarray],
        sample_rate: int = 10
    ) -> VideoAnalysisResult:
        """
        Analyze frame list to determine optimal ROI strategy.

        Args:
            frames: List of frames (H, W, C)
            sample_rate: Sample every Nth frame (default: 10)

        Returns:
            VideoAnalysisResult with recommended mode and statistics
        """
        total_frames = len(frames)
        sampled_frames = 0
        mask_mode_count = 0
        bbox_mode_count = 0
        no_detection_count = 0

        for frame_idx, frame in enumerate(frames):
            if frame_idx % sample_rate == 0:
                resized = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
                h, w = resized.shape[:2]

                results = self.model(resized, conf=self.confidence_threshold, verbose=False)[0]
                detections = self._extract_detections(results, (h, w))

                if not detections:
                    no_detection_count += 1
                else:
                    mask_dets = [d for d in detections if d.class_id in MASK_CLASSES]
                    frame_mode = self._decide_frame_mode(mask_dets, (h, w))

                    if frame_mode == ROIMode.MASK:
                        mask_mode_count += 1
                    else:
                        bbox_mode_count += 1

                sampled_frames += 1

        detection_frames = mask_mode_count + bbox_mode_count
        bbox_ratio = bbox_mode_count / sampled_frames if sampled_frames > 0 else 0.0

        if detection_frames == 0:
            recommended_mode = ROIMode.MASK
        elif bbox_ratio > BBOX_RATIO_THRESHOLD:
            recommended_mode = ROIMode.BBOX
        else:
            recommended_mode = ROIMode.MASK

        return VideoAnalysisResult(
            recommended_mode=recommended_mode,
            total_frames=total_frames,
            sampled_frames=sampled_frames,
            mask_mode_frames=mask_mode_count,
            bbox_mode_frames=bbox_mode_count,
            no_detection_frames=no_detection_count,
            bbox_ratio=bbox_ratio
        )

    def process_frames(
        self,
        frames: Sequence[np.ndarray],
        skip_no_detection: bool = True,
        force_mode: Optional[ROIMode] = None
    ) -> Generator[Optional[ProcessedFrame], None, None]:
        """
        Process a list of frames and yield processed results.

        This method analyzes the frames to determine optimal processing mode,
        then yields processed frames.

        Args:
            frames: List of frames (H, W, C) as np.ndarray
            skip_no_detection: If True, yields None for frames without detections
            force_mode: Force a specific mode instead of auto-detecting

        Yields:
            ProcessedFrame objects for frames with detections,
            None for frames without detections (if skip_no_detection=True)

        Example:
            frames = [cv2.imread(f) for f in image_files]
            for result in processor.process_frames(frames):
                if result is not None:
                    print(f"Frame {result.frame_idx}: {result.mode.value}")
        """
        if len(frames) == 0:
            return

        # Analyze frames to determine mode (unless forced)
        if force_mode is None:
            analysis = self.analyze_frames(frames)
            video_mode = analysis.recommended_mode

            # If no targets detected, skip processing
            if not analysis.has_targets:
                return
        else:
            video_mode = force_mode

        # Reset EMA for new sequence
        if self.use_ema:
            self._mask_ema = MaskEMA(alpha=self.ema_alpha)

        for frame_idx, frame in enumerate(frames):
            processed = self._process_frame(frame, video_mode)

            if processed is None:
                if skip_no_detection:
                    yield None
                else:
                    yield ProcessedFrame(
                        frame_idx=frame_idx,
                        data=np.zeros((*INFERENCE_SIZE, 3), dtype=np.uint8),
                        mode=video_mode,
                        detections=[]
                    )
            else:
                processed.frame_idx = frame_idx
                yield processed

    def process_frame(
        self,
        frame: np.ndarray,
        mode: ROIMode = ROIMode.MASK
    ) -> Optional[ProcessedFrame]:
        """
        Process a single frame.

        Args:
            frame: Input frame (H, W, C)
            mode: Processing mode (mask or bbox)

        Returns:
            ProcessedFrame if detections found, None otherwise
        """
        return self._process_frame(frame, mode)

    def _process_frame(
        self,
        frame: np.ndarray,
        mode: ROIMode
    ) -> Optional[ProcessedFrame]:
        """Internal frame processing implementation."""
        resized = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        h, w = resized.shape[:2]

        results = self.model(resized, conf=self.confidence_threshold, verbose=False)[0]
        detections = self._extract_detections(results, (h, w))

        # No detections -> return None
        if not detections:
            return None

        det_info = [
            {
                'class_id': d.class_id,
                'class_name': self.class_names.get(d.class_id, f"class_{d.class_id}"),
                'confidence': d.confidence
            }
            for d in detections
        ]

        if mode == ROIMode.BBOX:
            # Bbox mode: return full frame
            return ProcessedFrame(
                frame_idx=0,
                data=resized.copy(),
                mode=ROIMode.BBOX,
                detections=det_info
            )
        else:
            # Mask mode: apply mask to frame
            mask_dets = [d for d in detections if d.class_id in MASK_CLASSES]

            if not mask_dets:
                # No mask-compatible detections, fallback to full frame
                return ProcessedFrame(
                    frame_idx=0,
                    data=resized.copy(),
                    mode=ROIMode.BBOX,
                    detections=det_info
                )

            # Create combined mask
            combined_mask = self._create_combined_mask(mask_dets, (h, w))

            # Apply EMA smoothing if enabled
            if self._mask_ema is not None:
                combined_mask = self._mask_ema.update(combined_mask)

            # Apply mask to frame
            output = resized.copy()
            output[combined_mask == 0] = 0

            return ProcessedFrame(
                frame_idx=0,
                data=output,
                mode=ROIMode.MASK,
                detections=det_info
            )

    def _extract_detections(
        self,
        results,
        frame_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Extract Detection objects from YOLO results."""
        h, w = frame_shape
        detections = []

        if results.masks is not None:
            for box, mask in zip(results.boxes, results.masks.data):
                class_id = int(box.cls[0])
                if class_id in self.class_names:
                    mask_np = mask.cpu().numpy()
                    mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_binary = (mask_resized > 0.5).astype(np.uint8)
                    detections.append(Detection.from_yolo_box(box, mask_binary))

        return detections

    def _decide_frame_mode(
        self,
        mask_detections: List[Detection],
        frame_shape: Tuple[int, int]
    ) -> ROIMode:
        """Decide whether frame should use mask or bbox mode."""
        if not mask_detections:
            return ROIMode.BBOX

        h, w = frame_shape
        total_area = h * w

        combined_mask = self._create_combined_mask(mask_detections, frame_shape)
        mask_area = np.sum(combined_mask > 0)
        mask_ratio = mask_area / total_area

        return ROIMode.MASK if mask_ratio <= MASK_AREA_THRESHOLD else ROIMode.BBOX

    def _create_combined_mask(
        self,
        detections: List[Detection],
        frame_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Create combined mask from multiple detections."""
        h, w = frame_shape
        combined = np.zeros((h, w), dtype=np.uint8)

        for det in detections:
            if det.mask is not None:
                if det.mask.shape != (h, w):
                    mask_resized = cv2.resize(
                        det.mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = det.mask.astype(np.uint8)
                combined = np.maximum(combined, mask_resized)

        return combined


def process_video_api(
    video_path: Union[str, Path],
    model_path: Union[str, Path],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_ema: bool = True,
    skip_no_detection: bool = True
) -> Generator[Optional[ProcessedFrame], None, None]:
    """
    Convenience function to process a video file.

    This is a simple wrapper around VideoSegmentProcessor for quick usage.

    Args:
        video_path: Path to video file
        model_path: Path to YOLO model file
        confidence_threshold: Minimum detection confidence
        use_ema: Enable EMA smoothing
        skip_no_detection: Skip frames without detections

    Yields:
        ProcessedFrame objects for frames with detections

    Example:
        for frame in process_video_api("video.mp4", "model.pt"):
            if frame is not None:
                print(f"Frame {frame.frame_idx}: {frame.mode.value}")
                cv2.imshow("Result", frame.data)
    """
    processor = VideoSegmentProcessor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_ema=use_ema
    )

    yield from processor.process_video(
        video_path=video_path,
        skip_no_detection=skip_no_detection
    )


def process_frames_api(
    frames: Sequence[np.ndarray],
    model_path: Union[str, Path],
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_ema: bool = True,
    skip_no_detection: bool = True
) -> Generator[Optional[ProcessedFrame], None, None]:
    """
    Convenience function to process a list of frames.

    This is a simple wrapper around VideoSegmentProcessor for quick usage.

    Args:
        frames: List of frames (H, W, C) as np.ndarray
        model_path: Path to YOLO model file
        confidence_threshold: Minimum detection confidence
        use_ema: Enable EMA smoothing
        skip_no_detection: Skip frames without detections

    Yields:
        ProcessedFrame objects for frames with detections

    Example:
        frames = [cv2.imread(f) for f in image_files]
        for frame in process_frames_api(frames, "model.pt"):
            if frame is not None:
                print(f"Frame {frame.frame_idx}: {frame.mode.value}")
                cv2.imshow("Result", frame.data)
    """
    processor = VideoSegmentProcessor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_ema=use_ema
    )

    yield from processor.process_frames(
        frames=frames,
        skip_no_detection=skip_no_detection
    )
