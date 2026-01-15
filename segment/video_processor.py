"""
视频分割处理器 API

本模块提供使用 YOLO 分割模型处理视频的 API，
根据检测模式自动判断使用掩码模式还是边界框模式。

支持视频文件和帧列表（list[np.ndarray]）两种输入方式。

使用方法:
    from segment.video_processor import VideoSegmentProcessor, ProcessedFrame

    processor = VideoSegmentProcessor(model_path="path/to/model.pt")

    # 处理视频文件
    for frame in processor.process_video("path/to/video.mp4"):
        if frame is not None:
            # frame.data: 处理后的帧数据 (np.ndarray)
            # frame.mode: 'mask' 或 'bbox'
            # frame.frame_idx: 帧索引
            # frame.detections: 检测信息列表
            pass

    # 处理帧列表
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
    """ROI（感兴趣区域）处理模式。"""
    MASK = "mask"   # 使用像素级掩码作为 ROI
    BBOX = "bbox"   # 使用完整帧（边界框指示区域但不裁剪）


@dataclass
class ProcessedFrame:
    """单个视频帧的处理结果。"""
    frame_idx: int
    data: np.ndarray
    mode: ROIMode
    detections: List[Dict] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


@dataclass
class VideoAnalysisResult:
    """用于确定 ROI 策略的视频分析结果。"""
    recommended_mode: ROIMode
    total_frames: int
    sampled_frames: int
    mask_mode_frames: int
    bbox_mode_frames: int
    no_detection_frames: int
    bbox_ratio: float

    @property
    def has_targets(self) -> bool:
        """如果视频包含可检测目标则返回 True。"""
        return (self.mask_mode_frames + self.bbox_mode_frames) > 0


# 默认配置
DEFAULT_CLASS_NAMES = {
    0: "fire_core",
    1: "fire_obscured",
    2: "smoke_obscured",
    3: "smoke_source"
}

# 使用掩码模式的类别（像素级分割）
MASK_CLASSES = {0, 3}  # fire_core, smoke_source
# 使用边界框模式的类别（仅边界框）
BBOX_CLASSES = {1, 2}  # fire_obscured, smoke_obscured

# 处理参数
INFERENCE_SIZE = (640, 640)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
MASK_AREA_THRESHOLD = 0.25  # 掩码模式的最大掩码面积比（帧的25%）
BBOX_RATIO_THRESHOLD = 0.10  # 如果超过10%的帧需要边界框模式，则整个视频使用边界框模式
EMA_ALPHA = 0.2  # EMA 平滑因子


class VideoSegmentProcessor:
    """
    视频分割处理器 API。

    分析视频内容并根据检测模式自动确定最佳处理模式
    （掩码或边界框）。

    示例:
        processor = VideoSegmentProcessor("model.pt")

        # 处理视频并获取带有检测结果的帧
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
        初始化视频分割处理器。

        参数:
            model_path: YOLO 分割模型路径（.pt 文件）
            confidence_threshold: 检测的最小置信度（默认: 0.5）
            use_ema: 启用掩码的 EMA 平滑（默认: True）
            ema_alpha: EMA 平滑因子（默认: 0.2）
            class_names: 自定义类别名称映射（默认: 火焰/烟雾类别）
        """
        from ultralytics import YOLO

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"模型文件未找到: {model_path}")

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
        分析视频以确定最佳 ROI 策略。

        从视频中采样帧并确定掩码模式或边界框模式
        哪种更适合整个视频。

        参数:
            video_path: 视频文件路径
            sample_rate: 每隔 N 帧采样一次（默认: 10）

        返回:
            包含推荐模式和统计信息的 VideoAnalysisResult
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

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

        # 决策: 如果超过10%的帧需要边界框模式，则整个视频使用边界框模式
        detection_frames = mask_mode_count + bbox_mode_count
        bbox_ratio = bbox_mode_count / sampled_frames if sampled_frames > 0 else 0.0

        if detection_frames == 0:
            recommended_mode = ROIMode.MASK  # 无检测结果，但默认使用掩码模式
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
        处理视频并生成处理后的帧。

        这是主要的 API 方法。它分析视频以确定最佳处理模式，
        然后生成处理后的帧。

        参数:
            video_path: 视频文件路径
            skip_no_detection: 如果为 True，对无检测结果的帧返回 None
                               如果为 False，返回空帧数据
            force_mode: 强制使用指定模式而非自动检测

        生成:
            有检测结果的帧返回 ProcessedFrame 对象，
            无检测结果的帧返回 None（如果 skip_no_detection=True）
        """
        video_path = Path(video_path)

        # 分析视频以确定模式（除非强制指定）
        if force_mode is None:
            analysis = self.analyze_video(video_path)
            video_mode = analysis.recommended_mode

            # 如果整个视频未检测到目标，跳过处理
            if not analysis.has_targets:
                return
        else:
            video_mode = force_mode

        # 为新视频重置 EMA
        if self.use_ema:
            self._mask_ema = MaskEMA(alpha=self.ema_alpha)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

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
        分析帧列表以确定最佳 ROI 策略。

        参数:
            frames: 帧列表 (H, W, C)
            sample_rate: 每隔 N 帧采样一次（默认: 10）

        返回:
            包含推荐模式和统计信息的 VideoAnalysisResult
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
        处理帧列表并生成处理结果。

        此方法分析帧以确定最佳处理模式，然后生成处理后的帧。

        参数:
            frames: 帧列表 (H, W, C)，类型为 np.ndarray
            skip_no_detection: 如果为 True，对无检测结果的帧返回 None
            force_mode: 强制使用指定模式而非自动检测

        生成:
            有检测结果的帧返回 ProcessedFrame 对象，
            无检测结果的帧返回 None（如果 skip_no_detection=True）

        示例:
            frames = [cv2.imread(f) for f in image_files]
            for result in processor.process_frames(frames):
                if result is not None:
                    print(f"Frame {result.frame_idx}: {result.mode.value}")
        """
        if len(frames) == 0:
            return

        # 分析帧以确定模式（除非强制指定）
        if force_mode is None:
            analysis = self.analyze_frames(frames)
            video_mode = analysis.recommended_mode

            # 如果未检测到目标，跳过处理
            if not analysis.has_targets:
                return
        else:
            video_mode = force_mode

        # 为新序列重置 EMA
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
        处理单个帧。

        参数:
            frame: 输入帧 (H, W, C)
            mode: 处理模式（掩码或边界框）

        返回:
            如果有检测结果返回 ProcessedFrame，否则返回 None
        """
        return self._process_frame(frame, mode)

    def _process_frame(
        self,
        frame: np.ndarray,
        mode: ROIMode
    ) -> Optional[ProcessedFrame]:
        """内部帧处理实现。"""
        resized = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        h, w = resized.shape[:2]

        results = self.model(resized, conf=self.confidence_threshold, verbose=False)[0]
        detections = self._extract_detections(results, (h, w))

        # 无检测结果 -> 返回 None
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
            # 边界框模式: 返回完整帧
            return ProcessedFrame(
                frame_idx=0,
                data=resized.copy(),
                mode=ROIMode.BBOX,
                detections=det_info
            )
        else:
            # 掩码模式: 对帧应用掩码
            mask_dets = [d for d in detections if d.class_id in MASK_CLASSES]

            if not mask_dets:
                # 无掩码兼容的检测结果，回退到完整帧
                return ProcessedFrame(
                    frame_idx=0,
                    data=resized.copy(),
                    mode=ROIMode.BBOX,
                    detections=det_info
                )

            # 创建合并掩码
            combined_mask = self._create_combined_mask(mask_dets, (h, w))

            # 如果启用则应用 EMA 平滑
            if self._mask_ema is not None:
                combined_mask = self._mask_ema.update(combined_mask)

            # 对帧应用掩码
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
        """从 YOLO 结果中提取 Detection 对象。"""
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
        """决定帧应使用掩码模式还是边界框模式。"""
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
        """从多个检测结果创建合并掩码。"""
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
    处理视频文件的便捷函数。

    这是 VideoSegmentProcessor 的简单封装，便于快速使用。

    参数:
        video_path: 视频文件路径
        model_path: YOLO 模型文件路径
        confidence_threshold: 最小检测置信度
        use_ema: 启用 EMA 平滑
        skip_no_detection: 跳过无检测结果的帧

    生成:
        有检测结果的帧返回 ProcessedFrame 对象

    示例:
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
    处理帧列表的便捷函数。

    这是 VideoSegmentProcessor 的简单封装，便于快速使用。

    参数:
        frames: 帧列表 (H, W, C)，类型为 np.ndarray
        model_path: YOLO 模型文件路径
        confidence_threshold: 最小检测置信度
        use_ema: 启用 EMA 平滑
        skip_no_detection: 跳过无检测结果的帧

    生成:
        有检测结果的帧返回 ProcessedFrame 对象

    示例:
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
