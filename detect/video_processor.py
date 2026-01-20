"""
视频检测处理器 API

本模块提供使用 YOLO Detect 模型处理视频的 API，
支持 BBox EMA 平滑以减少检测框抖动。

使用方法:
    from detect import VideoDetectProcessor, ProcessedFrame

    processor = VideoDetectProcessor(model_path="path/to/model.pt")

    # 处理视频文件
    for frame in processor.process_video("path/to/video.mp4"):
        if frame is not None:
            # frame.data: 原始帧数据 (np.ndarray)
            # frame.frame_idx: 帧索引
            # frame.detections: 检测信息列表
            pass

    # 处理帧列表
    frames = [frame1, frame2, frame3]  # list[np.ndarray]
    for frame in processor.process_frames(frames):
        if frame is not None:
            print(f"Frame {frame.frame_idx}: {len(frame.detections)} detections")
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Generator, Sequence
from dataclasses import dataclass, field

from .detection import Detection
from .bbox_ema import BBoxEMA


@dataclass
class ProcessedFrame:
    """单个视频帧的处理结果。"""
    frame_idx: int
    data: np.ndarray
    detections: List[Dict] = field(default_factory=list)

    @property
    def has_detections(self) -> bool:
        return len(self.detections) > 0


# 默认配置
DEFAULT_CLASS_NAMES = {
    0: "fire",
    1: "smoke",
    2: "standing",
    3: "squatting"
}

# 处理参数
INFERENCE_SIZE = (640, 640)
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
EMA_ALPHA = 0.3  # EMA 平滑因子


class VideoDetectProcessor:
    """
    视频检测处理器 API。

    使用 YOLO Detect 模型处理视频帧，支持 BBox EMA 平滑。
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        class_names: Optional[Dict[int, str]] = None,
        use_ema: bool = True,
        ema_alpha: float = EMA_ALPHA,
        device: str = 'auto'
    ):
        """
        初始化视频检测处理器。

        参数:
            model_path: YOLO Detect 模型文件路径
            confidence_threshold: 检测置信度阈值 (默认: 0.5)
            class_names: 类别 ID 到名称的映射 (可选)
            use_ema: 是否使用 EMA 平滑 (默认: True)
            ema_alpha: EMA 平滑因子 (默认: 0.3)
            device: 推理设备 ('auto', 'cuda', 'cpu')
        """
        from ultralytics import YOLO

        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.class_names = class_names or DEFAULT_CLASS_NAMES
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha

        # 设置设备
        if device == 'auto':
            import torch
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # EMA 平滑器（按需初始化）
        self._bbox_ema: Optional[BBoxEMA] = None

    def process_video(
        self,
        video_path: str,
        skip_no_detection: bool = True
    ) -> Generator[Optional[ProcessedFrame], None, None]:
        """
        处理视频文件并生成检测结果。

        参数:
            video_path: 视频文件路径
            skip_no_detection: 如果为 True，对无检测结果的帧返回 None

        生成:
            有检测结果的帧返回 ProcessedFrame 对象，
            无检测结果的帧返回 None（如果 skip_no_detection=True）
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        # 为新视频重置 EMA
        if self.use_ema:
            self._bbox_ema = BBoxEMA(alpha=self.ema_alpha)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            processed = self._process_frame(frame)

            if processed is None or not processed.has_detections:
                if skip_no_detection:
                    yield None
                else:
                    yield ProcessedFrame(
                        frame_idx=frame_idx,
                        data=frame,
                        detections=[]
                    )
            else:
                processed.frame_idx = frame_idx
                processed.data = frame  # 保留原始帧
                yield processed

            frame_idx += 1

        cap.release()

    def process_frames(
        self,
        frames: Sequence[np.ndarray],
        skip_no_detection: bool = True
    ) -> Generator[Optional[ProcessedFrame], None, None]:
        """
        处理帧列表并生成检测结果。

        参数:
            frames: 帧列表 (H, W, C)，类型为 np.ndarray
            skip_no_detection: 如果为 True，对无检测结果的帧返回 None

        生成:
            有检测结果的帧返回 ProcessedFrame 对象，
            无检测结果的帧返回 None（如果 skip_no_detection=True）
        """
        if len(frames) == 0:
            return

        # 为新序列重置 EMA
        if self.use_ema:
            self._bbox_ema = BBoxEMA(alpha=self.ema_alpha)

        for frame_idx, frame in enumerate(frames):
            processed = self._process_frame(frame)

            if processed is None or not processed.has_detections:
                if skip_no_detection:
                    yield None
                else:
                    yield ProcessedFrame(
                        frame_idx=frame_idx,
                        data=frame,
                        detections=[]
                    )
            else:
                processed.frame_idx = frame_idx
                processed.data = frame  # 保留原始帧
                yield processed

    def process_frame(self, frame: np.ndarray) -> Optional[ProcessedFrame]:
        """
        处理单个帧。

        参数:
            frame: 输入帧 (H, W, C)

        返回:
            如果有检测结果返回 ProcessedFrame，否则返回 None
        """
        return self._process_frame(frame)

    def _process_frame(self, frame: np.ndarray) -> Optional[ProcessedFrame]:
        """内部帧处理实现。"""
        # 调整尺寸进行推理
        resized = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        h, w = resized.shape[:2]

        # YOLO 推理
        results = self.model(resized, conf=self.confidence_threshold, verbose=False)[0]

        # 提取检测结果
        detections = self._extract_detections(results)

        # 转换为字典格式
        det_info = [
            {
                'class_id': d.class_id,
                'class_name': self.class_names.get(d.class_id, f"class_{d.class_id}"),
                'confidence': d.confidence,
                'bbox': d.bbox
            }
            for d in detections
        ]

        # 应用 EMA 平滑
        if self.use_ema and self._bbox_ema is not None:
            det_info = self._bbox_ema.update(det_info)

        # 无检测结果 -> 返回 None
        if not det_info:
            return None

        return ProcessedFrame(
            frame_idx=0,
            data=resized,
            detections=det_info
        )

    def _extract_detections(self, results) -> List[Detection]:
        """从 YOLO 结果中提取 Detection 对象。"""
        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id in self.class_names:
                    detections.append(Detection.from_yolo_box(box))

        return detections

    def reset_ema(self):
        """重置 EMA 状态。"""
        if self._bbox_ema is not None:
            self._bbox_ema.reset()


def process_video_api(
    video_path: str,
    model_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_ema: bool = True,
    skip_no_detection: bool = True
) -> Generator[Optional[ProcessedFrame], None, None]:
    """
    便捷函数：处理视频文件。

    参数:
        video_path: 视频文件路径
        model_path: YOLO 模型文件路径
        confidence_threshold: 检测置信度阈值
        use_ema: 是否使用 EMA 平滑
        skip_no_detection: 是否跳过无检测结果的帧

    生成:
        ProcessedFrame 对象或 None
    """
    processor = VideoDetectProcessor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_ema=use_ema
    )

    yield from processor.process_video(video_path, skip_no_detection=skip_no_detection)


def process_frames_api(
    frames: Sequence[np.ndarray],
    model_path: str,
    confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
    use_ema: bool = True,
    skip_no_detection: bool = True
) -> Generator[Optional[ProcessedFrame], None, None]:
    """
    便捷函数：处理帧列表。

    参数:
        frames: 帧列表
        model_path: YOLO 模型文件路径
        confidence_threshold: 检测置信度阈值
        use_ema: 是否使用 EMA 平滑
        skip_no_detection: 是否跳过无检测结果的帧

    生成:
        ProcessedFrame 对象或 None
    """
    processor = VideoDetectProcessor(
        model_path=model_path,
        confidence_threshold=confidence_threshold,
        use_ema=use_ema
    )

    yield from processor.process_frames(frames, skip_no_detection=skip_no_detection)
