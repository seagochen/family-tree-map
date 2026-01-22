"""
视频 ROI 提取处理器

从视频中提取火焰/烟雾的感兴趣区域 (ROI)：
- fire (class 0): 使用分割 mask + EMA 平滑 + 边缘平滑
- smoke (class 2): 使用并集 bbox（计算视频中所有 smoke 的并集）
- person (class 1): 忽略（负样本）

当同时检测到 fire 和 smoke 时：
- fire 区域使用 mask 提取
- smoke 区域使用 bbox 提取
- 组合两个区域的 mask
"""

import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Generator, Literal
from ultralytics import YOLO

from .constants import (
    CLASS_NAMES,
    TARGET_CLASSES,
    SEGMENT_CLASSES,
    BBOX_CLASSES,
    INFERENCE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
    BBOX_PADDING_RATIO,
    DEFAULT_SAMPLE_RATE,
)
from .detection import (
    Detection,
    extract_detections_from_yolo,
    compute_bbox_union,
    add_bbox_padding,
)
from .mask_ema import MaskEMA


# 检测模式类型
DetectMode = Literal['all', 'fire', 'smoke']


@dataclass
class ProcessedFrame:
    """处理后的帧数据"""
    frame_idx: int                           # 帧索引
    data: np.ndarray                         # 处理后的帧数据 (ROI 提取后)
    roi_type: str                            # ROI 类型: 'fire', 'smoke', 'fire+smoke', 'none'
    has_fire: bool = False                   # 是否包含 fire
    has_smoke: bool = False                  # 是否包含 smoke
    detections: List[Dict] = field(default_factory=list)  # 检测信息列表

    @property
    def has_detections(self) -> bool:
        """是否有检测结果"""
        return self.has_fire or self.has_smoke


@dataclass
class VideoAnalysisResult:
    """视频分析结果"""
    total_frames: int
    sampled_frames: int
    frames_with_fire: int
    frames_with_smoke: int
    smoke_bboxes_count: int
    smoke_union_bbox: Optional[Tuple[int, int, int, int]]
    fps: float
    detect_mode: str


class VideoROIProcessor:
    """
    视频 ROI 处理器

    处理流程：
    1. 分析视频获取 smoke 的并集 bbox（可选）
    2. 逐帧处理：
       - fire: 提取分割 mask + EMA 平滑
       - smoke: 使用预计算的并集 bbox
       - 组合两个区域
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_ema: bool = True,
        ema_alpha: float = EMA_ALPHA,
        bbox_padding: float = BBOX_PADDING_RATIO,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        """
        初始化处理器

        Args:
            model_path: YOLO 分割模型路径
            confidence_threshold: 置信度阈值
            use_ema: 是否使用 EMA 平滑 fire mask
            ema_alpha: EMA 平滑因子
            bbox_padding: smoke bbox 内边距比例
            sample_rate: 视频分析时的采样率（每 N 帧采样一次）
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.bbox_padding = bbox_padding
        self.sample_rate = sample_rate

        # 内部状态
        self._mask_ema: Optional[MaskEMA] = None
        self._smoke_union_bbox: Optional[Tuple[int, int, int, int]] = None
        self._frame_idx = 0

    def analyze_video_for_smoke_bbox(
        self,
        video_path: str,
        detect_mode: DetectMode = 'all'
    ) -> VideoAnalysisResult:
        """
        分析视频计算 smoke 的稳定并集 bbox

        Args:
            video_path: 视频文件路径
            detect_mode: 检测模式 ('all', 'fire', 'smoke')

        Returns:
            VideoAnalysisResult 分析结果
        """
        # 如果只检测 fire，跳过 smoke bbox 分析
        if detect_mode == 'fire':
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"无法打开视频: {video_path}")
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            self._smoke_union_bbox = None
            return VideoAnalysisResult(
                total_frames=total_frames,
                sampled_frames=0,
                frames_with_fire=0,
                frames_with_smoke=0,
                smoke_bboxes_count=0,
                smoke_union_bbox=None,
                fps=fps,
                detect_mode=detect_mode
            )

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        all_smoke_bboxes = []
        frame_idx = 0
        sampled_frames = 0
        frames_with_smoke = 0
        frames_with_fire = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.sample_rate == 0:
                resized_frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
                h, w = resized_frame.shape[:2]

                results = self.model(resized_frame, conf=self.confidence_threshold, verbose=False)[0]
                detections = extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, self.confidence_threshold)

                # 分离 fire 和 smoke 检测
                smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]
                fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]

                if smoke_dets:
                    frames_with_smoke += 1
                    for det in smoke_dets:
                        all_smoke_bboxes.append((det.x1, det.y1, det.x2, det.y2))

                if fire_dets:
                    frames_with_fire += 1

                sampled_frames += 1

            frame_idx += 1

        cap.release()

        # 计算 smoke 的并集 bbox
        smoke_union_bbox = compute_bbox_union(all_smoke_bboxes)

        if smoke_union_bbox is not None:
            h, w = INFERENCE_SIZE
            smoke_union_bbox = add_bbox_padding(smoke_union_bbox, (h, w), self.bbox_padding)

        self._smoke_union_bbox = smoke_union_bbox

        return VideoAnalysisResult(
            total_frames=total_frames,
            sampled_frames=sampled_frames,
            frames_with_fire=frames_with_fire,
            frames_with_smoke=frames_with_smoke,
            smoke_bboxes_count=len(all_smoke_bboxes),
            smoke_union_bbox=smoke_union_bbox,
            fps=fps,
            detect_mode=detect_mode
        )

    def process_frame(
        self,
        frame: np.ndarray,
        detect_mode: DetectMode = 'all'
    ) -> ProcessedFrame:
        """
        处理单帧

        Args:
            frame: 输入帧 (BGR)
            detect_mode: 检测模式 ('all', 'fire', 'smoke')

        Returns:
            ProcessedFrame 处理结果
        """
        resized_frame = cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)
        h, w = resized_frame.shape[:2]

        results = self.model(resized_frame, conf=self.confidence_threshold, verbose=False)[0]
        detections = extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, self.confidence_threshold)

        # 根据检测模式筛选
        if detect_mode == 'fire':
            fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]
            smoke_dets = []
        elif detect_mode == 'smoke':
            fire_dets = []
            smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]
        else:  # 'all'
            fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]
            smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]

        # 构建检测信息
        det_info = [
            {'class_id': d.class_id, 'class_name': CLASS_NAMES[d.class_id], 'confidence': d.confidence, 'bbox': d.bbox}
            for d in (fire_dets + smoke_dets)
        ]

        # 初始化组合 mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        has_fire = len(fire_dets) > 0
        has_smoke = len(smoke_dets) > 0 and self._smoke_union_bbox is not None

        # 处理 fire: 使用分割 mask
        if has_fire:
            fire_mask = np.zeros((h, w), dtype=np.uint8)
            for det in fire_dets:
                if det.mask is not None:
                    if det.mask.shape != (h, w):
                        mask_resized = cv2.resize(det.mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
                    else:
                        mask_resized = det.mask.astype(np.uint8)
                    fire_mask = np.maximum(fire_mask, mask_resized)

            # 应用 EMA 平滑
            if self._mask_ema is not None:
                fire_mask = self._mask_ema.update(fire_mask)

            combined_mask = np.maximum(combined_mask, fire_mask)

        # 处理 smoke: 使用并集 bbox
        if has_smoke:
            x1, y1, x2, y2 = self._smoke_union_bbox
            bbox_mask = np.zeros((h, w), dtype=np.uint8)
            bbox_mask[y1:y2, x1:x2] = 1
            combined_mask = np.maximum(combined_mask, bbox_mask)

        # 应用组合 mask
        if combined_mask.sum() > 0:
            output_frame = resized_frame.copy()
            output_frame[combined_mask == 0] = 0
            roi_type = 'fire+smoke' if (has_fire and has_smoke) else ('fire' if has_fire else 'smoke')
        else:
            output_frame = np.zeros_like(resized_frame)
            roi_type = 'none'

        result = ProcessedFrame(
            frame_idx=self._frame_idx,
            data=output_frame,
            roi_type=roi_type,
            has_fire=has_fire,
            has_smoke=has_smoke,
            detections=det_info
        )

        self._frame_idx += 1
        return result

    def process_video(
        self,
        video_path: str,
        detect_mode: DetectMode = 'all',
        skip_no_detection: bool = True,
        analyze_first: bool = True
    ) -> Generator[ProcessedFrame, None, None]:
        """
        处理视频并生成处理后的帧

        Args:
            video_path: 视频文件路径
            detect_mode: 检测模式 ('all', 'fire', 'smoke')
            skip_no_detection: 是否跳过无检测的帧
            analyze_first: 是否先分析视频（获取 smoke 并集 bbox）

        Yields:
            ProcessedFrame 处理结果
        """
        # 重置状态
        self.reset()

        # 第一步：分析视频获取 smoke 并集 bbox
        if analyze_first and detect_mode != 'fire':
            self.analyze_video_for_smoke_bbox(video_path, detect_mode)

        # 第二步：逐帧处理
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = self.process_frame(frame, detect_mode)

            if skip_no_detection and not result.has_detections:
                continue

            yield result

        cap.release()

    def process_frames(
        self,
        frames: List[np.ndarray],
        detect_mode: DetectMode = 'all',
        skip_no_detection: bool = False
    ) -> Generator[ProcessedFrame, None, None]:
        """
        处理帧列表

        注意：使用此方法前，如果需要 smoke bbox，应先调用 analyze_video_for_smoke_bbox

        Args:
            frames: 帧列表
            detect_mode: 检测模式
            skip_no_detection: 是否跳过无检测的帧

        Yields:
            ProcessedFrame 处理结果
        """
        for frame in frames:
            result = self.process_frame(frame, detect_mode)

            if skip_no_detection and not result.has_detections:
                continue

            yield result

    def reset(self):
        """重置处理器状态（处理新视频时调用）"""
        self._mask_ema = MaskEMA(alpha=self.ema_alpha) if self.use_ema else None
        self._smoke_union_bbox = None
        self._frame_idx = 0

    @property
    def smoke_union_bbox(self) -> Optional[Tuple[int, int, int, int]]:
        """获取当前的 smoke 并集 bbox"""
        return self._smoke_union_bbox

    @smoke_union_bbox.setter
    def smoke_union_bbox(self, value: Optional[Tuple[int, int, int, int]]):
        """设置 smoke 并集 bbox（用于外部预计算的情况）"""
        self._smoke_union_bbox = value
