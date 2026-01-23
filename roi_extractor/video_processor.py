"""
ROI 提取 Pipeline

处理一批连续图片，提取火焰/烟雾的感兴趣区域 (ROI)：
- fire (class 0): 使用分割 mask + EMA 平滑
- smoke (class 2): 使用该批次所有 smoke bbox 的并集
- person (class 1): 忽略

处理流程：
1. 对所有帧运行检测
2. 计算 smoke 的 union bbox（整个批次）
3. 对每帧应用 fire mask (EMA 平滑) + smoke mask (union bbox)
4. 不需要的像素置 0
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Literal, Any

from .constants import (
    CLASS_NAMES,
    TARGET_CLASSES,
    SEGMENT_CLASSES,
    BBOX_CLASSES,
    INFERENCE_SIZE,
    DEFAULT_CONFIDENCE_THRESHOLD,
    EMA_ALPHA,
    BBOX_PADDING_RATIO,
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
    frame_idx: int                           # 帧索引（在批次中的位置）
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
class BatchResult:
    """批次处理结果"""
    frames: List[ProcessedFrame]             # 处理后的帧列表
    smoke_union_bbox: Optional[Tuple[int, int, int, int]]  # smoke 并集 bbox
    frames_with_fire: int                    # 包含 fire 的帧数
    frames_with_smoke: int                   # 包含 smoke 的帧数


class ROIPipeline:
    """
    ROI 提取 Pipeline

    处理一批连续图片，提取火焰/烟雾区域，不需要的像素置 0。

    使用示例:
        model = YOLO("model.pt")
        pipeline = ROIPipeline(model)

        frames = [frame1, frame2, ..., frame10]  # 10 张连续图片
        result = pipeline.process_batch(frames)

        for processed in result.frames:
            cv2.imshow("ROI", processed.data)
    """

    def __init__(
        self,
        model: Any,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        use_ema: bool = True,
        ema_alpha: float = EMA_ALPHA,
        bbox_padding: float = BBOX_PADDING_RATIO,
    ):
        """
        初始化 Pipeline

        Args:
            model: YOLO 模型对象（由外部传入）
            confidence_threshold: 置信度阈值
            use_ema: 是否使用 EMA 平滑 fire mask
            ema_alpha: EMA 平滑因子
            bbox_padding: smoke bbox 内边距比例
        """
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.use_ema = use_ema
        self.ema_alpha = ema_alpha
        self.bbox_padding = bbox_padding

    # ==================== 主要接口 ====================

    def process_batch(
        self,
        frames: List[np.ndarray],
        detect_mode: DetectMode = 'all',
        skip_no_detection: bool = False
    ) -> BatchResult:
        """
        处理一批连续图片

        Args:
            frames: 连续图片列表 (BGR)
            detect_mode: 检测模式 ('all', 'fire', 'smoke')
            skip_no_detection: 是否跳过无检测的帧

        Returns:
            BatchResult 批次处理结果
        """
        if not frames:
            return BatchResult(
                frames=[],
                smoke_union_bbox=None,
                frames_with_fire=0,
                frames_with_smoke=0
            )

        # 1. 预处理所有帧
        resized_frames = [self._preprocess_frame(f) for f in frames]
        h, w = resized_frames[0].shape[:2]

        # 2. 对所有帧运行检测
        all_detections = [self._run_detection(f) for f in resized_frames]
        all_filtered = [self._filter_detections(d, detect_mode) for d in all_detections]

        # 3. 计算 smoke union bbox（整个批次）
        smoke_union_bbox = self._compute_batch_smoke_union(all_filtered, (h, w))

        # 4. 初始化 EMA
        mask_ema = MaskEMA(alpha=self.ema_alpha) if self.use_ema else None

        # 5. 处理每帧
        results = []
        frames_with_fire = 0
        frames_with_smoke = 0

        for idx, (resized_frame, (fire_dets, smoke_dets)) in enumerate(zip(resized_frames, all_filtered)):
            has_fire = len(fire_dets) > 0
            has_smoke = len(smoke_dets) > 0 and smoke_union_bbox is not None

            if has_fire:
                frames_with_fire += 1
            if has_smoke:
                frames_with_smoke += 1

            # 构建 mask
            fire_mask = self._build_fire_mask(fire_dets, (h, w), mask_ema) if has_fire else np.zeros((h, w), dtype=np.uint8)
            smoke_mask = self._build_smoke_mask(smoke_union_bbox, (h, w)) if has_smoke else np.zeros((h, w), dtype=np.uint8)

            # 应用 mask
            output_frame, roi_type = self._apply_mask_to_frame(resized_frame, fire_mask, smoke_mask)

            result = ProcessedFrame(
                frame_idx=idx,
                data=output_frame,
                roi_type=roi_type,
                has_fire=has_fire,
                has_smoke=has_smoke,
                detections=self._build_detection_info(fire_dets, smoke_dets)
            )

            if not skip_no_detection or result.has_detections:
                results.append(result)

        return BatchResult(
            frames=results,
            smoke_union_bbox=smoke_union_bbox,
            frames_with_fire=frames_with_fire,
            frames_with_smoke=frames_with_smoke
        )

    # ==================== 辅助方法 ====================

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """预处理帧（resize 到推理尺寸）"""
        return cv2.resize(frame, INFERENCE_SIZE, interpolation=cv2.INTER_LINEAR)

    def _run_detection(self, frame: np.ndarray) -> List[Detection]:
        """运行 YOLO 检测"""
        h, w = frame.shape[:2]
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)[0]
        return extract_detections_from_yolo(results, (h, w), TARGET_CLASSES, self.confidence_threshold)

    def _filter_detections(
        self,
        detections: List[Detection],
        detect_mode: DetectMode
    ) -> Tuple[List[Detection], List[Detection]]:
        """根据检测模式筛选 fire 和 smoke 检测"""
        if detect_mode == 'fire':
            return [d for d in detections if d.class_id in SEGMENT_CLASSES], []
        elif detect_mode == 'smoke':
            return [], [d for d in detections if d.class_id in BBOX_CLASSES]
        else:  # 'all'
            fire_dets = [d for d in detections if d.class_id in SEGMENT_CLASSES]
            smoke_dets = [d for d in detections if d.class_id in BBOX_CLASSES]
            return fire_dets, smoke_dets

    def _compute_batch_smoke_union(
        self,
        all_filtered: List[Tuple[List[Detection], List[Detection]]],
        shape: Tuple[int, int]
    ) -> Optional[Tuple[int, int, int, int]]:
        """计算整个批次的 smoke union bbox"""
        all_smoke_bboxes = []
        for _, smoke_dets in all_filtered:
            for det in smoke_dets:
                all_smoke_bboxes.append((det.x1, det.y1, det.x2, det.y2))

        if not all_smoke_bboxes:
            return None

        union_bbox = compute_bbox_union(all_smoke_bboxes)
        if union_bbox is not None:
            union_bbox = add_bbox_padding(union_bbox, shape, self.bbox_padding)

        return union_bbox

    def _build_fire_mask(
        self,
        fire_dets: List[Detection],
        shape: Tuple[int, int],
        mask_ema: Optional[MaskEMA]
    ) -> np.ndarray:
        """从 fire 检测构建 mask（含 EMA 平滑）"""
        h, w = shape
        fire_mask = np.zeros((h, w), dtype=np.uint8)

        for det in fire_dets:
            if det.mask is not None:
                if det.mask.shape != (h, w):
                    mask_resized = cv2.resize(
                        det.mask.astype(np.uint8), (w, h),
                        interpolation=cv2.INTER_NEAREST
                    )
                else:
                    mask_resized = det.mask.astype(np.uint8)
                fire_mask = np.maximum(fire_mask, mask_resized)

        # 应用 EMA 平滑
        if mask_ema is not None:
            fire_mask = mask_ema.update(fire_mask)

        return fire_mask

    def _build_smoke_mask(
        self,
        smoke_union_bbox: Optional[Tuple[int, int, int, int]],
        shape: Tuple[int, int]
    ) -> np.ndarray:
        """从 union bbox 构建 smoke mask"""
        h, w = shape
        smoke_mask = np.zeros((h, w), dtype=np.uint8)

        if smoke_union_bbox is not None:
            x1, y1, x2, y2 = smoke_union_bbox
            smoke_mask[y1:y2, x1:x2] = 1

        return smoke_mask

    def _apply_mask_to_frame(
        self,
        frame: np.ndarray,
        fire_mask: np.ndarray,
        smoke_mask: np.ndarray
    ) -> Tuple[np.ndarray, str]:
        """合并 mask 并应用到帧（不需要的像素置 0）"""
        combined_mask = np.maximum(fire_mask, smoke_mask)
        has_fire = fire_mask.sum() > 0
        has_smoke = smoke_mask.sum() > 0

        if combined_mask.sum() > 0:
            output_frame = frame.copy()
            output_frame[combined_mask == 0] = 0
            if has_fire and has_smoke:
                roi_type = 'fire+smoke'
            elif has_fire:
                roi_type = 'fire'
            else:
                roi_type = 'smoke'
        else:
            output_frame = np.zeros_like(frame)
            roi_type = 'none'

        return output_frame, roi_type

    def _build_detection_info(
        self,
        fire_dets: List[Detection],
        smoke_dets: List[Detection]
    ) -> List[Dict]:
        """构建检测信息列表"""
        return [
            {
                'class_id': d.class_id,
                'class_name': CLASS_NAMES[d.class_id],
                'confidence': d.confidence,
                'bbox': d.bbox
            }
            for d in (fire_dets + smoke_dets)
        ]
