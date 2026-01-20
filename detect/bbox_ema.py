"""
Exponential Moving Average for bounding box smoothing.

通过 EMA 平滑减少检测框的抖动，使检测结果更稳定。
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TrackedBox:
    """跟踪的边界框状态"""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_id: int
    confidence: float
    age: int = 0  # 未匹配帧数
    hits: int = 1  # 匹配次数


class BBoxEMA:
    """
    Bounding Box 的指数移动平均平滑器。

    通过时序平滑减少检测框的抖动，同时支持简单的跟踪关联。
    """

    def __init__(
        self,
        alpha: float = 0.3,
        iou_threshold: float = 0.3,
        max_age: int = 3,
        min_hits: int = 2
    ):
        """
        Args:
            alpha: EMA 平滑因子 (0-1)
                   - 较低的值 = 更平滑但延迟更大 (如 0.2)
                   - 较高的值 = 响应更快但平滑度较低 (如 0.5)
            iou_threshold: 用于关联检测框的 IoU 阈值
            max_age: 未匹配多少帧后删除跟踪框
            min_hits: 需要多少次匹配后才输出跟踪框
        """
        self.alpha = alpha
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracked_boxes: List[TrackedBox] = []

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        使用新检测结果更新跟踪状态并返回平滑后的检测框。

        Args:
            detections: 当前帧的检测结果列表，每个元素包含:
                - bbox: [x1, y1, x2, y2]
                - class_id: int
                - class_name: str
                - confidence: float

        Returns:
            平滑后的检测结果列表，格式与输入相同
        """
        if not detections:
            # 无检测时，增加所有跟踪框的 age
            self._age_tracked_boxes()
            return self._get_output_detections()

        # 将检测转换为数组格式
        det_bboxes = np.array([d['bbox'] for d in detections])
        det_classes = [d['class_id'] for d in detections]
        det_confs = [d['confidence'] for d in detections]
        det_names = [d.get('class_name', '') for d in detections]

        # 匹配检测与跟踪框
        matched, unmatched_dets, unmatched_tracks = self._associate(
            det_bboxes, det_classes
        )

        # 更新匹配的跟踪框
        for det_idx, track_idx in matched:
            track = self.tracked_boxes[track_idx]
            det_bbox = det_bboxes[det_idx]

            # EMA 平滑
            track.bbox = self.alpha * det_bbox + (1 - self.alpha) * track.bbox
            track.confidence = self.alpha * det_confs[det_idx] + (1 - self.alpha) * track.confidence
            track.age = 0
            track.hits += 1

        # 为未匹配的检测创建新跟踪框
        for det_idx in unmatched_dets:
            self.tracked_boxes.append(TrackedBox(
                bbox=det_bboxes[det_idx].copy(),
                class_id=det_classes[det_idx],
                confidence=det_confs[det_idx]
            ))

        # 增加未匹配跟踪框的 age
        for track_idx in unmatched_tracks:
            self.tracked_boxes[track_idx].age += 1

        # 移除过期的跟踪框
        self.tracked_boxes = [
            t for t in self.tracked_boxes if t.age <= self.max_age
        ]

        return self._get_output_detections(det_names, det_classes)

    def _associate(
        self,
        det_bboxes: np.ndarray,
        det_classes: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        使用 IoU 和类别匹配关联检测与跟踪框。

        Returns:
            matched: [(det_idx, track_idx), ...]
            unmatched_dets: [det_idx, ...]
            unmatched_tracks: [track_idx, ...]
        """
        if not self.tracked_boxes or len(det_bboxes) == 0:
            return [], list(range(len(det_bboxes))), list(range(len(self.tracked_boxes)))

        # 计算 IoU 矩阵
        track_bboxes = np.array([t.bbox for t in self.tracked_boxes])
        iou_matrix = self._compute_iou_matrix(det_bboxes, track_bboxes)

        # 类别不匹配时设置 IoU 为 0
        for i, det_cls in enumerate(det_classes):
            for j, track in enumerate(self.tracked_boxes):
                if det_cls != track.class_id:
                    iou_matrix[i, j] = 0

        # 贪婪匹配
        matched = []
        unmatched_dets = list(range(len(det_bboxes)))
        unmatched_tracks = list(range(len(self.tracked_boxes)))

        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break

            det_idx, track_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            matched.append((det_idx, track_idx))

            if det_idx in unmatched_dets:
                unmatched_dets.remove(det_idx)
            if track_idx in unmatched_tracks:
                unmatched_tracks.remove(track_idx)

            # 将已匹配的行列设为 0
            iou_matrix[det_idx, :] = 0
            iou_matrix[:, track_idx] = 0

        return matched, unmatched_dets, unmatched_tracks

    def _compute_iou_matrix(self, boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
        """计算两组边界框之间的 IoU 矩阵。"""
        n1, n2 = len(boxes1), len(boxes2)
        iou_matrix = np.zeros((n1, n2))

        for i in range(n1):
            for j in range(n2):
                iou_matrix[i, j] = self._compute_iou(boxes1[i], boxes2[j])

        return iou_matrix

    @staticmethod
    def _compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
        """计算两个边界框的 IoU。"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = area1 + area2 - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def _age_tracked_boxes(self):
        """增加所有跟踪框的 age 并移除过期的。"""
        for track in self.tracked_boxes:
            track.age += 1
        self.tracked_boxes = [
            t for t in self.tracked_boxes if t.age <= self.max_age
        ]

    def _get_output_detections(
        self,
        class_names: Optional[List[str]] = None,
        class_ids: Optional[List[int]] = None
    ) -> List[Dict]:
        """获取满足 min_hits 要求的跟踪框作为输出。"""
        # 构建 class_id -> class_name 映射
        name_map = {}
        if class_names and class_ids:
            for cid, name in zip(class_ids, class_names):
                if cid not in name_map and name:
                    name_map[cid] = name

        output = []
        for track in self.tracked_boxes:
            if track.hits >= self.min_hits and track.age == 0:
                output.append({
                    'bbox': track.bbox.tolist(),
                    'class_id': track.class_id,
                    'class_name': name_map.get(track.class_id, f'class_{track.class_id}'),
                    'confidence': track.confidence
                })
        return output

    def reset(self):
        """重置跟踪状态（处理新视频时调用）。"""
        self.tracked_boxes = []
