"""
Detection 数据类和工具函数

提供从 YOLO 检测结果提取和处理的功能。
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Set, Tuple, Optional

from .constants import CLASS_COLORS


@dataclass
class Detection:
    """YOLO 模型的检测结果"""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_yolo_box(cls, box, mask_data: Optional[np.ndarray] = None) -> 'Detection':
        """
        从 YOLO box 结果创建 Detection

        Args:
            box: YOLO box 对象，包含 xyxy, conf, cls 属性
            mask_data: 可选的分割 mask 数组

        Returns:
            Detection 实例
        """
        xyxy = box.xyxy[0].cpu().numpy()
        return cls(
            x1=float(xyxy[0]),
            y1=float(xyxy[1]),
            x2=float(xyxy[2]),
            y2=float(xyxy[3]),
            confidence=float(box.conf[0]),
            class_id=int(box.cls[0]),
            mask=mask_data
        )

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """返回整数格式的 bbox (x1, y1, x2, y2)"""
        return (int(self.x1), int(self.y1), int(self.x2), int(self.y2))

    @property
    def width(self) -> float:
        """返回 bbox 宽度"""
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        """返回 bbox 高度"""
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        """返回 bbox 面积"""
        return self.width * self.height


def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """
    获取类别对应的颜色

    Args:
        class_id: 类别 ID

    Returns:
        BGR 颜色元组
    """
    default_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    return CLASS_COLORS.get(class_id, default_colors[class_id % len(default_colors)])


def extract_detections_from_yolo(
    results,
    frame_shape: Tuple[int, int],
    active_classes: Set[int],
    conf_threshold: float = 0.5
) -> List[Detection]:
    """
    从 YOLO 分割结果提取 Detection 对象

    Args:
        results: YOLO 推理结果
        frame_shape: 帧尺寸 (height, width)
        active_classes: 要包含的类别 ID 集合
        conf_threshold: 最低置信度阈值

    Returns:
        包含 mask（如果有）的 Detection 对象列表
    """
    h, w = frame_shape
    detections = []

    # 处理分割结果（带 mask）
    if hasattr(results, 'masks') and results.masks is not None:
        for box, mask in zip(results.boxes, results.masks.data):
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id in active_classes and conf >= conf_threshold:
                mask_np = mask.cpu().numpy()
                mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                det = Detection.from_yolo_box(box, mask_data=mask_binary)
                detections.append(det)

    # 处理检测结果（仅 bbox）
    elif hasattr(results, 'boxes') and results.boxes is not None:
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            if class_id in active_classes and conf >= conf_threshold:
                det = Detection.from_yolo_box(box)
                detections.append(det)

    return detections


def compute_bbox_union(
    bboxes: List[Tuple[float, float, float, float]]
) -> Optional[Tuple[int, int, int, int]]:
    """
    计算所有 bbox 的并集（覆盖所有输入 bbox 的最小外接矩形）

    Args:
        bboxes: (x1, y1, x2, y2) 元组列表

    Returns:
        并集 bbox (x1, y1, x2, y2)，如果列表为空则返回 None
    """
    if not bboxes:
        return None

    x1_min = min(bbox[0] for bbox in bboxes)
    y1_min = min(bbox[1] for bbox in bboxes)
    x2_max = max(bbox[2] for bbox in bboxes)
    y2_max = max(bbox[3] for bbox in bboxes)

    return (int(x1_min), int(y1_min), int(x2_max), int(y2_max))


def add_bbox_padding(
    bbox: Tuple[int, int, int, int],
    frame_shape: Tuple[int, int],
    padding_ratio: float = 0.05
) -> Tuple[int, int, int, int]:
    """
    给 bbox 添加内边距，并限制在帧边界内

    Args:
        bbox: (x1, y1, x2, y2)
        frame_shape: (height, width)
        padding_ratio: 内边距比例

    Returns:
        添加内边距后的 bbox (x1, y1, x2, y2)
    """
    h, w = frame_shape
    x1, y1, x2, y2 = bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    pad_w = int(bbox_w * padding_ratio)
    pad_h = int(bbox_h * padding_ratio)

    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w, x2 + pad_w)
    y2 = min(h, y2 + pad_h)

    return (x1, y1, x2, y2)


def apply_bbox_mask(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """
    应用 bbox 掩码到帧上，将 bbox 外的像素置为 0

    Args:
        frame: 输入帧 (H, W, C)
        bbox: (x1, y1, x2, y2)

    Returns:
        只保留 bbox 区域的帧
    """
    masked_frame = np.zeros_like(frame)
    x1, y1, x2, y2 = bbox
    masked_frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return masked_frame


def apply_segmentation_mask(
    frame: np.ndarray,
    detections: List[Detection]
) -> np.ndarray:
    """
    应用分割掩码到帧上，只保留检测区域

    Args:
        frame: 原始帧 (H, W, C)
        detections: 包含 mask 的检测列表

    Returns:
        只保留检测区域的帧
    """
    if not detections:
        return np.zeros_like(frame)

    h, w = frame.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    for det in detections:
        if det.mask is not None:
            if det.mask.shape != (h, w):
                mask_resized = cv2.resize(
                    det.mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST
                )
            else:
                mask_resized = det.mask.astype(np.uint8)
            combined_mask = np.maximum(combined_mask, mask_resized)

    masked_frame = frame.copy()
    masked_frame[combined_mask == 0] = 0

    return masked_frame
