"""
Detection dataclass for segmentation results.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Detection:
    """Detection result with bounding box, confidence, class and optional mask."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    mask: Optional[np.ndarray] = None

    @classmethod
    def from_yolo_box(cls, box, mask_data: Optional[np.ndarray] = None) -> 'Detection':
        """Create Detection from YOLO box result."""
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
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)
