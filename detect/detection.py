"""
Detection dataclass for YOLO detect results.
"""

import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class Detection:
    """Detection result with bounding box, confidence and class."""
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int

    @classmethod
    def from_yolo_box(cls, box) -> 'Detection':
        """Create Detection from YOLO box result."""
        xyxy = box.xyxy[0].cpu().numpy()
        return cls(
            x1=float(xyxy[0]),
            y1=float(xyxy[1]),
            x2=float(xyxy[2]),
            y2=float(xyxy[3]),
            confidence=float(box.conf[0]),
            class_id=int(box.cls[0])
        )

    @property
    def bbox(self) -> List[float]:
        """Return bbox as [x1, y1, x2, y2]."""
        return [self.x1, self.y1, self.x2, self.y2]

    @property
    def area(self) -> float:
        """Calculate bounding box area."""
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def center(self) -> tuple:
        """Get center point of bounding box."""
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1
