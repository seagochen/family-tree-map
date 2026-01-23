"""
火焰检测模块

提供完整的火焰检测 Pipeline，串联 ROI 提取和时序分类。
"""

from .pipeline import (
    FireDetectionPipeline,
    DetectionResult,
    CLASS_NAMES,
)

__all__ = [
    'FireDetectionPipeline',
    'DetectionResult',
    'CLASS_NAMES',
]
