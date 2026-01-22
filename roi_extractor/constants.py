"""
ROI 提取模块常量定义

类别配置（对应 data.yaml: fire, person, smoke）：
- fire (class 0): 使用分割 mask + EMA 平滑
- person (class 1): 忽略（负样本）
- smoke (class 2): 使用 bbox 并集策略
"""

from typing import Dict, Set, Tuple

# =============================================================================
# 类别配置 (对应 data.yaml)
# =============================================================================

CLASS_NAMES: Dict[int, str] = {
    0: "fire",
    1: "person",  # 负样本，处理时忽略
    2: "smoke"
}

# 目标类别（需要处理的类别）
TARGET_CLASSES: Set[int] = {0, 2}  # fire, smoke

# 使用分割 mask 的类别（fire）
SEGMENT_CLASSES: Set[int] = {0}

# 使用 bbox 的类别（smoke）
BBOX_CLASSES: Set[int] = {2}

# 负样本类别（忽略）
NEGATIVE_CLASSES: Set[int] = {1}  # person

# 类别颜色（BGR 格式，用于可视化）
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 255),         # fire - 红色
    1: (0, 255, 0),         # person - 绿色（仅用于可视化）
    2: (0, 255, 255),       # smoke - 黄色
}

# =============================================================================
# 处理参数
# =============================================================================

# YOLO 推理尺寸
INFERENCE_SIZE: Tuple[int, int] = (640, 640)

# 默认置信度阈值
DEFAULT_CONFIDENCE_THRESHOLD: float = 0.5

# =============================================================================
# ROI 处理参数
# =============================================================================

# EMA 平滑因子（用于 mask 平滑）
# 较低的值 = 更平滑但延迟更大
# 较高的值 = 响应更快但平滑度较低
EMA_ALPHA: float = 0.2

# 高斯模糊核大小（用于 mask 边缘平滑）
MASK_BLUR_KERNEL_SIZE: int = 5

# Bbox 内边距比例（用于 smoke bbox）
BBOX_PADDING_RATIO: float = 0.05  # 5%

# 视频分析采样率（每 N 帧采样一次）
DEFAULT_SAMPLE_RATE: int = 5
