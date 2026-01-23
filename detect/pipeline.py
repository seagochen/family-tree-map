"""
火焰检测总 Pipeline

串联 ROI 提取和时序分类两个步骤：
1. ROIPipeline: 使用 YOLO 分割模型提取火焰/烟雾 ROI 区域
2. TemporalClassifier: 使用 ConvLSTM 对 ROI 序列进行分类

使用示例:
    from ultralytics import YOLO
    from detect.pipeline import FireDetectionPipeline
    from convlstm.models.temporal_classifier import create_model

    # 加载模型
    yolo_model = YOLO("yolo_seg.pt")
    classifier = create_model("convlstm.pt", num_classes=3)

    # 创建 Pipeline
    pipeline = FireDetectionPipeline(yolo_model, classifier)

    # 处理一批连续图片
    frames = [frame1, frame2, ..., frame10]
    result = pipeline.predict(frames)

    print(f"预测类别: {result.predicted_class}")
    print(f"类别概率: {result.class_probs}")
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Any, Literal

from roi_extractor.video_processor import ROIPipeline, BatchResult, DetectMode
from roi_extractor.constants import INFERENCE_SIZE
from convlstm.models.temporal_classifier import heatmap_to_prob, heatmap_to_pred


@dataclass
class DetectionResult:
    """火焰检测结果"""
    predicted_class: int                    # 预测的类别索引
    class_probs: np.ndarray                 # 各类别概率 (num_classes,)
    heatmap: np.ndarray                     # 分类热力图 (num_classes, 20, 20)
    roi_result: BatchResult                 # ROI 提取结果
    has_detection: bool                     # 是否检测到火焰/烟雾


# 类别名称映射（可根据实际情况调整）
CLASS_NAMES = {
    0: 'static',    # 静态/无火焰
    1: 'dynamic',   # 动态/有火焰
    2: 'negative',  # 负样本
}


class FireDetectionPipeline:
    """
    火焰检测总 Pipeline

    处理流程:
    1. 接收一批连续图片（如 10 帧）
    2. ROIPipeline 提取 ROI 区域（火焰/烟雾），不需要的像素置 0
    3. 将处理后的帧转换为 tensor
    4. TemporalClassifier 进行时序分类
    5. 返回分类结果
    """

    def __init__(
        self,
        yolo_model: Any,
        classifier: Any,
        device: str = 'cuda',
        confidence_threshold: float = 0.5,
        use_ema: bool = True,
        ema_alpha: float = 0.3,
        bbox_padding: float = 0.1,
    ):
        """
        初始化 Pipeline

        Args:
            yolo_model: YOLO 分割模型对象
            classifier: TemporalClassifier 模型对象
            device: 推理设备 ('cuda' 或 'cpu')
            confidence_threshold: YOLO 检测置信度阈值
            use_ema: 是否使用 EMA 平滑 fire mask
            ema_alpha: EMA 平滑因子
            bbox_padding: smoke bbox 内边距比例
        """
        self.roi_pipeline = ROIPipeline(
            model=yolo_model,
            confidence_threshold=confidence_threshold,
            use_ema=use_ema,
            ema_alpha=ema_alpha,
            bbox_padding=bbox_padding,
        )

        self.classifier = classifier
        self.device = device

        # 将 classifier 移到指定设备
        if hasattr(classifier, 'to'):
            self.classifier = classifier.to(device)
            self.classifier.eval()

    def predict(
        self,
        frames: List[np.ndarray],
        detect_mode: DetectMode = 'all',
    ) -> DetectionResult:
        """
        对一批连续图片进行火焰检测

        Args:
            frames: 连续图片列表 (BGR, 任意尺寸)
            detect_mode: 检测模式 ('all', 'fire', 'smoke')

        Returns:
            DetectionResult 检测结果
        """
        if not frames:
            raise ValueError("frames 不能为空")

        # 1. ROI 提取
        roi_result = self.roi_pipeline.process_batch(frames, detect_mode)

        # 检查是否有检测结果
        has_detection = roi_result.frames_with_fire > 0 or roi_result.frames_with_smoke > 0

        # 2. 准备 classifier 输入
        # 从 ROI 结果中提取处理后的帧
        processed_frames = [f.data for f in roi_result.frames]

        # 如果帧数不足，填充
        if len(processed_frames) == 0:
            # 没有检测到任何目标，使用全零帧
            h, w = INFERENCE_SIZE[1], INFERENCE_SIZE[0]
            processed_frames = [np.zeros((h, w, 3), dtype=np.uint8) for _ in frames]

        # 3. 转换为 tensor
        # (T, H, W, 3) BGR -> (1, T, 3, H, W) RGB normalized
        tensor_input = self._frames_to_tensor(processed_frames)

        # 4. 分类推理
        with torch.no_grad():
            heatmap = self.classifier(tensor_input)
            # heatmap: (1, num_classes, 20, 20)

            probs = heatmap_to_prob(heatmap)  # (1,) or (1, num_classes)
            pred = heatmap_to_pred(heatmap)   # (1,)

        # 5. 构建结果
        heatmap_np = heatmap[0].cpu().numpy()  # (num_classes, 20, 20)

        if probs.dim() == 1:
            # 二分类
            class_probs = np.array([1 - probs[0].item(), probs[0].item()])
        else:
            # 多分类
            class_probs = probs[0].cpu().numpy()

        return DetectionResult(
            predicted_class=pred[0].item(),
            class_probs=class_probs,
            heatmap=heatmap_np,
            roi_result=roi_result,
            has_detection=has_detection,
        )

    def _frames_to_tensor(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        将帧列表转换为 classifier 输入 tensor

        Args:
            frames: 处理后的帧列表 (H, W, 3) BGR

        Returns:
            tensor: (1, T, 3, H, W) RGB normalized [0, 1]
        """
        # BGR -> RGB, 归一化
        rgb_frames = []
        for frame in frames:
            rgb = frame[..., ::-1].copy()  # BGR -> RGB
            rgb_normalized = rgb.astype(np.float32) / 255.0
            rgb_frames.append(rgb_normalized)

        # Stack: (T, H, W, 3)
        stacked = np.stack(rgb_frames, axis=0)

        # Transpose: (T, H, W, 3) -> (T, 3, H, W)
        transposed = stacked.transpose(0, 3, 1, 2)

        # Add batch dim: (1, T, 3, H, W)
        tensor = torch.from_numpy(transposed).unsqueeze(0)

        return tensor.to(self.device)

    def get_class_name(self, class_idx: int) -> str:
        """获取类别名称"""
        return CLASS_NAMES.get(class_idx, f'class_{class_idx}')
