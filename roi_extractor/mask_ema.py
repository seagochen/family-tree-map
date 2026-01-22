"""
Mask EMA 平滑器

通过指数移动平均平滑 mask，减少帧间闪烁。
结合形态学操作和高斯模糊实现平滑的边缘过渡。
"""

import cv2
import numpy as np

from .constants import EMA_ALPHA, MASK_BLUR_KERNEL_SIZE


class MaskEMA:
    """
    Mask 的指数移动平均平滑器

    通过时间维度上的平滑减少 mask 区域的闪烁。
    同时使用形态学操作和高斯模糊平滑边缘。
    """

    def __init__(self, alpha: float = EMA_ALPHA, blur_kernel: int = MASK_BLUR_KERNEL_SIZE):
        """
        Args:
            alpha: EMA 平滑因子 (0-1)
                   - 较低的值 = 更平滑但延迟更大 (如 0.2 约 5 帧延迟)
                   - 较高的值 = 响应更快但平滑度较低 (如 0.5 约 2 帧延迟)
            blur_kernel: 高斯模糊核大小
        """
        self.alpha = alpha
        self.blur_kernel = blur_kernel
        self.ema_mask = None

    def update(self, current_mask: np.ndarray) -> np.ndarray:
        """
        更新 EMA mask 并返回平滑后的 mask

        Args:
            current_mask: 当前帧的二值 mask (H, W)，值为 0 或 1

        Returns:
            平滑后的二值 mask (H, W)，带有平滑边缘
        """
        if self.ema_mask is None:
            self.ema_mask = current_mask.astype(np.float32)
        else:
            # EMA 公式: new = alpha * current + (1 - alpha) * old
            self.ema_mask = self.alpha * current_mask.astype(np.float32) + (1 - self.alpha) * self.ema_mask

        # 阈值化得到二值 mask
        smoothed_mask = (self.ema_mask > 0.5).astype(np.uint8)

        # 应用边缘平滑
        smoothed_mask = self._smooth_edges(smoothed_mask)

        return smoothed_mask

    def _smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        """
        使用形态学操作和高斯模糊平滑 mask 边缘

        Args:
            mask: 二值 mask (H, W)，值为 0 或 1

        Returns:
            边缘平滑后的二值 mask
        """
        if mask.sum() == 0:
            return mask

        kernel_size = self.blur_kernel
        mask_uint8 = mask.astype(np.uint8) * 255

        # 形态学闭合操作填充小孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)

        # 高斯模糊平滑边缘
        mask_blurred = cv2.GaussianBlur(mask_closed, (kernel_size, kernel_size), 0)

        # 重新阈值化
        _, mask_smooth = cv2.threshold(mask_blurred, 127, 1, cv2.THRESH_BINARY)

        return mask_smooth.astype(np.uint8)

    def reset(self):
        """重置 EMA 状态（处理新视频时调用）"""
        self.ema_mask = None
