"""
Exponential Moving Average for mask smoothing.

Reduces flickering by smoothly transitioning mask regions over time.
"""

import cv2
import numpy as np


class MaskEMA:
    """
    Exponential Moving Average for mask smoothing.

    Reduces flickering by smoothly transitioning mask regions over time.
    """

    def __init__(self, alpha: float = 0.2):
        """
        Args:
            alpha: EMA smoothing factor (0-1)
                   - Lower values = smoother but more lag (e.g., 0.2 ≈ 5 frame lag)
                   - Higher values = more responsive but less smooth (e.g., 0.5 ≈ 2 frame lag)
        """
        self.alpha = alpha
        self.ema_mask = None

    def update(self, current_mask: np.ndarray) -> np.ndarray:
        """
        Update EMA mask and return smoothed mask with edge smoothing.

        Args:
            current_mask: Current binary mask (H, W) with values 0 or 1

        Returns:
            Smoothed binary mask (H, W) with smooth edges
        """
        if self.ema_mask is None:
            self.ema_mask = current_mask.astype(np.float32)
        else:
            self.ema_mask = self.alpha * current_mask.astype(np.float32) + (1 - self.alpha) * self.ema_mask

        smoothed_mask = (self.ema_mask > 0.5).astype(np.uint8)
        smoothed_mask = self._smooth_edges(smoothed_mask)

        return smoothed_mask

    def _smooth_edges(self, mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Smooth mask edges using morphological operations and Gaussian blur.

        Args:
            mask: Binary mask (H, W) with values 0 or 1
            kernel_size: Kernel size for smoothing operations

        Returns:
            Edge-smoothed binary mask
        """
        if mask.sum() == 0:
            return mask

        mask_uint8 = mask.astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_closed = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel)
        mask_blurred = cv2.GaussianBlur(mask_closed, (kernel_size, kernel_size), 0)
        _, mask_smooth = cv2.threshold(mask_blurred, 127, 1, cv2.THRESH_BINARY)

        return mask_smooth.astype(np.uint8)

    def reset(self):
        """Reset EMA state (call this when starting a new video)."""
        self.ema_mask = None
