#!/usr/bin/env python3
"""
火灾检测完整流程脚本

支持视频文件和 RTSP 视频流输入，实现完整的火灾检测流程:
1. 从视频/RTSP流中每次取10帧
2. 使用 VideoSegmentProcessor 进行火焰/烟雾检测
3. 如果检测到目标，使用 TemporalClassifier 判断动态/静态
4. 输出检测结果

Usage:
    # 从视频文件检测
    python fire_detect.py --source video.mp4 --segment_model segment.pt --temporal_model temporal.pth

    # 从 RTSP 流检测
    python fire_detect.py --source rtsp://192.168.1.100:554/stream --segment_model segment.pt --temporal_model temporal.pth

    # 显示检测结果窗口
    python fire_detect.py --source video.mp4 --segment_model segment.pt --temporal_model temporal.pth --show

    # 保存检测结果视频
    python fire_detect.py --source video.mp4 --segment_model segment.pt --temporal_model temporal.pth --output results/
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from convlstm import TemporalClassifier, create_model, heatmap_to_prob
from segment import VideoSegmentProcessor, ProcessedFrame, ROIMode


class DetectionResult(Enum):
    """检测结果类型"""
    NO_DETECTION = "no_detection"      # 未检测到火焰/烟雾
    FIRE_DETECTED = "fire_detected"    # 检测到火灾（动态）
    STATIC_SUSPECTED = "static_suspected"  # 静态疑似（可能是图片或反光）


@dataclass
class FireDetectionOutput:
    """火灾检测输出"""
    result: DetectionResult
    confidence: float
    message: str
    frame_count: int
    detections: List[dict]
    heatmap: Optional[np.ndarray] = None


def parse_args():
    parser = argparse.ArgumentParser(description='Fire detection pipeline')

    # 输入源
    parser.add_argument('--source', type=str, required=True,
                        help='视频文件路径或 RTSP 流地址')

    # 模型参数
    parser.add_argument('--segment_model', type=str, required=True,
                        help='YOLO 分割模型路径 (.pt)')
    parser.add_argument('--temporal_model', type=str, required=True,
                        help='时序分类模型路径 (.pth)')

    # 检测参数
    parser.add_argument('--seq_length', type=int, default=10,
                        help='每次处理的帧数 (默认: 10)')
    parser.add_argument('--frame_size', type=int, nargs=2, default=[640, 640],
                        help='帧尺寸 (H, W) (默认: 640 640)')
    parser.add_argument('--segment_conf', type=float, default=0.5,
                        help='分割模型置信度阈值 (默认: 0.5)')
    parser.add_argument('--temporal_threshold', type=float, default=0.5,
                        help='时序分类阈值 (默认: 0.5)')

    # 设备
    parser.add_argument('--device', type=str, default='auto',
                        help='推理设备 (cuda/cpu/auto)')

    # 输出控制
    parser.add_argument('--show', action='store_true',
                        help='显示检测结果窗口')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (保存结果视频和截图)')
    parser.add_argument('--verbose', action='store_true',
                        help='详细输出模式')

    # 流控制
    parser.add_argument('--loop', action='store_true',
                        help='循环播放视频 (仅对视频文件有效)')
    parser.add_argument('--skip_frames', type=int, default=0,
                        help='每次检测后跳过的帧数 (用于加速处理)')

    return parser.parse_args()


class FireDetector:
    """火灾检测器"""

    def __init__(
        self,
        segment_model_path: str,
        temporal_model_path: str,
        seq_length: int = 10,
        frame_size: Tuple[int, int] = (640, 640),
        segment_conf: float = 0.5,
        temporal_threshold: float = 0.5,
        device: str = 'auto'
    ):
        """
        初始化火灾检测器

        Args:
            segment_model_path: YOLO 分割模型路径
            temporal_model_path: 时序分类模型路径
            seq_length: 每次处理的帧数
            frame_size: 帧尺寸 (H, W)
            segment_conf: 分割模型置信度阈值
            temporal_threshold: 时序分类阈值
            device: 推理设备
        """
        self.seq_length = seq_length
        self.frame_size = frame_size
        self.temporal_threshold = temporal_threshold

        # 设置设备
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        print(f"[INFO] 使用设备: {self.device}")

        # 加载分割模型
        print(f"[INFO] 加载分割模型: {segment_model_path}")
        self.segment_processor = VideoSegmentProcessor(
            model_path=segment_model_path,
            confidence_threshold=segment_conf,
            use_ema=True
        )

        # 加载时序分类模型
        print(f"[INFO] 加载时序分类模型: {temporal_model_path}")
        self.temporal_model = create_model(temporal_model_path)
        self.temporal_model = self.temporal_model.to(self.device)
        self.temporal_model.eval()

        total_params = sum(p.numel() for p in self.temporal_model.parameters())
        print(f"[INFO] 时序模型参数量: {total_params:,}")

    def process_frames(self, frames: List[np.ndarray]) -> FireDetectionOutput:
        """
        处理一批帧进行火灾检测

        Args:
            frames: 帧列表，每帧为 (H, W, 3) BGR 格式

        Returns:
            FireDetectionOutput 检测结果
        """
        if len(frames) < self.seq_length:
            return FireDetectionOutput(
                result=DetectionResult.NO_DETECTION,
                confidence=0.0,
                message=f"帧数不足: {len(frames)} < {self.seq_length}",
                frame_count=len(frames),
                detections=[]
            )

        # Step 1: 调整帧尺寸
        resized_frames = []
        for frame in frames:
            if frame.shape[:2] != self.frame_size:
                resized = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]),
                                     interpolation=cv2.INTER_LINEAR)
            else:
                resized = frame
            resized_frames.append(resized)

        # Step 2: 使用分割模型检测火焰/烟雾
        processed_frames = []
        all_detections = []
        has_detection = False

        for processed in self.segment_processor.process_frames(
            resized_frames, skip_no_detection=False, force_mode=ROIMode.MASK
        ):
            if processed is not None and processed.has_detections:
                has_detection = True
                processed_frames.append(processed)
                all_detections.extend(processed.detections)
            else:
                # 无检测时使用空帧
                processed_frames.append(processed)

        # Step 3: 检查是否有检测目标
        if not has_detection:
            return FireDetectionOutput(
                result=DetectionResult.NO_DETECTION,
                confidence=0.0,
                message="未检测到火焰或烟雾",
                frame_count=len(frames),
                detections=[]
            )

        # Step 4: 准备时序模型输入
        # 使用处理后的帧（带掩码）进行时序分析
        temporal_input = []
        for pf in processed_frames:
            if pf is not None:
                # BGR -> RGB, HWC -> CHW, normalize
                frame_rgb = cv2.cvtColor(pf.data, cv2.COLOR_BGR2RGB)
                frame_chw = frame_rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
                temporal_input.append(frame_chw)
            else:
                # 无检测帧使用零填充
                temporal_input.append(np.zeros((3, self.frame_size[0], self.frame_size[1]),
                                               dtype=np.float32))

        # 确保有足够的帧
        while len(temporal_input) < self.seq_length:
            temporal_input.append(temporal_input[-1] if temporal_input else
                                  np.zeros((3, self.frame_size[0], self.frame_size[1]),
                                           dtype=np.float32))

        # 只取最后 seq_length 帧
        temporal_input = temporal_input[-self.seq_length:]

        # Step 5: 时序分类推理
        frames_tensor = torch.from_numpy(np.stack(temporal_input, axis=0))
        frames_tensor = frames_tensor.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

        with torch.no_grad():
            heatmap = self.temporal_model(frames_tensor)  # (1, 1, 20, 20)
            prob = heatmap_to_prob(heatmap).item()

        heatmap_np = heatmap[0, 0].cpu().numpy()

        # Step 6: 判断结果
        if prob > self.temporal_threshold:
            return FireDetectionOutput(
                result=DetectionResult.FIRE_DETECTED,
                confidence=prob,
                message=f"[警告] 发现火灾! 动态特征置信度: {prob:.3f}",
                frame_count=len(frames),
                detections=all_detections,
                heatmap=heatmap_np
            )
        else:
            return FireDetectionOutput(
                result=DetectionResult.STATIC_SUSPECTED,
                confidence=prob,
                message=f"未发现火灾，疑似烟雾或火焰的图片 (静态置信度: {1-prob:.3f})",
                frame_count=len(frames),
                detections=all_detections,
                heatmap=heatmap_np
            )


class VideoSource:
    """视频源封装类，支持视频文件和 RTSP 流"""

    def __init__(self, source: str, loop: bool = False):
        """
        初始化视频源

        Args:
            source: 视频文件路径或 RTSP 流地址
            loop: 是否循环播放（仅对视频文件有效）
        """
        self.source = source
        self.loop = loop
        self.is_rtsp = source.lower().startswith('rtsp://')
        self.cap = None
        self.frame_count = 0
        self.fps = 0

        self._open()

    def _open(self):
        """打开视频源"""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频源: {self.source}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        if not self.is_rtsp:
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read_frames(self, count: int) -> List[np.ndarray]:
        """
        读取指定数量的帧

        Args:
            count: 要读取的帧数

        Returns:
            帧列表
        """
        frames = []
        for _ in range(count):
            ret, frame = self.cap.read()
            if not ret:
                if self.loop and not self.is_rtsp:
                    # 循环播放视频
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                else:
                    break
            frames.append(frame)
        return frames

    def skip_frames(self, count: int):
        """跳过指定数量的帧"""
        for _ in range(count):
            ret, _ = self.cap.read()
            if not ret:
                break

    def release(self):
        """释放视频源"""
        if self.cap is not None:
            self.cap.release()

    @property
    def is_opened(self) -> bool:
        """检查视频源是否打开"""
        return self.cap is not None and self.cap.isOpened()


def create_visualization(
    frame: np.ndarray,
    output: FireDetectionOutput,
    frame_size: Tuple[int, int]
) -> np.ndarray:
    """
    创建可视化结果

    Args:
        frame: 原始帧
        output: 检测结果
        frame_size: 帧尺寸

    Returns:
        可视化帧
    """
    vis = frame.copy()
    if vis.shape[:2] != frame_size:
        vis = cv2.resize(vis, (frame_size[1], frame_size[0]))

    # 绘制结果状态
    if output.result == DetectionResult.FIRE_DETECTED:
        color = (0, 0, 255)  # 红色
        status = "FIRE DETECTED!"
    elif output.result == DetectionResult.STATIC_SUSPECTED:
        color = (0, 165, 255)  # 橙色
        status = "Static Suspected"
    else:
        color = (0, 255, 0)  # 绿色
        status = "No Detection"

    # 绘制状态栏
    cv2.rectangle(vis, (0, 0), (frame_size[1], 60), (0, 0, 0), -1)
    cv2.putText(vis, status, (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, color, 2)
    cv2.putText(vis, f"Conf: {output.confidence:.3f}", (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 如果有热力图，叠加显示
    if output.heatmap is not None:
        heatmap_resized = cv2.resize(output.heatmap, (frame_size[1], frame_size[0]))
        heatmap_colored = cv2.applyColorMap(
            (heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET
        )
        vis = cv2.addWeighted(vis, 0.7, heatmap_colored, 0.3, 0)

    return vis


def main():
    args = parse_args()

    # 创建输出目录
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化检测器
    print("=" * 60)
    print("火灾检测系统启动")
    print("=" * 60)

    detector = FireDetector(
        segment_model_path=args.segment_model,
        temporal_model_path=args.temporal_model,
        seq_length=args.seq_length,
        frame_size=tuple(args.frame_size),
        segment_conf=args.segment_conf,
        temporal_threshold=args.temporal_threshold,
        device=args.device
    )

    # 打开视频源
    print(f"\n[INFO] 打开视频源: {args.source}")
    video_source = VideoSource(args.source, loop=args.loop)
    print(f"[INFO] 视频 FPS: {video_source.fps:.1f}")
    if video_source.frame_count > 0:
        print(f"[INFO] 视频总帧数: {video_source.frame_count}")

    # 视频写入器
    video_writer = None
    if args.output:
        output_video_path = output_dir / "detection_result.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(output_video_path),
            fourcc,
            video_source.fps / (args.seq_length + args.skip_frames),
            (args.frame_size[1], args.frame_size[0])
        )

    # 开始检测循环
    print("\n[INFO] 开始检测...")
    print("-" * 60)

    batch_count = 0
    fire_count = 0
    static_count = 0
    no_detection_count = 0

    try:
        while video_source.is_opened:
            # 读取帧
            frames = video_source.read_frames(args.seq_length)
            if len(frames) == 0:
                if not args.loop or video_source.is_rtsp:
                    break
                continue

            batch_count += 1
            start_time = time.time()

            # 执行检测
            output = detector.process_frames(frames)
            elapsed = time.time() - start_time

            # 统计
            if output.result == DetectionResult.FIRE_DETECTED:
                fire_count += 1
            elif output.result == DetectionResult.STATIC_SUSPECTED:
                static_count += 1
            else:
                no_detection_count += 1

            # 输出结果
            if args.verbose or output.result != DetectionResult.NO_DETECTION:
                print(f"[Batch {batch_count:04d}] {output.message} ({elapsed:.2f}s)")

            # 可视化
            if args.show or video_writer:
                vis_frame = create_visualization(
                    frames[-1], output, tuple(args.frame_size)
                )

                if video_writer:
                    video_writer.write(vis_frame)

                if args.show:
                    cv2.imshow('Fire Detection', vis_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\n[INFO] 用户中断检测")
                        break

            # 保存火灾截图
            if output.result == DetectionResult.FIRE_DETECTED and args.output:
                screenshot_path = output_dir / f"fire_batch{batch_count:04d}.jpg"
                cv2.imwrite(str(screenshot_path), frames[-1])

            # 跳帧处理
            if args.skip_frames > 0:
                video_source.skip_frames(args.skip_frames)

    except KeyboardInterrupt:
        print("\n[INFO] 检测被中断")
    finally:
        # 清理资源
        video_source.release()
        if video_writer:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    # 输出统计
    print("\n" + "=" * 60)
    print("检测统计")
    print("=" * 60)
    print(f"  处理批次: {batch_count}")
    print(f"  火灾检出: {fire_count} 次")
    print(f"  静态疑似: {static_count} 次")
    print(f"  无检测: {no_detection_count} 次")

    if args.output:
        print(f"\n  结果已保存到: {args.output}")

    print("\n[INFO] 检测完成")


if __name__ == '__main__':
    main()
