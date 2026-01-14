#!/usr/bin/env python3
"""
时序分类推理/检测脚本

用于区分动态图像序列和静态图像序列

Usage:
    # 检测单个视频文件夹
    python detect.py --source /path/to/frames --weights best_model.pth

    # 检测多个文件夹
    python detect.py --source /path/to/data --weights best_model.pth --batch

    # 保存可视化结果
    python detect.py --source /path/to/frames --weights best_model.pth --save_viz --output ./results
"""

import os
import sys
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from convlstm import TemporalClassifier, create_model, heatmap_to_prob


def parse_args():
    parser = argparse.ArgumentParser(description='Temporal classification inference')

    parser.add_argument('--source', type=str, required=True,
                        help='输入源: 图片文件夹路径')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='序列长度 (连续帧数)')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                        help='目标图像尺寸 (H, W)')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='分类阈值')
    parser.add_argument('--device', type=str, default='auto',
                        help='推理设备 (cuda/cpu/auto)')

    # 输出参数
    parser.add_argument('--output', type=str, default='./results',
                        help='输出目录')
    parser.add_argument('--save_viz', action='store_true',
                        help='保存可视化结果')
    parser.add_argument('--batch', action='store_true',
                        help='批量处理模式 (source 为包含多个子文件夹的目录)')

    return parser.parse_args()


def get_sorted_frames(folder_path: Path) -> List[Path]:
    """获取文件夹中按编号排序的图片列表"""
    import re

    def extract_frame_number(filename: str) -> int:
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\.[^.]+$', filename)
        if match:
            return int(match.group(1))
        return -1

    image_extensions = {'.png', '.jpg', '.jpeg'}
    frames = [f for f in folder_path.iterdir() if f.suffix.lower() in image_extensions]
    frames.sort(key=lambda x: extract_frame_number(x.name))
    return frames


def load_and_preprocess(image_path: Path, target_size: Tuple[int, int]) -> np.ndarray:
    """加载并预处理图片"""
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Failed to load image: {image_path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.shape[:2] != target_size:
        img = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR)

    img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
    return img


def detect_folder(
    model: torch.nn.Module,
    folder_path: Path,
    seq_length: int,
    target_size: Tuple[int, int],
    device: torch.device,
    threshold: float = 0.5
) -> dict:
    """
    检测单个文件夹

    Returns:
        {
            'folder': str,
            'prediction': str,  # 'dynamic' or 'static'
            'probability': float,
            'heatmap': np.ndarray,  # (20, 20)
            'frames_used': List[str]
        }
    """
    frames_list = get_sorted_frames(folder_path)

    if len(frames_list) < seq_length:
        return {
            'folder': folder_path.name,
            'prediction': 'error',
            'probability': 0.0,
            'heatmap': None,
            'frames_used': [],
            'error': f'Not enough frames: {len(frames_list)} < {seq_length}'
        }

    # 加载最后 seq_length 帧
    start_idx = len(frames_list) - seq_length
    frames = []
    for i in range(seq_length):
        frame = load_and_preprocess(frames_list[start_idx + i], target_size)
        frames.append(frame)

    frames = np.stack(frames, axis=0)  # (T, 3, H, W)
    frames_tensor = torch.from_numpy(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

    # 推理
    model.eval()
    with torch.no_grad():
        heatmap = model(frames_tensor)  # (1, 1, 20, 20)
        prob = heatmap_to_prob(heatmap).item()

    prediction = 'dynamic' if prob > threshold else 'static'
    heatmap_np = heatmap[0, 0].cpu().numpy()

    return {
        'folder': folder_path.name,
        'prediction': prediction,
        'probability': prob,
        'heatmap': heatmap_np,
        'frames_used': [frames_list[start_idx + i].name for i in range(seq_length)]
    }


def visualize_result(
    result: dict,
    folder_path: Path,
    target_size: Tuple[int, int],
    output_path: Path
):
    """可视化检测结果"""
    import matplotlib.pyplot as plt

    if result['heatmap'] is None:
        return

    # 加载最后一帧
    frames_list = get_sorted_frames(folder_path)
    last_frame_path = frames_list[-1]
    last_frame = cv2.imread(str(last_frame_path))
    last_frame = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)

    if last_frame.shape[:2] != target_size:
        last_frame = cv2.resize(last_frame, (target_size[1], target_size[0]))

    # 上采样热力图
    heatmap = result['heatmap']
    heatmap_resized = cv2.resize(heatmap, (target_size[1], target_size[0]))

    # 创建可视化
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    axes[0].imshow(last_frame)
    axes[0].set_title('Original Frame')
    axes[0].axis('off')

    # 热力图
    im = axes[1].imshow(heatmap, cmap='hot', vmin=0, vmax=1)
    axes[1].set_title(f'Heatmap (max={heatmap.max():.3f})')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    # 叠加
    axes[2].imshow(last_frame)
    axes[2].imshow(heatmap_resized, cmap='hot', alpha=0.5, vmin=0, vmax=1)
    pred_color = 'red' if result['prediction'] == 'dynamic' else 'green'
    axes[2].set_title(
        f"Prediction: {result['prediction'].upper()} ({result['probability']:.3f})",
        color=pred_color,
        fontweight='bold'
    )
    axes[2].axis('off')

    plt.suptitle(f"Folder: {result['folder']}", fontsize=14)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # 设置设备
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Model: {args.weights}")

    # 加载模型
    print("\n加载模型...")
    model = create_model(args.weights)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    # 获取要处理的文件夹列表
    source_path = Path(args.source)
    target_size = tuple(args.target_size)

    if args.batch:
        # 批量模式: 处理子文件夹
        folders = [f for f in source_path.iterdir() if f.is_dir()]
        print(f"\n找到 {len(folders)} 个子文件夹")
    else:
        # 单文件夹模式
        folders = [source_path]

    # 检测
    results = []
    print("\n开始检测...")
    print("-" * 60)

    for folder in folders:
        result = detect_folder(
            model=model,
            folder_path=folder,
            seq_length=args.seq_length,
            target_size=target_size,
            device=device,
            threshold=args.threshold
        )
        results.append(result)

        # 打印结果
        if 'error' in result:
            print(f"[ERROR] {result['folder']}: {result['error']}")
        else:
            status = '[D]' if result['prediction'] == 'dynamic' else '[S]'
            print(f"{status} {result['folder']}: {result['prediction'].upper()} "
                  f"(prob={result['probability']:.3f})")

        # 保存可视化
        if args.save_viz and result['heatmap'] is not None:
            output_path = Path(args.output) / f"{result['folder']}_result.png"
            visualize_result(result, folder, target_size, output_path)

    # 统计
    print("-" * 60)
    print("\n检测统计:")

    valid_results = [r for r in results if 'error' not in r]
    dynamic_count = sum(1 for r in valid_results if r['prediction'] == 'dynamic')
    static_count = len(valid_results) - dynamic_count

    print(f"  总计: {len(results)} 个文件夹")
    print(f"  成功: {len(valid_results)}")
    print(f"  Dynamic (动态): {dynamic_count}")
    print(f"  Static (静态): {static_count}")

    if args.save_viz:
        print(f"\n可视化结果已保存到: {args.output}")


if __name__ == '__main__':
    main()
