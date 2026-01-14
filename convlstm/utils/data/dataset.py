import os
import re
import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from torch.utils.data import random_split


class SequenceDataset(Dataset):
    """
    时序分类数据集

    将文件夹中的连续帧图片转换为时序样本，用于区分动态和静态图像序列

    输入:
        - data_root: 数据集根目录 (包含 data.csv)
        - seq_length: 每个样本的帧数 (Time 维度)
        - stride: 滑动窗口步长
        - target_size: 输出图像尺寸 (H, W)

    输出 (per sample):
        - frames: (seq_length, 3, H, W) 连续帧张量
        - label: 0=static(静态), 1=dynamic(动态)
    """

    def __init__(
        self,
        data_root: str,
        seq_length: int = 5,
        stride: int = 1,
        target_size: Tuple[int, int] = (640, 640),
        transform=None
    ):
        """
        Args:
            data_root: 数据集根目录，需包含 data.csv
            seq_length: 每个样本包含的连续帧数
            stride: 滑动窗口步长 (stride=1 表示最大重叠)
            target_size: 输出图像尺寸 (height, width)
            transform: 可选的数据增强
        """
        self.data_root = Path(data_root)
        self.seq_length = seq_length
        self.stride = stride
        self.target_size = target_size
        self.transform = transform

        # 加载 data.csv
        csv_path = self.data_root / "data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"data.csv not found in {data_root}")

        self.metadata = pd.read_csv(csv_path)

        # 构建样本索引
        # samples: List of (folder_path, start_idx, label)
        self.samples = []
        self.folder_frames = {}  # 缓存每个文件夹的帧列表

        self._build_samples()

    def _extract_frame_number(self, filename: str) -> int:
        """
        从文件名提取帧编号
        例如: 'xxx_frame_000123.png' -> 123
        """
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        # 如果没有 frame_ 前缀，尝试提取最后的数字
        match = re.search(r'(\d+)\.[^.]+$', filename)
        if match:
            return int(match.group(1))
        return -1

    def _get_sorted_frames(self, folder_path: Path) -> List[Path]:
        """
        获取文件夹中按帧编号排序的图片列表
        """
        if folder_path in self.folder_frames:
            return self.folder_frames[folder_path]

        # 获取所有图片文件
        image_extensions = {'.png', '.jpg', '.jpeg'}
        frames = [
            f for f in folder_path.iterdir()
            if f.suffix.lower() in image_extensions
        ]

        # 按帧编号排序
        frames.sort(key=lambda x: self._extract_frame_number(x.name))

        self.folder_frames[folder_path] = frames
        return frames

    def _build_samples(self):
        """
        构建所有样本的索引

        对每个文件夹，使用滑动窗口生成样本:
        - 窗口大小: seq_length
        - 步长: stride
        """
        for _, row in self.metadata.iterrows():
            folder_name = row['folder_name']
            folder_type = row['type']  # 'dynamic' or 'static'

            folder_path = self.data_root / folder_name
            if not folder_path.exists():
                print(f"Warning: folder not found: {folder_path}")
                continue

            # label: dynamic=1 (动态), static=0 (静态)
            label = 1 if folder_type == 'dynamic' else 0

            # 获取排序后的帧列表
            frames = self._get_sorted_frames(folder_path)
            num_frames = len(frames)

            if num_frames < self.seq_length:
                print(f"Warning: folder {folder_name} has only {num_frames} frames, "
                      f"need at least {self.seq_length}")
                continue

            # 滑动窗口生成样本
            # 样本数 = (num_frames - seq_length) // stride + 1
            for start_idx in range(0, num_frames - self.seq_length + 1, self.stride):
                self.samples.append((folder_path, start_idx, label))

        print(f"Built {len(self.samples)} samples from {len(self.metadata)} folders")

    def _load_and_preprocess(self, image_path: Path) -> np.ndarray:
        """
        加载图片并预处理

        Returns:
            (3, H, W) 归一化后的图像张量
        """
        # 读取图片 (BGR)
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        if img.shape[:2] != self.target_size:
            img = cv2.resize(img, (self.target_size[1], self.target_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        # (H, W, C) -> (C, H, W)
        img = img.transpose(2, 0, 1)

        # 归一化到 [0, 1]
        img = img.astype(np.float32) / 255.0

        return img

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        获取一个样本

        Returns:
            frames: (seq_length, 3, H, W) 连续帧张量
            label: 0=static, 1=dynamic
        """
        folder_path, start_idx, label = self.samples[idx]

        # 获取该文件夹的帧列表
        all_frames = self.folder_frames[folder_path]

        # 加载连续 seq_length 帧
        frames = []
        for i in range(self.seq_length):
            frame_path = all_frames[start_idx + i]
            frame = self._load_and_preprocess(frame_path)
            frames.append(frame)

        # Stack: (seq_length, 3, H, W)
        frames = np.stack(frames, axis=0)
        frames = torch.from_numpy(frames)

        if self.transform:
            frames = self.transform(frames)

        return frames, label

    def get_sample_info(self, idx: int) -> Dict:
        """
        获取样本的详细信息 (用于调试)
        """
        folder_path, start_idx, label = self.samples[idx]
        all_frames = self.folder_frames[folder_path]

        return {
            'folder': folder_path.name,
            'start_idx': start_idx,
            'end_idx': start_idx + self.seq_length - 1,
            'label': label,
            'label_name': 'dynamic' if label == 1 else 'static',
            'frame_files': [all_frames[start_idx + i].name for i in range(self.seq_length)]
        }

    def get_statistics(self) -> Dict:
        """
        获取数据集统计信息
        """
        labels = [s[2] for s in self.samples]
        dynamic_count = sum(labels)
        static_count = len(labels) - dynamic_count

        return {
            'total_samples': len(self.samples),
            'dynamic_samples': dynamic_count,
            'static_samples': static_count,
            'seq_length': self.seq_length,
            'stride': self.stride,
            'target_size': self.target_size,
            'folders': len(self.metadata)
        }


# 向后兼容别名
FireSequenceDataset = SequenceDataset


def build_dataloader(
        data_root: str,
        seq_length: int = 5,
        stride: int = 1,
        target_size: Tuple[int, int] = (640, 640),
        batch_size: int = 4,
        num_workers: int = 4,
        transform=None,
        val_ratio: float = 0.2,
        seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    构建数据加载器

    Args:
        data_root: 数据根目录
        seq_length: 序列长度
        stride: 步长
        target_size: 目标尺寸 (H, W)
        batch_size: 批次大小
        num_workers: 数据加载线程数
        transform: 数据增强变换
        val_ratio: 验证集比例
        seed: 随机种子

    Returns:
        (train_loader, val_loader)
    """

    # 创建数据集
    dataset = SequenceDataset(
        data_root=data_root,
        seq_length=seq_length,
        stride=stride,
        target_size=target_size,
        transform=transform
    )

    # 设置随机种子
    generator = torch.Generator().manual_seed(seed)

    # 计算划分数量
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size

    # 随机划分
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader
