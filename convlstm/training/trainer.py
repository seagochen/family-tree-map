"""
时序分类训练器
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import cv2
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
from datetime import datetime

from ..models import TemporalClassifier, heatmap_to_prob


class Trainer:
    """
    TemporalClassifier 训练器

    用于训练时序分类模型，区分动态和静态图像序列

    Args:
        model: TemporalClassifier 模型
        device: 训练设备
        learning_rate: 学习率
        weight_decay: 权重衰减
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # 损失函数
        self.criterion = nn.BCELoss()

        # 优化器
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

        self.best_val_acc = 0
        self.best_epoch = 0

    def train_one_epoch(self, dataloader) -> Tuple[float, float]:
        """
        训练一个 epoch

        Args:
            dataloader: 训练数据加载器

        Returns:
            (avg_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc="Training")
        for frames, labels in pbar:
            frames = frames.to(self.device)  # (B, T, 3, 640, 640)
            labels = labels.float().to(self.device)  # (B,)

            self.optimizer.zero_grad()

            # Forward
            heatmap = self.model(frames)  # (B, 1, 20, 20)
            probs = heatmap_to_prob(heatmap)  # (B,)

            # Loss
            loss = self.criterion(probs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Statistics
            total_loss += loss.item() * frames.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += frames.size(0)

            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{correct/total:.4f}'})

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, dataloader) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        验证

        Args:
            dataloader: 验证数据加载器

        Returns:
            (avg_loss, accuracy, all_probs, all_labels)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_probs = []
        all_labels = []

        for frames, labels in tqdm(dataloader, desc="Validating"):
            frames = frames.to(self.device)
            labels = labels.float().to(self.device)

            heatmap = self.model(frames)
            probs = heatmap_to_prob(heatmap)

            loss = self.criterion(probs, labels)

            total_loss += loss.item() * frames.size(0)
            preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += frames.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return total_loss / total, correct / total, np.array(all_probs), np.array(all_labels)

    @torch.no_grad()
    def generate_detection_report(
        self,
        dataloader,
        report_dir: Path,
        epoch: int,
        num_samples: int = 10
    ):
        """
        生成检测报告

        Args:
            dataloader: 数据加载器
            report_dir: 报告保存目录
            epoch: 当前轮数
            num_samples: 抽样数量 (每类各取一半)
        """
        self.model.eval()

        # 创建报告目录
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        # 创建本次报告的子目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        epoch_report_dir = report_dir / f"epoch_{epoch:03d}_{timestamp}"
        epoch_report_dir.mkdir(parents=True, exist_ok=True)
        heatmap_dir = epoch_report_dir / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)

        # 第一遍：收集所有样本的预测结果（不保存图像和热力图）
        all_sample_metadata = []

        for batch_idx, (frames, labels) in enumerate(tqdm(dataloader, desc=f"Collecting predictions for epoch {epoch}")):
            frames = frames.to(self.device)
            labels = labels.to(self.device)

            heatmaps = self.model(frames)  # (B, 1, 20, 20)
            probs = heatmap_to_prob(heatmaps)  # (B,)
            preds = (probs > 0.5).float()

            # 只保存元数据，不保存图像
            for i in range(frames.size(0)):
                sample_idx = batch_idx * dataloader.batch_size + i

                # 获取样本详细信息
                dataset = dataloader.dataset
                if hasattr(dataset, 'dataset'):  # 处理 Subset
                    actual_idx = dataset.indices[sample_idx]
                    base_dataset = dataset.dataset
                else:
                    actual_idx = sample_idx
                    base_dataset = dataset

                sample_info = base_dataset.get_sample_info(actual_idx)

                all_sample_metadata.append({
                    'sample_idx': sample_idx,
                    'actual_idx': actual_idx,
                    'folder': sample_info['folder'],
                    'frame_files': ', '.join(sample_info['frame_files']),
                    'true_label': labels[i].item(),
                    'true_label_name': 'dynamic' if labels[i].item() == 1 else 'static',
                    'pred_prob': probs[i].item(),
                    'pred_label': preds[i].item(),
                    'pred_label_name': 'dynamic' if preds[i].item() == 1 else 'static',
                    'correct': (preds[i] == labels[i]).item(),
                })

        # 分类抽样
        dynamic_metadata = [s for s in all_sample_metadata if s['true_label'] == 1]
        static_metadata = [s for s in all_sample_metadata if s['true_label'] == 0]

        samples_per_class = num_samples // 2

        # 随机抽样
        import random
        random.seed(epoch)

        selected_dynamic_meta = random.sample(dynamic_metadata, min(samples_per_class, len(dynamic_metadata)))
        selected_static_meta = random.sample(static_metadata, min(samples_per_class, len(static_metadata)))
        selected_metadata = selected_dynamic_meta + selected_static_meta

        # 第二遍：只对选中的样本加载图像和生成热力图
        print(f"\nLoading images for {len(selected_metadata)} selected samples...")
        selected_samples = []

        # 获取数据集引用
        dataset = dataloader.dataset
        if hasattr(dataset, 'dataset'):
            base_dataset = dataset.dataset
        else:
            base_dataset = dataset

        for meta in tqdm(selected_metadata, desc="Loading selected samples"):
            # 重新加载选中的样本
            frames, label = base_dataset[meta['actual_idx']]
            frames = frames.unsqueeze(0).to(self.device)  # (1, T, 3, H, W)

            # 生成热力图
            with torch.no_grad():
                heatmap = self.model(frames)  # (1, 1, 20, 20)

            selected_samples.append({
                **meta,
                'frames': frames.squeeze(0).cpu(),  # (T, 3, H, W)
                'heatmap': heatmap.squeeze(0).cpu()  # (1, 20, 20)
            })

        # 保存 CSV 报告
        csv_path = epoch_report_dir / f"detection_report_epoch_{epoch:03d}.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Sample_Index', 'Folder', 'Frame_Files',
                'True_Label', 'True_Label_Name',
                'Pred_Probability', 'Pred_Label', 'Pred_Label_Name',
                'Correct', 'Heatmap_File'
            ])

            for sample in selected_samples:
                # 保存热力图
                heatmap_file = f"heatmap_sample_{sample['sample_idx']:04d}.png"
                self._save_heatmap_visualization(
                    sample['frames'],
                    sample['heatmap'],
                    heatmap_dir / heatmap_file,
                    sample['true_label_name'],
                    sample['pred_label_name'],
                    sample['pred_prob']
                )

                writer.writerow([
                    sample['sample_idx'],
                    sample['folder'],
                    sample['frame_files'],
                    int(sample['true_label']),
                    sample['true_label_name'],
                    f"{sample['pred_prob']:.4f}",
                    int(sample['pred_label']),
                    sample['pred_label_name'],
                    'Yes' if sample['correct'] else 'No',
                    heatmap_file
                ])

        # 计算并保存准确率统计
        total_correct = sum(s['correct'] for s in all_sample_metadata)
        total_samples = len(all_sample_metadata)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        stats_path = epoch_report_dir / f"statistics_epoch_{epoch:03d}.txt"
        with open(stats_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch {epoch} Detection Report\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Total Samples: {total_samples}\n")
            f.write(f"Correct Predictions: {total_correct}\n")
            f.write(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n\n")
            f.write(f"Dynamic Samples: {len(dynamic_metadata)}\n")
            f.write(f"Static Samples: {len(static_metadata)}\n\n")
            f.write(f"Selected Samples for Report: {len(selected_samples)}\n")
            f.write(f"  - Dynamic: {len(selected_dynamic_meta)}\n")
            f.write(f"  - Static: {len(selected_static_meta)}\n\n")
            f.write(f"Report saved to: {epoch_report_dir}\n")
            f.write(f"CSV file: {csv_path.name}\n")
            f.write(f"Heatmaps saved in: {heatmap_dir.name}/\n")

        print(f"\n{'='*60}")
        print(f"Detection Report Generated for Epoch {epoch}")
        print(f"{'='*60}")
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total Samples: {total_samples}")
        print(f"Report saved to: {epoch_report_dir}")
        print(f"{'='*60}\n")

        return accuracy

    def _save_heatmap_visualization(
        self,
        frames: torch.Tensor,
        heatmap: torch.Tensor,
        save_path: Path,
        true_label: str,
        pred_label: str,
        pred_prob: float
    ):
        """
        保存热力图可视化

        Args:
            frames: (T, 3, H, W) 输入帧序列
            heatmap: (1, H_heat, W_heat) 热力图
            save_path: 保存路径
            true_label: 真实标签
            pred_label: 预测标签
            pred_prob: 预测概率
        """
        # 取最后一帧作为背景
        last_frame = frames[-1].permute(1, 2, 0).numpy()  # (H, W, 3)
        last_frame = (last_frame * 255).astype(np.uint8)

        # 热力图上采样到原图尺寸
        heatmap_np = heatmap.squeeze(0).numpy()  # (H_heat, W_heat)
        heatmap_resized = cv2.resize(heatmap_np, (last_frame.shape[1], last_frame.shape[0]))

        # 归一化到 [0, 255]
        heatmap_resized = (heatmap_resized * 255).astype(np.uint8)

        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # 叠加到原图
        overlay = cv2.addWeighted(last_frame, 0.6, heatmap_colored, 0.4, 0)

        # 添加文本信息
        color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
        cv2.putText(overlay, f"True: {true_label}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, f"Pred: {pred_label} ({pred_prob:.3f})", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 保存
        cv2.imwrite(str(save_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    def fit(
        self,
        train_loader,
        val_loader,
        num_epochs: int = 10,
        save_dir: str = './checkpoints',
        save_best_only: bool = True,
        report_interval: int = 5,
        report_dir: str = './reports',
        report_samples: int = 10,
        start_epoch: int = 0
    ) -> Dict:
        """
        完整训练流程

        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            num_epochs: 训练轮数
            save_dir: 模型保存目录
            save_best_only: 是否只保存最佳模型
            report_interval: 每隔多少轮生成一次检测报告 (0 表示不生成)
            report_dir: 报告保存目录
            report_samples: 每次报告抽样的样本数量
            start_epoch: 起始 epoch（用于恢复训练）

        Returns:
            训练历史
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 60)
        if start_epoch > 0:
            print(f"从 Epoch {start_epoch + 1} 恢复训练")
        else:
            print("开始训练")
        print("=" * 60)

        for epoch in range(start_epoch, num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)

            # 训练
            train_loss, train_acc = self.train_one_epoch(train_loader)

            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader)

            # 学习率调整
            self.scheduler.step(val_loss)

            # 记录
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印结果
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

            # 保存模型
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
                self.best_epoch = epoch + 1

            # 始终保存最新模型
            self.save_checkpoint(save_dir / 'last.pth', epoch + 1, val_acc)

            # 如果是最佳模型，额外保存为 best.pth
            if is_best:
                self.save_checkpoint(save_dir / 'best.pth', epoch + 1, val_acc)
                print(f"  ★ 新最佳模型! Val Acc: {val_acc:.4f}")

            # 生成检测报告
            if report_interval > 0 and (epoch + 1) % report_interval == 0:
                print(f"\n生成第 {epoch + 1} 轮检测报告...")
                self.generate_detection_report(
                    dataloader=val_loader,
                    report_dir=Path(report_dir),
                    epoch=epoch + 1,
                    num_samples=report_samples
                )

        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证准确率: {self.best_val_acc:.4f} (Epoch {self.best_epoch})")
        print("=" * 60)

        return self.history

    def save_checkpoint(self, path: str, epoch: int, val_acc: float):
        """
        保存检查点

        Args:
            path: 保存路径
            epoch: 当前 epoch
            val_acc: 验证准确率
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': self.best_val_acc,
            'history': self.history
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """
        加载检查点

        Args:
            path: 检查点路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0)
        self.history = checkpoint.get('history', self.history)

        return checkpoint.get('epoch', 0)
