#!/usr/bin/env python3
"""
FireConvLSTM 训练脚本

Usage:
    python train.py --data_root /path/to/data --epochs 10
    python train.py --config config.yaml
"""

import os
import sys
import argparse
import time
import torch
import yaml
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from convlstm import FireConvLSTM, Trainer, build_dataloader


def increment_path(path: Path, exist_ok: bool = False, sep: str = '_') -> Path:
    """
    自动增加路径后缀以避免覆盖

    Args:
        path: 路径
        exist_ok: 如果为True，允许路径存在
        sep: 分隔符

    Returns:
        新路径

    Example:
        runs/exp -> runs/exp_2 -> runs/exp_3
    """
    path = Path(path)

    if not path.exists() or exist_ok:
        return path

    path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')

    # 查找下一个可用编号
    for n in range(2, 100):
        p = f'{path}{sep}{n}{suffix}'
        if not Path(p).exists():
            return Path(p)

    # 如果都存在，使用时间戳
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    return Path(f'{path}{sep}{timestamp}{suffix}')


def parse_args():
    parser = argparse.ArgumentParser(description='Train FireConvLSTM for fire detection')

    # 数据参数
    parser.add_argument('--data_root', type=str, default=None,
                        help='数据集根目录 (包含 data.csv)')
    parser.add_argument('--seq_length', type=int, default=5,
                        help='序列长度 (连续帧数)')
    parser.add_argument('--stride', type=int, default=3,
                        help='滑动窗口步长')
    parser.add_argument('--target_size', type=int, nargs=2, default=[640, 640],
                        help='目标图像尺寸 (H, W)')

    # 训练参数
    parser.add_argument('--batch_size', type=int, default=2,
                        help='批次大小')
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='权重衰减')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='验证集比例')

    # 其他参数
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载线程数')
    parser.add_argument('--device', type=str, default='auto',
                        help='训练设备 (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')

    # 输出目录参数
    parser.add_argument('--project', type=str, default='runs/train',
                        help='保存目录 (default: runs/train)')
    parser.add_argument('--name', type=str, default='exp',
                        help='实验名称 (default: exp)')
    parser.add_argument('--resume', type=str, default=None,
                        help='从实验名称恢复训练 (例如: exp 或 exp_2)')

    # 检测报告参数
    parser.add_argument('--report_interval', type=int, default=0,
                        help='每隔多少轮生成一次检测报告 (0 表示不生成)')
    parser.add_argument('--report_samples', type=int, default=10,
                        help='每次报告抽样的样本数量')

    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                        help='YAML 配置文件路径')

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """加载 YAML 配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    args = parse_args()

    # 如果指定了配置文件，从配置文件加载参数
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)

    # 检查必需参数
    if args.data_root is None:
        raise ValueError("--data_root 参数是必需的，请通过命令行或配置文件提供")

    # 设置输出目录
    if args.resume:
        # 恢复训练时使用指定的实验目录
        save_dir = Path(args.project) / args.resume
        if not save_dir.exists():
            raise ValueError(f"实验目录不存在: {save_dir}")
    else:
        # 新训练时自动递增目录名
        save_dir = Path(increment_path(Path(args.project) / args.name))
    save_dir.mkdir(parents=True, exist_ok=True)

    # 创建子目录
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = save_dir / 'reports'

    print(f"Save directory: {save_dir}")

    # 设置设备
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    print(f"Device: {device}")
    print(f"Data root: {args.data_root}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")

    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # 构建数据加载器
    print("\n加载数据集...")
    train_loader, val_loader = build_dataloader(
        data_root=args.data_root,
        seq_length=args.seq_length,
        stride=args.stride,
        target_size=tuple(args.target_size),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_ratio=args.val_ratio,
        seed=args.seed
    )

    print(f"训练 batches: {len(train_loader)}")
    print(f"验证 batches: {len(val_loader)}")

    # 创建模型
    print("\n创建模型...")
    model = FireConvLSTM()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")

    # 创建训练器
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )

    # 恢复训练
    start_epoch = 0
    if args.resume:
        checkpoint_path = weights_dir / 'last.pth'
        if checkpoint_path.exists():
            print(f"\n从 {checkpoint_path} 恢复训练...")
            start_epoch = trainer.load_checkpoint(str(checkpoint_path))
            print(f"从 epoch {start_epoch} 继续训练")
        else:
            print(f"警告: 未找到检查点 {checkpoint_path}，从头开始训练")

    # 开始训练
    history = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        save_dir=str(weights_dir),
        save_best_only=True,
        report_interval=args.report_interval,
        report_dir=str(reports_dir),
        report_samples=args.report_samples,
        start_epoch=start_epoch
    )

    # 打印最终结果
    print("\n训练历史:")
    print(f"  最终训练准确率: {history['train_acc'][-1]:.4f}")
    print(f"  最终验证准确率: {history['val_acc'][-1]:.4f}")
    print(f"  最佳验证准确率: {trainer.best_val_acc:.4f} (Epoch {trainer.best_epoch})")


if __name__ == '__main__':
    main()
