#!/usr/bin/env python3
"""
ConvLSTM (TemporalClassifier) 模型转换工具

将 .pth 格式的 TemporalClassifier 模型转换为 ONNX 格式

用法:
    python convlstm_to_onnx.py --input model.pth --output model.onnx
    python convlstm_to_onnx.py --input model.pth --output model.onnx --num-classes 3 --seq-len 8
"""

import argparse
import sys
from pathlib import Path

import torch

# 添加项目根目录到 sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from convlstm.models.temporal_classifier import TemporalClassifier


def export_to_onnx(
    input_path: str,
    output_path: str,
    num_classes: int = 3,
    seq_len: int = 8,
    batch_size: int = 1,
    opset_version: int = 17,
    dynamic_batch: bool = True,
    dynamic_seq: bool = False,
):
    """
    将 TemporalClassifier 模型转换为 ONNX 格式

    Args:
        input_path: 输入 .pth 文件路径
        output_path: 输出 .onnx 文件路径
        num_classes: 分类类别数量
        seq_len: 序列长度 (时间步数)
        batch_size: 批次大小
        opset_version: ONNX opset 版本
        dynamic_batch: 是否支持动态 batch size
        dynamic_seq: 是否支持动态序列长度
    """
    print(f"Loading model from: {input_path}")

    # 创建模型
    model = TemporalClassifier(num_classes=num_classes)

    # 加载权重
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Model loaded successfully (num_classes={num_classes})")

    # 创建 dummy input: (Batch, Time, 3, 640, 640)
    dummy_input = torch.randn(batch_size, seq_len, 3, 640, 640)
    print(f"Input shape: {dummy_input.shape}")

    # 配置动态轴
    dynamic_axes = {}
    if dynamic_batch:
        dynamic_axes['input'] = {0: 'batch_size'}
        dynamic_axes['output'] = {0: 'batch_size'}
    if dynamic_seq:
        if 'input' not in dynamic_axes:
            dynamic_axes['input'] = {}
        dynamic_axes['input'][1] = 'seq_len'

    # 导出 ONNX
    print(f"Exporting to ONNX (opset_version={opset_version})...")

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes=dynamic_axes if dynamic_axes else None,
        )

    # 尝试将外部数据合并到单个文件（如果生成了 .data 文件）
    data_file = Path(output_path).with_suffix('.onnx.data')
    if data_file.exists():
        try:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data, load_external_data_for_model

            print("Merging external data into single file...")
            onnx_model = onnx.load(output_path, load_external_data=True)

            # 保存为单个文件（移除外部数据引用）
            for tensor in onnx_model.graph.initializer:
                if tensor.HasField("data_location"):
                    tensor.ClearField("data_location")

            onnx.save(onnx_model, output_path)
            data_file.unlink()  # 删除 .data 文件
            print("Merged into single .onnx file")
        except Exception as e:
            print(f"Note: Could not merge external data: {e}")
            print("Both .onnx and .onnx.data files are needed for deployment")

    print(f"ONNX model saved to: {output_path}")

    # 验证导出的模型
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")
    except ImportError:
        print("Warning: onnx package not installed, skipping validation")
    except Exception as e:
        print(f"Warning: ONNX validation failed: {e}")

    # 打印模型信息
    print("\n--- Model Info ---")
    print(f"Input:  (batch, {seq_len}, 3, 640, 640)")
    print(f"Output: (batch, {num_classes}, 20, 20)")
    if dynamic_axes:
        print(f"Dynamic axes: {dynamic_axes}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert TemporalClassifier (.pth) to ONNX format'
    )
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Input .pth file path'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='Output .onnx file path'
    )
    parser.add_argument(
        '--num-classes', '-c',
        type=int,
        default=3,
        help='Number of classes (default: 3)'
    )
    parser.add_argument(
        '--seq-len', '-s',
        type=int,
        default=8,
        help='Sequence length / time steps (default: 8)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=1,
        help='Batch size for dummy input (default: 1)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=17,
        help='ONNX opset version (default: 17)'
    )
    parser.add_argument(
        '--no-dynamic-batch',
        action='store_true',
        help='Disable dynamic batch size'
    )
    parser.add_argument(
        '--dynamic-seq',
        action='store_true',
        help='Enable dynamic sequence length (may not work with all runtimes)'
    )

    args = parser.parse_args()

    export_to_onnx(
        input_path=args.input,
        output_path=args.output,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        opset_version=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
        dynamic_seq=args.dynamic_seq,
    )


if __name__ == '__main__':
    main()
