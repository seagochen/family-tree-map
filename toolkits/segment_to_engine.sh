#!/bin/bash
#
# YOLO Segment 模型转换脚本
# 将 .pt 转换为 .engine（经过 .onnx 中间格式）
#
# 用法:
#   ./yolo_seg_to_engine.sh                  # 默认: models/segment.pt -> models/segment.engine

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="$PROJECT_ROOT/models/segment.pt"
ONNX_PATH="$PROJECT_ROOT/models/segment.onnx"
CONFIG_PATH="$PROJECT_ROOT/configs/segment.json"

echo "=============================================="
echo "YOLO Segment: .pt -> .onnx -> .engine"
echo "=============================================="
echo "Input:  $MODEL_PATH"
echo "ONNX:   $ONNX_PATH"
echo "Config: $CONFIG_PATH"
echo ""

# Step 1: Convert .pt to .onnx
echo "[Step 1/2] Converting .pt to .onnx ..."
yolo export model="$MODEL_PATH" format=onnx simplify=False dynamic=False

echo ""

# Step 2: Build TensorRT engine
echo "[Step 2/2] Building TensorRT engine ..."
"$SCRIPT_DIR/build_engine.sh" "$CONFIG_PATH"
