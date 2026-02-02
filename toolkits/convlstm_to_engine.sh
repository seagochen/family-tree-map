#!/bin/bash
#
# ConvLSTM (TemporalClassifier) 模型转换脚本
# 将 .pth 转换为 .engine（经过 .onnx 中间格式）
#
# 用法:
#   ./convlstm_to_engine.sh                  # 默认: models/convlstm.pth -> models/convlstm.engine
#   ./convlstm_to_engine.sh 10 3             # 指定 seq_len 和 num_classes

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

MODEL_PATH="$PROJECT_ROOT/models/convlstm.pth"
ONNX_PATH="$PROJECT_ROOT/models/convlstm.onnx"
CONFIG_PATH="$PROJECT_ROOT/configs/convlstm.json"
SEQ_LEN=${1:-10}
NUM_CLASSES=${2:-3}

echo "=============================================="
echo "ConvLSTM: .pth -> .onnx -> .engine"
echo "=============================================="
echo "Input:      $MODEL_PATH"
echo "ONNX:       $ONNX_PATH"
echo "Config:     $CONFIG_PATH"
echo "Seq length: $SEQ_LEN"
echo "Classes:    $NUM_CLASSES"
echo ""

# Step 1: Convert .pth to .onnx
echo "[Step 1/2] Converting .pth to .onnx ..."
python3 "$SCRIPT_DIR/convlstm_to_onnx.py" \
    --input "$MODEL_PATH" \
    --output "$ONNX_PATH" \
    --seq-len "$SEQ_LEN" \
    --num-classes "$NUM_CLASSES" \
    --no-dynamic-batch \
    --opset 13

echo ""

# Step 2: Build TensorRT engine
echo "[Step 2/2] Building TensorRT engine ..."
"$SCRIPT_DIR/build_engine.sh" "$CONFIG_PATH"
