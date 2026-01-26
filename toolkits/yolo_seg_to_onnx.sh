#!/bin/bash
#
# YOLO Segment 模型转换脚本
# 使用 ultralytics CLI 将 .pt 转换为 .onnx
#
# 用法:
#   ./yolo_seg_to_onnx.sh model.pt
#   ./yolo_seg_to_onnx.sh model.pt 640
#   ./yolo_seg_to_onnx.sh model.pt 640 17

MODEL_PATH=${1:?"Usage: $0 <model.pt> [imgsz] [opset]"}

echo "Converting: $MODEL_PATH"

yolo export model="$MODEL_PATH" format=onnx simplify=False dynamic=True
