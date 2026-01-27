from torch import nn
import torch
import torch.nn.functional as F
from .convlstm import ConvLSTM

class TemporalClassifier(nn.Module):
    """
    时序分类 ConvLSTM 模型

    用于区分动态图像序列和静态图像序列
    - Dynamic: 具有时序运动模式的图像（如闪烁、扩散、移动等）
    - Static: 无明显运动模式的静态图像

    输入: (Batch, Time, 3, 640, 640) - 连续 T 帧 RGB 图像
    输出:
        - num_classes=1: (Batch, 1, 20, 20) - 二分类概率热力图，值域 [0, 1]
        - num_classes>1: (Batch, num_classes, 20, 20) - 多分类 logits（原始输出，未经 softmax）

    热力图每个格子对应原图 32x32 像素区域
    热力图坐标 (i,j) → 原图区域 (i*32 : i*32+32, j*32 : j*32+32)

    Args:
        num_classes: 分类类别数量
                     - 1: 二分类模式（向后兼容），使用 Sigmoid
                     - 2+: 多分类模式，使用 Softmax
                     默认为 3（static, dynamic, negative）
    """

    def __init__(self, num_classes: int = 3):
        super(TemporalClassifier, self).__init__()

        self.num_classes = num_classes

        # 假设输入已经是 YOLO-Seg 预处理后的 (Image * Mask) 数据

        # 空间编码器: 640x640 → 20x20 (每层 stride=2, 共5层: 640→320→160→80→40→20)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # (B*T, 3, 640, 640) → (B*T, 16, 320, 320)
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # → (B*T, 32, 160, 160)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # → (B*T, 64, 80, 80)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # → (B*T, 128, 40, 40)
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1), # → (B*T, 64, 20, 20)
            nn.ReLU()
        )

        # 时序处理器: ConvLSTM 提取时序特征（运动模式、闪烁、扩散等动态特征）
        # 输入: (B, T, 64, 20, 20) → 输出: (B, T, 32, 20, 20)
        self.temporal_layer = ConvLSTM(input_dim=64, hidden_dim=32, kernel_size=3)

        # 分类器: 生成最终热力图
        # 输入: (B, 32, 20, 20) → 输出: (B, num_classes, 20, 20)
        self.final_classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x_sequence):
        """
        Args:
            x_sequence: (Batch, Time, 3, 640, 640) 连续视频帧

        Returns:
            - num_classes=1: (Batch, 1, 20, 20) 二分类概率热力图，值域 [0, 1]
            - num_classes>1: (Batch, num_classes, 20, 20) 多分类概率热力图
        """
        batch, time, c, h, w = x_sequence.shape
        # x_sequence: (B, T, 3, 640, 640)

        # Step 1: 合并 Batch 和 Time 维度，一起过 CNN
        x_reshaped = x_sequence.view(batch * time, c, h, w)
        # x_reshaped: (B*T, 3, 640, 640)

        features = self.encoder(x_reshaped)
        # features: (B*T, 64, 20, 20)

        # Step 2: 恢复时序维度，送入 ConvLSTM
        features_seq = features.view(batch, time, 64, 20, 20)
        # features_seq: (B, T, 64, 20, 20)

        lstm_out, _ = self.temporal_layer(features_seq)
        # lstm_out: (B, T, 32, 20, 20)

        # Step 3: 取最后一帧的特征图 (包含了所有历史信息)
        last_frame_feat = lstm_out[:, -1, :, :, :]
        # last_frame_feat: (B, 32, 20, 20)

        # Step 4: 生成热力图
        result_map = self.final_classifier(last_frame_feat)
        # result_map: (B, num_classes, 20, 20)

        # Step 5: 应用激活函数
        if self.num_classes == 1:
            # 二分类模式: 使用 Sigmoid (BCELoss 需要概率输入)
            return torch.sigmoid(result_map)
            # output: (B, 1, 20, 20), 值域 [0, 1]
        else:
            # 多分类模式: 返回 logits (CrossEntropyLoss 内部会处理 softmax)
            return result_map
            # output: (B, num_classes, 20, 20), 原始 logits


def create_model(checkpoint_path: str = None, num_classes: int = 3):
    """
    创建 TemporalClassifier 模型

    Args:
        checkpoint_path: 可选的预训练权重路径
        num_classes: 分类类别数量（默认 3）

    Returns:
        TemporalClassifier 模型实例
    """
    model = TemporalClassifier(num_classes=num_classes)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    return model


def heatmap_to_prob(heatmap: torch.Tensor) -> torch.Tensor:
    """
    将热力图转换为分类分数

    Args:
        heatmap: 热力图张量
                 - 二分类: (B, 1, 20, 20) 概率值
                 - 多分类: (B, num_classes, 20, 20) logits

    Returns:
        - 二分类: (B,) 正类概率值
        - 多分类: (B, num_classes) 每个类别的 logits（用于 CrossEntropyLoss）

    策略: 取热力图每个通道的空间最大值
    """
    # heatmap: (B, C, H, W)
    # 对每个通道取空间最大值
    probs = heatmap.amax(dim=(2, 3))  # (B, C)

    if probs.shape[1] == 1:
        # 二分类: 返回 (B,)
        return probs.squeeze(1)
    else:
        # 多分类: 返回 (B, num_classes)
        return probs


def heatmap_to_pred(heatmap: torch.Tensor) -> torch.Tensor:
    """
    将热力图转换为预测类别

    Args:
        heatmap: 热力图张量 (B, num_classes, 20, 20)

    Returns:
        (B,) 预测的类别索引
    """
    probs = heatmap_to_prob(heatmap)  # (B,) or (B, num_classes)

    if probs.dim() == 1:
        # 二分类
        return (probs > 0.5).long()
    else:
        # 多分类: 取 argmax
        return probs.argmax(dim=1)
