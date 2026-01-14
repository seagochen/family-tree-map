from torch import nn
import torch
from .convlstm import ConvLSTM

class TemporalClassifier(nn.Module):
    """
    时序分类 ConvLSTM 模型

    用于区分动态图像序列和静态图像序列
    - Dynamic: 具有时序运动模式的图像（如闪烁、扩散、移动等）
    - Static: 无明显运动模式的静态图像

    输入: (Batch, Time, 3, 640, 640) - 连续 T 帧 RGB 图像
    输出: (Batch, 1, 20, 20) - 概率热力图

    热力图每个格子对应原图 32x32 像素区域
    热力图坐标 (i,j) → 原图区域 (i*32 : i*32+32, j*32 : j*32+32)
    """

    def __init__(self):
        super(TemporalClassifier, self).__init__()

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
        # 输入: (B, 32, 20, 20) → 输出: (B, 1, 20, 20)
        self.final_classifier = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x_sequence):
        """
        Args:
            x_sequence: (Batch, Time, 3, 640, 640) 连续视频帧

        Returns:
            (Batch, 1, 20, 20) 概率热力图，值域 [0, 1]
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
        # result_map: (B, 1, 20, 20)

        return torch.sigmoid(result_map)
        # output: (B, 1, 20, 20), 值域 [0, 1]


# 保持向后兼容
FireConvLSTM = TemporalClassifier


def create_model(checkpoint_path: str = None):
    """
    创建 TemporalClassifier 模型

    Args:
        checkpoint_path: 可选的预训练权重路径

    Returns:
        TemporalClassifier 模型实例
    """
    model = TemporalClassifier()

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    return model


def heatmap_to_prob(heatmap: torch.Tensor) -> torch.Tensor:
    """
    将热力图转换为分类概率

    Args:
        heatmap: (B, 1, 20, 20) 热力图

    Returns:
        (B,) 概率值

    策略: 取热力图最大值作为分类概率
    """
    return heatmap.amax(dim=(1, 2, 3))
