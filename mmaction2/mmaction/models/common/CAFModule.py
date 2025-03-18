import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class CAFModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(CAFModule, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  # 使用 Sigmoid 激活函数生成缩放因子
        )

    def forward(self, x):
        batch_size, channels, _, _, _ = x.size()
        # 全局平均池化
        y = self.global_avg_pool(x).view(batch_size, channels)  # 变形为 [batch_size, channels]
        # 通过全连接层生成缩放因子
        y = self.fc(y).view(batch_size, channels, 1, 1, 1)  # 变形为 [batch_size, channels, 1, 1]
        # 对每个通道进行缩放
        return x * y

