# original/models/decoder.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from .transformer import TransformerEncoder


class Decoder(nn.Module):
    def __init__(self, params):
        super(Decoder, self).__init__()
        self.params = params

        # 简化版 Decoder (4层)，输入维度适配 Encoder 的 256
        # 目标: 从 (256, 1) 还原回 (2, 3000)
        # 总上采样倍率 = 10 * 10 * 6 * 5 = 3000

        # --- Block 1: 256 -> 128 ---
        # Length: 1 -> 10 (Stride 10)
        self.tconv1 = nn.ConvTranspose1d(256, 128, kernel_size=10, stride=10, padding=0, bias=False)
        self.norm1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(params.dropout)

        # --- Block 2: 128 -> 64 ---
        # Length: 10 -> 100 (Stride 10)
        self.tconv2 = nn.ConvTranspose1d(128, 64, kernel_size=10, stride=10, padding=0, bias=False)
        self.norm2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(params.dropout)

        # --- Block 3: 64 -> 32 ---
        # Length: 100 -> 600 (Stride 6)
        self.tconv3 = nn.ConvTranspose1d(64, 32, kernel_size=6, stride=6, padding=0, bias=False)
        self.norm3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(params.dropout)

        # --- Final Block: 32 -> 2 ---
        # Length: 600 -> 3000 (Stride 5)
        self.tconv_final = nn.ConvTranspose1d(32, 2, kernel_size=5, stride=5, padding=0, bias=False)

    def forward(self, x):
        bz = x.shape[0]

        # [FIX] 动态获取特征维度，避免硬编码报错
        feature_dim = x.shape[2]
        # x shape 预期为 (bz, 20, 256)
        x = x.view(bz * 20, feature_dim, 1)

        # Block 1
        x = self.tconv1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.drop1(x)

        # Block 2
        x = self.tconv2(x)
        x = self.norm2(x)
        x = F.gelu(x)
        x = self.drop2(x)

        # Block 3
        x = self.tconv3(x)
        x = self.norm3(x)
        x = F.gelu(x)
        x = self.drop3(x)

        # Final Block
        x = self.tconv_final(x)

        return x.view(bz, 20, 2, 3000)