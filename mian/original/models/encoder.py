import torch
import torch.nn as nn
from .transformer import TransformerEncoder
from .SSA import SpectralStructureAlignment1d


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.params = params

        # --- [设置] 核心维度 ---
        feature_dim = 256
        # ---------------------

        external_centers = getattr(params, 'external_centers', None)

        self.epoch_encoder = EpochEncoder2Layer(
            self.params,
            out_channels=feature_dim,
            external_centers=external_centers
        )

        self.seq_encoder = TransformerEncoder(
            seq_length=20,
            num_layers=1,
            num_heads=4,
            hidden_dim=feature_dim,  # 256
            mlp_dim=feature_dim,  # 256
            dropout=self.params.dropout,
            attention_dropout=self.params.dropout,
        )

        # [!!! 关键修改 !!!]
        # 之前是: self.fc_mu = nn.Linear(feature_dim, 512)
        # 现在必须改为 256 -> 256，与 Decoder 的入口对齐
        self.fc_mu = nn.Linear(feature_dim, feature_dim)

    def forward(self, x, domain_ids=None):
        bz = x.shape[0]
        x = x.view(bz * 20, 2, 3000)

        # 把 domain_ids 传进去
        x = self.epoch_encoder(x, domain_ids=domain_ids)  # (B*20, 256)
        x_epoch = x.view(bz, 20, -1)  # (B, 20, 256)

        x_seq = self.seq_encoder(x_epoch)  # (B, 20, 256)

        mu = self.fc_mu(x_seq)  # (B, 20, 256) <--- 现在输出是 256 了
        return mu


# EpochEncoder2Layer 保持结构不变，防止破坏 state_dict
class EpochEncoder2Layer(nn.Module):
    def __init__(self, params, out_channels=256, external_centers=None):
        super(EpochEncoder2Layer, self).__init__()

        self.ssam1 = SpectralStructureAlignment1d(
            num_channels=64,
            F_len=5,
            momentum=1e-2,
            external_centers=external_centers
        )

        self.layer1 = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=49, stride=6, bias=False, padding=24),
            nn.GELU(),
            self.ssam1,
            nn.MaxPool1d(kernel_size=9, stride=2, padding=4),
            nn.Dropout(params.dropout),
        )

        self.layer2 = nn.Sequential(
            nn.Conv1d(64, out_channels, kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
        )

        self.avg = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, domain_ids=None):
        # 手动迭代 layer1，只有遇到 ssam1 才传入 domain_ids
        for module in self.layer1:
            if isinstance(module, SpectralStructureAlignment1d):
                x = module(x, domain_ids=domain_ids)
            else:
                x = module(x)
                
        x = self.layer2(x)
        x = self.avg(x).squeeze(-1)
        return x