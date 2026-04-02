
# losses/psd_geometric_loss.py
#
# 严格遵循 psdnorm.py 中的计算逻辑来实现对齐损失。
# 1. 使用 _encode_spectral_structure 计算PSD (经过 unfold 向量化加速)。
# 2. 使用 _build_consensus_spectrum 计算几何均值中心。
# 3. 使用 MSE 计算各域PSD与中心PSD的距离。

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

    # --- Helper functions copied directly from original/models/psdnorm.py ---


def _hann_window(F_len: int, device, dtype):
    # 频谱结构编码的窗函数，做成 L2=1，避免能量引入额外尺度
    w = torch.hann_window(F_len, periodic=True, dtype=dtype, device=device)
    w = w / torch.linalg.vector_norm(w)
    return w


# _segment_indices 在向量化实现中不再显式调用，但逻辑通过 unfold 保持一致
def _segment_indices(L: int, F_len: int, hop: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    if hop is None:
        hop = max(1, F_len // 2)  # 默认 50% overlap
    if L < F_len:
        # 序列过短也要至少做一次结构编码
        return torch.tensor([0]), 1
    num = 1 + (L - F_len) // hop
    starts = torch.arange(0, num * hop, hop)
    return starts, hop


# --- End of copied functions ---


class PSDGeometricLoss(nn.Module):
    """
    计算特征 (B, T, D) 在时间维度 (T) 上的功率谱密度(PSD)的几何中心对齐损失。

    严格遵循 `original/models/psdnorm.py` 的计算逻辑。
    优化：使用 unfold 和 批处理 替代 Python 循环以大幅提升计算速度。
    """

    def __init__(self, F_len: int = 9, hop: Optional[int] = None, eps: float = 1e-8):
        """
        初始化。
        T=20 (来自 original/models/encoder.py) 是一个短序列。
        F_len=9, hop=4 (默认) 是一个合理的选择。
        """
        super().__init__()
        self.F_len = int(F_len)
        self.hop = hop
        self.eps = float(eps)

        # 窗函数缓存
        self._cached_window = None
        self._cached_window_key = (None, None)

    # --- Methods copied/adapted from original/models/psdnorm.py ---

    def _get_window(self, device, dtype):
        key = (device, dtype)
        if self._cached_window is None or self._cached_window_key != key:
            self._cached_window = _hann_window(self.F_len, device, dtype)
            self._cached_window_key = key
        return self._cached_window

    def _encode_spectral_structure(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, C, L) —— 已经去掉了时域均值的特征
        return: spectral_repr: (N, C, F_r)

        优化说明：使用 unfold 替代 for 循环进行滑窗，不降低精度。
        """
        N, C, L = x.shape
        device, dtype = x.device, x.dtype
        F_len = self.F_len
        hop = self.hop if self.hop is not None else max(1, F_len // 2)

        # 1. 序列过短时的 Padding 处理 (逻辑同原版)
        if L < F_len:
            pad = F_len - L
            x_pad = F.pad(x, (0, pad))  # (N, C, F_len)
        else:
            x_pad = x

        # 2. 使用 unfold 生成滑窗，替代手动切片循环
        # x_pad: (N, C, L_pad) -> (N, C, num_windows, F_len)
        # unfold 的步长逻辑与 _segment_indices 一致
        windows = x_pad.unfold(dimension=-1, size=F_len, step=hop)

        # 3. 应用窗函数
        w = self._get_window(device, dtype)  # (F_len,)
        # 利用广播机制: (N, C, num_windows, F_len) * (F_len,)
        windows_weighted = windows * w

        # 4. 批量 FFT 计算
        # torch.fft.rfft 默认在最后一维进行
        # 结果: (N, C, num_windows, F_r)
        Xf = torch.fft.rfft(windows_weighted, n=F_len, dim=-1)

        # 5. 计算能量谱
        P = (Xf.abs() ** 2)

        # 6. 对所有窗口取平均
        # dim=-2 对应 num_windows 维度
        spectral_repr = P.mean(dim=-2)  # (N, C, F_r)

        return spectral_repr

    def _build_consensus_spectrum(self, P_batch: torch.Tensor) -> torch.Tensor:
        """
        P_batch: (N_domains, D, F_r)
        return: consensus: (D, F_r) (几何均值)
        """
        # (N_domains, D, F_r) -> (D, F_r)
        return (P_batch.clamp_min(self.eps).sqrt().mean(dim=0)) ** 2

    # --- End of copied methods ---

    def forward(self, features: torch.Tensor, domains: torch.Tensor) -> torch.Tensor:
        """
        计算总损失。
        features: (B, T, D) - 例如 (128, 20, 512)
        domains: (B,) - 例如 (128,)
        """
        # --- 基本形状与类型检查 ---
        if features.dim() != 3:
            raise ValueError(f"[PSDGeometricLoss] features 期望是 3D (B, T, D)，但得到 {features.shape}")
        if domains.dim() != 1:
            raise ValueError(f"[PSDGeometricLoss] domains 期望是一维 (B,)，但得到 {domains.shape}")

        B, T, D = features.shape
        if domains.shape[0] != B:
            raise ValueError(
                f"[PSDGeometricLoss] features batch={B} 与 domains batch={domains.shape[0]} 不一致"
            )

        # 确保在 float 计算
        x_feat = features.float()
        dom = domains.long()

        domain_ids = torch.unique(dom)

        # 只有一个域时，无法做“跨域对齐”，返回 0
        if len(domain_ids) < 2:
            return x_feat.new_tensor(0.0)

        if not hasattr(self, "_debug_cnt"):
            self._debug_cnt = 0
        if self._debug_cnt < 10:
            print("[PSD DEBUG] unique domains in batch:", domain_ids.tolist())
            self._debug_cnt += 1

        # --- 优化：预先计算整个 Batch 的 PSD，避免在循环中重复调用开销 ---

        # 1. 准备输入: (B, T, D) -> (B, D, T)
        x_all = x_feat.permute(0, 2, 1)  # (N=B, C=D, L=T)

        # 2. 移除 DC (去中心化)
        mu_all = x_all.mean(dim=-1, keepdim=True)
        x_all_centered = x_all - mu_all

        # 3. 批量计算所有样本的 PSD (B, D, F_r)
        # 这里的计算量最大，向量化后在 GPU 上极快
        all_samples_psd = self._encode_spectral_structure(x_all_centered)

        # --- 按 Domain 聚合 ---

        all_domain_psds = []

        for i in domain_ids:
            mask = (dom == i)
            # 使用 mask 索引直接提取当前域的预计算 PSD
            # 这里的 mask 操作是在 tensor 上进行的，非常快
            features_psd_domain = all_samples_psd[mask]  # (B_i, D, F_r)

            if features_psd_domain.shape[0] == 0:
                continue

            # 4. 计算该域的平均 PSD (D, F_r)
            P_domain_i = features_psd_domain.mean(dim=0)
            all_domain_psds.append(P_domain_i)

        if len(all_domain_psds) < 2:
            return x_feat.new_tensor(0.0)

        # (N_domains, D, F_r)
        psd_stack = torch.stack(all_domain_psds, dim=0)

        # 5. 计算几何中心 (D, F_r)，并视作“对齐目标”（不反向更新）
        psd_center = self._build_consensus_spectrum(psd_stack).detach()

        # 6. 计算损失: (PSD_i, PSD_center) 之间的 MSE，取平均
        total_loss = x_feat.new_tensor(0.0)
        for P_domain_i in all_domain_psds:
            total_loss = total_loss + F.mse_loss(P_domain_i, psd_center)

        total_loss = total_loss / len(all_domain_psds)
        return total_loss


if __name__ == '__main__':
    # 测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, T, D = 128, 20, 512
    N_DOMAINS = 4

    features = torch.randn(B, T, D, device=device)
    domains = torch.randint(0, N_DOMAINS, (B,), device=device)

    # F_len=9, hop=4 (默认)
    loss_fn = PSDGeometricLoss(F_len=9).to(device)

    # 预热一下 (可选)
    loss_fn(features, domains)

    import time

    torch.cuda.synchronize()
    start = time.time()

    loss = loss_fn(features, domains)

    torch.cuda.synchronize()
    end = time.time()

    print(f"B={B}, T={T}, D={D}, N_Domains={N_DOMAINS}")
    print(f"PSD Geometric Loss (F_len=9): {loss.item()}")
    print(f"Time: {(end - start) * 1000:.4f} ms")

    loss.backward()
    print("Backward pass successful.")
