# -*- coding: utf-8 -*-
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _hann_window(F_len: int, device, dtype):
    w = torch.hann_window(F_len, periodic=True, dtype=dtype, device=device)
    w = w / torch.linalg.vector_norm(w)
    return w


def _segment_indices(L: int, F_len: int, hop: Optional[int] = None) -> Tuple[torch.Tensor, int]:
    if hop is None:
        hop = max(1, F_len // 2)
    if L < F_len:
        return torch.tensor([0]), 1
    num = 1 + (L - F_len) // hop
    starts = torch.arange(0, num * hop, hop)
    return starts, hop

"""
1. 加载外部几何中心。
2. 接收 domain_ids（subject_id），将同一批次下同一个人的所有 epoch 求平均。
3. 把这个平均人的 PSD 和中心算距离，找到最近的中心。
4. 将该人的所有 epoch 都映射向他对应的这一个几何中心。
"""
class SpectralStructureAlignment1d(nn.Module):
    def __init__(self, num_channels: int, F_len: int = 5,
                 momentum: float = 1e-2, eps: float = 1e-8,
                 hop: Optional[int] = None,
                 external_centers: Optional[torch.Tensor] = None):
        super().__init__()
        assert F_len >= 1, "F_len 必须 >= 1"
        self.C = num_channels
        self.F_len = int(F_len)
        self.eps = float(eps)
        self.hop = hop

        self._cached_window = None
        self._cached_window_key = (None, None)

        if external_centers is not None:
            # 注册外部中心 (K, D) 或 (K, C, Fr)
            self.register_buffer("cluster_centers", external_centers.clone().detach())
            self.use_external = True
            print(f"[SSAM] 已初始化外部几何中心，数量: {len(self.cluster_centers)}")
        else:
            # 回退模式
            self.register_buffer("running_spectral_consensus", torch.ones(self.C, (self.F_len // 2) + 1))
            self.use_external = False
            self.momentum = float(momentum)

    def _get_window(self, device, dtype):
        key = (device, dtype)
        if self._cached_window is None or self._cached_window_key != key:
            self._cached_window = _hann_window(self.F_len, device, dtype)
            self._cached_window_key = key
        return self._cached_window

    def _encode_spectral_structure(self, x: torch.Tensor) -> torch.Tensor:
        N, C, L = x.shape
        device, dtype = x.device, x.dtype
        F_len = self.F_len
        w = self._get_window(device, dtype)
        starts, hop = _segment_indices(L, F_len, self.hop)

        if starts.numel() == 1 and L < F_len:
            pad = F_len - L
            x_pad = F.pad(x, (0, pad))
        else:
            x_pad = x

        seg_list = []
        for s in starts.tolist():
            seg = x_pad[..., s:s + F_len]
            if seg.shape[-1] < F_len:
                seg = F.pad(seg, (0, F_len - seg.shape[-1]))
            seg = seg * w
            Xf = torch.fft.rfft(seg, n=F_len, dim=-1)
            P = (Xf.abs() ** 2)
            seg_list.append(P)

        P_stack = torch.stack(seg_list, dim=0)
        spectral_repr = P_stack.mean(dim=0)
        return spectral_repr

    def _build_consensus_spectrum(self, P_batch: torch.Tensor) -> torch.Tensor:
        return (P_batch.clamp_min(self.eps).sqrt().mean(dim=0)) ** 2

    @torch.no_grad()
    def _update_running_consensus(self, P_consensus: torch.Tensor):
        alpha = self.momentum
        run_s = self.running_spectral_consensus.clamp_min(self.eps).sqrt()
        cur_s = P_consensus.clamp_min(self.eps).sqrt()
        new_run = ((1.0 - alpha) * run_s + alpha * cur_s) ** 2
        self.running_spectral_consensus.copy_(new_run)

    def _transport_to_consensus(self, x_centered: torch.Tensor,
                                P_hat: torch.Tensor,
                                P_ref: torch.Tensor) -> torch.Tensor:
        N, C, L = x_centered.shape
        L_r = (L // 2) + 1

        # 维度自动适配 (广播)
        if P_ref.ndim == P_hat.ndim - 1:
            P_ref = P_ref.unsqueeze(0)

        H_welch = torch.sqrt(
            (P_ref.expand_as(P_hat).clamp_min(self.eps)) /
            P_hat.clamp_min(self.eps)
        )

        H = F.interpolate(
            H_welch,
            size=L_r,
            mode='linear',
            align_corners=True
        )

        Xf_full = torch.fft.rfft(x_centered, n=L, dim=-1)
        Yf = Xf_full * H
        y = torch.fft.irfft(Yf, n=L, dim=-1)
        return y

    def forward(self, x: torch.Tensor, domain_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[1] == self.C, f"Input must be (N, {self.C}, L)"
        N, C, L = x.shape

        mu = x.mean(dim=-1, keepdim=True)
        x_centered = x - mu

        # 1. 计算当前样本的 PSD
        P_hat = self._encode_spectral_structure(x_centered)  # (N, C, Fr)

        P_ref = None

        if self.use_external:
            # === [核心逻辑] 寻找最近几何中心 ===
            centers = self.cluster_centers

            # 1. 维度还原: (K, 192) -> (K, 64, 3)
            if centers.dim() == 2:
                K, D = centers.shape
                Fr_expected = P_hat.shape[-1]
                if D == self.C * Fr_expected:
                    centers = centers.view(K, self.C, Fr_expected)
                else:
                    pass

            # === 2. [新增] 按照人(domain_ids)分组计算对齐 ===
            if domain_ids is not None:
                bz = domain_ids.shape[0]
                seq_len = N // bz
                
                # domain_ids 是 (bz,)，扩展并拉平到 (N,) 和 P_hat 对齐
                expanded_domain_ids = domain_ids.unsqueeze(1).expand(bz, seq_len).reshape(N)
                
                # 找到当前批次涉及到的所有“人”的 ID
                unique_domains = torch.unique(expanded_domain_ids)
                
                P_ref = torch.empty_like(P_hat)
                
                # 对当前批次的每一个人(subject)进行循环
                for uid in unique_domains:
                    mask = (expanded_domain_ids == uid)
                    
                    # 取出这个人在当前批次下所有的 epoch，并在 epoch 维度上求平均
                    P_hat_uid_mean = P_hat[mask].mean(dim=0, keepdim=True) # (1, C, Fr)
                    
                    # 拿着这个人的平均 PSD，跟固定好的所有中心算距离
                    flat_P_mean = P_hat_uid_mean.view(1, -1)  # (1, D)
                    flat_C = centers.view(centers.shape[0], -1)  # (K, D)
                    
                    # 欧氏距离矩阵 (1, K)
                    dists = torch.cdist(flat_P_mean, flat_C, p=2)
                    
                    # 找到这个“人”匹配的最接近的 1 个中心
                    closest_idx = torch.argmin(dists, dim=1)[0]
                    
                    # 把这 1 个中心，赋给属于这个人的所有 epoch
                    P_ref[mask] = centers[closest_idx]
            else:
                # [兜底回退]：如果由于某处没传入 domain_ids(例如 Inference 没传)，回退到单独的 epoch 独立对齐
                flat_P = P_hat.view(N, -1)
                flat_C = centers.view(centers.shape[0], -1)
                dists = torch.cdist(flat_P, flat_C, p=2)
                closest_indices = torch.argmin(dists, dim=1)
                P_ref = centers[closest_indices]

        elif self.training:
            # 无外部中心 -> 沿用 Batch 均值
            P_consensus = self._build_consensus_spectrum(P_hat)
            if self.running_spectral_consensus.shape[-1] != P_consensus.shape[-1]:
                self.running_spectral_consensus = torch.ones_like(P_consensus)
            self._update_running_consensus(P_consensus)
            P_ref = P_consensus

        else:
            # 测试时
            P_ref = self.running_spectral_consensus

        # 5. 执行对齐
        y = self._transport_to_consensus(x_centered, P_hat, P_ref)

        return y