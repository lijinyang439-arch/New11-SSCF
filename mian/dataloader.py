#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
比较多个 PSG 数据集的 EEG 功率谱 + 频谱几何中心差异。

流程：
1）使用你给的 LoadDataset，以 test 模式分别加载多个数据集；
2）从每个数据集中抽取若干 EEG epoch；
3）对每个 epoch 计算 PSD，做平均得到“该数据集的平均 PSD”；
4）计算每个数据集的频谱几何中心（Spectral Centroid）；
5）画出：
   - 不同数据集的平均 PSD 对比图；
   - 不同数据集的几何中心柱状图。
"""

import os
from types import SimpleNamespace

import numpy as np
import torch
import matplotlib.pyplot as plt

from datasets.dataset import LoadDataset
import matplotlib
import matplotlib.pyplot as plt




# ====================== 工具函数：PSD + 几何中心 ======================

def compute_psd_and_freq(x: np.ndarray, fs: float):
    """
    使用 FFT 计算功率谱 (简单版，不依赖 scipy)

    参数
    ----
    x : np.ndarray
        一段 1D 信号，例如长度 3000
    fs : float
        采样率，例如 100 Hz

    返回
    ----
    freqs : np.ndarray
        频率轴 0 ~ fs/2
    Pxx : np.ndarray
        对应的功率谱
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    # 汉宁窗，减小频谱泄露
    win = np.hanning(n)
    xw = x * win

    # 只算非负频率部分
    X = np.fft.rfft(xw)
    Pxx = (1.0 / (fs * np.sum(win ** 2))) * np.abs(X) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    return freqs, Pxx


def compute_spectral_centroid(freqs: np.ndarray, Pxx: np.ndarray) -> float:
    """
    计算功率谱几何中心（频谱重心）

    C = sum(f * S(f)) / sum(S(f))
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    Pxx = np.asarray(Pxx, dtype=np.float64)

    power_sum = np.sum(Pxx)
    if power_sum <= 0:
        return 0.0

    centroid = np.sum(freqs * Pxx) / power_sum
    return float(centroid)


# ====================== 从单个数据集提取平均 PSD ======================

def build_test_dataloader_for_dataset(datasets_dir: str,
                                      dataset_name: str,
                                      batch_size: int = 8,
                                      num_workers: int = 0):
    """
    对于单个数据集名 dataset_name，构建只包含 test 的 DataLoader
    """
    params = SimpleNamespace(
        datasets_dir=datasets_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        test_dataset=dataset_name,     # 触发 LoadDataset 的 test 模式
        max_samples_per_dataset=None,  # 不在这里限制数量
    )

    loader_dict, _ = LoadDataset(params).get_data_loader()
    test_loader = loader_dict["test"]

    if test_loader is None:
        raise RuntimeError(f"[错误] 数据集 {dataset_name} 未成功创建 test DataLoader。")

    return test_loader


def compute_avg_psd_for_dataset(datasets_dir: str,
                                dataset_name: str,
                                fs: float = 100.0,
                                max_epochs: int = 1000):
    """
    为某个数据集计算“平均 PSD”以及对应的频率轴和几何中心。

    参数
    ----
    datasets_dir : str
        总数据目录
    dataset_name : str
        单个数据集文件夹名，例如 'sleep-edfx'
    fs : float
        采样率
    max_epochs : int
        最多抽取多少个 epoch 来估计平均 PSD（避免太慢）

    返回
    ----
    freqs : np.ndarray
    avg_Pxx : np.ndarray
    centroid : float
    """

    print(f"\n[信息] 开始处理数据集: {dataset_name}")
    test_loader = build_test_dataloader_for_dataset(
        datasets_dir=datasets_dir,
        dataset_name=dataset_name,
        batch_size=8,
        num_workers=0,
    )

    all_psd = []
    freqs_ref = None
    used_epochs = 0

    for batch in test_loader:
        x, y, sid = batch  # x: (B, 20, 2, 3000)
        if x is None:
            continue

        # 转 numpy
        x_np = x.numpy()  # (B, 20, 2, 3000)

        B, T, C, L = x_np.shape

        # EEG 在第 0 通道：x_np[:, :, 0, :]
        # reshape 成 (B*T, L) 方便遍历
        eeg_segments = x_np[:, :, 0, :].reshape(-1, L)

        for seg in eeg_segments:
            freqs, Pxx = compute_psd_and_freq(seg, fs)

            if freqs_ref is None:
                freqs_ref = freqs
            else:
                # 简单检查频率轴是否一致
                if not np.allclose(freqs_ref, freqs):
                    raise ValueError("[错误] 频率轴不一致，可能采样率或长度不同。")

            all_psd.append(Pxx)
            used_epochs += 1

            if max_epochs is not None and used_epochs >= max_epochs:
                break

        if max_epochs is not None and used_epochs >= max_epochs:
            break

    if len(all_psd) == 0:
        raise RuntimeError(f"[错误] 数据集 {dataset_name} 中没有有效的 EEG epoch。")

    all_psd = np.stack(all_psd, axis=0)  # (N, F)
    avg_Pxx = np.mean(all_psd, axis=0)   # (F,)

    centroid = compute_spectral_centroid(freqs_ref, avg_Pxx)
    print(f"[信息] 数据集 {dataset_name} 使用 {used_epochs} 个 epoch 估计平均 PSD，几何中心 = {centroid:.3f} Hz")

    return freqs_ref, avg_Pxx, centroid


# ====================== 绘图：多数据集对比 ======================

def plot_multi_dataset_psd_and_centroid(results_dict,
                                        save_dir: str = "/data/lijinyang/1_sleep/photo"):
    """
    绘制多数据集的平均 PSD 对比图 + 频谱几何中心柱状图

    参数
    ----
    results_dict : dict
        {dataset_name: (freqs, avg_Pxx, centroid)}
    save_dir : str
        保存图片的文件夹
    """
    os.makedirs(save_dir, exist_ok=True)

    dataset_names = list(results_dict.keys())

    # --------- 图 1：平均 PSD 曲线 ---------
    plt.figure(figsize=(10, 6))
    for name in dataset_names:
        freqs, avg_Pxx, centroid = results_dict[name]
        plt.semilogy(freqs, avg_Pxx, label=f"{name} (C={centroid:.2f} Hz)")
    plt.title("multi_dataset_avg_psd")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power")
    plt.legend()
    plt.tight_layout()
    psd_path = os.path.join(save_dir, "multi_dataset_avg_psd.png")
    plt.savefig(psd_path, dpi=300)
    print(f"[信息] 平均 PSD 图已保存到: {psd_path}")
    plt.show()

    # --------- 图 2：几何中心柱状图 ---------
    centroids = [results_dict[name][2] for name in dataset_names]

    plt.figure(figsize=(8, 5))
    x_pos = np.arange(len(dataset_names))
    plt.bar(x_pos, centroids)
    plt.xticks(x_pos, dataset_names, rotation=30)
    plt.ylabel("Spectral Centroid (Hz)")
    plt.title("multi_dataset_centroids")
    plt.tight_layout()
    centroid_path = os.path.join(save_dir, "multi_dataset_centroids.png")
    plt.savefig(centroid_path, dpi=300)
    print(f"[信息] 频谱几何中心柱状图已保存到: {centroid_path}")
    plt.show()


# ====================== 主函数 ======================

def main():
    # ===== 这里改成你自己的路径和数据集名 =====
    datasets_dir = "/data/lijinyang/SleepSLeep/datasets_dir"

    # 想比较哪些数据集，就把名字写在这里
    dataset_names = [
        "sleep-edfx",
        "HMC",
        "ISRUC",
        "SHHS1",
        "P2018",
        "ABC",
        "CCSHS",
        "MROS1",
    ]

    fs = 100.0          # 你的预处理采样率
    max_epochs = 10000   # 每个数据集最多用多少个 epoch 估计平均 PSD（可以根据速度改）

    results = {}
    for name in dataset_names:
        try:
            freqs, avg_Pxx, centroid = compute_avg_psd_for_dataset(
                datasets_dir=datasets_dir,
                dataset_name=name,
                fs=fs,
                max_epochs=max_epochs,
            )
            results[name] = (freqs, avg_Pxx, centroid)
        except Exception as e:
            print(f"[警告] 处理数据集 {name} 时出错，将跳过。错误信息: {e}")

    if not results:
        print("[错误] 没有任何数据集被成功处理，请检查路径和数据。")
        return

    # 画图
    plot_multi_dataset_psd_and_centroid(results)


if __name__ == "__main__":
    main()
