import numpy as np
from scipy import signal
import torch


def calc_psd_30s_epoch(epoch_data, fs=100):
    """
    计算单个30秒epoch的PSD。
    逻辑：
    1. 将30s信号划分为6个5s片段。
    2. 每个5s片段用Welch方法计算PSD (不log, 不归一化)。
    3. 6个PSD平均得到PSD30。

    Args:
        epoch_data (np.array): Shape (3000,) or (Channels, 3000)
        fs (int): Sampling rate
    Returns:
        f (np.array): Frequency bins
        psd_30 (np.array): Averaged PSD for this epoch
    """
    # 假设输入是 (3000,) 单通道，如果是多通道需调整轴
    # 这里按单通道处理，如果是 (C, 3000) 请在外部循环或调整axis

    if epoch_data.ndim > 1:
        epoch_data = epoch_data.squeeze()

    n_segments = 6
    seg_len = 5 * fs  # 500 points

    psd_segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = (i + 1) * seg_len
        seg = epoch_data[start:end]

        # 对5s数据进行Welch计算
        # nperseg建议设为fs或者fs*2，这里为了保证频率分辨率适中，设为fs*2(2s窗口)或fs(1s窗口)
        # 考虑到只有5s数据，用 fs*2 (200点) 窗口可能会只有2-3个重叠段
        # 或者直接对整个5s做Periodogram (nperseg=500)。
        # 根据常用习惯，对5s短数据，常用nperseg=fs*2 (2s)
        f, p = signal.welch(seg, fs=fs, nperseg=2 * fs, noverlap=fs)
        psd_segments.append(p)

    # 6个5s的PSD平均 -> PSD30
    psd_30 = np.mean(psd_segments, axis=0)

    return f, psd_30


def calc_subject_psd(all_epochs, fs=100):
    """
    计算被试级的代表性PSD。
    逻辑：
    1. 计算该人所有Epoch的 PSD30 (Linear scale)。
    2. 平均所有 PSD30 得到 PSD-subj (Linear scale)。
    3. 对 PSD-subj 进行 Log 变换。

    Args:
        all_epochs (np.array): Shape (N_epochs, 3000)
    Returns:
        f (np.array): Frequencies
        psd_subj_log (np.array): Log-transformed average PSD
    """
    psd_30_list = []
    f = None

    for epoch in all_epochs:
        f_ep, psd_30 = calc_psd_30s_epoch(epoch, fs=fs)
        psd_30_list.append(psd_30)
        if f is None:
            f = f_ep

    # 平均得到 PSD-subj (Linear)
    psd_subj_linear = np.mean(psd_30_list, axis=0)

    # Log 变换
    # 加上 eps 防止 log(0)
    psd_subj_log = np.log(psd_subj_linear + 1e-10)

    return f, psd_subj_log


def align_and_restore_epoch(epoch_data, target_psd_log, f_axis, fs=100, low_cut=0.5, high_cut=35.0):
    """
    训练模型时的对齐与还原步骤。
    逻辑：
    1. 对当前30s epoch计算 PSD30 (计算过程隐含在对齐中，其实我们需要的是FFT幅度)。
    2. 将 PSD30 (Log) 向 Target PSD (Log) 对齐。
    3. Log反变换。
    4. 傅里叶逆变换 (IFFT) 还原为30s时域信号。

    注意：为了做IFFT，必须保留原始信号的相位(Phase)。
    对齐实质上是修改幅频特性(Magnitude)。

    Args:
        epoch_data (np.array): (3000,) 原始时域信号
        target_psd_log (np.array): 目标域中心的PSD (Log scale)
        f_axis (np.array): 频率轴，用于匹配 target_psd 的索引
    Returns:
        restored_epoch (np.array): (3000,) 对齐后的时域信号
    """
    n = len(epoch_data)

    # 1. FFT 获取 幅度(Mag) 和 相位(Phase)
    fft_vals = np.fft.rfft(epoch_data)
    original_mag = np.abs(fft_vals)
    original_phase = np.angle(fft_vals)

    # 获取FFT对应的频率轴
    fft_freqs = np.fft.rfftfreq(n, d=1 / fs)

    # 2. 准备 Target Magnitude
    # target_psd_log 通常是基于 Welch 方法得到的频率轴 (f_axis)，
    # 而 rfft 的频率轴 (fft_freqs) 分辨率可能不同 (Welch通常更平滑)。
    # 我们需要将 Target PSD 插值到 FFT 的频率点上。

    target_psd_linear = np.exp(target_psd_log)

    # 插值 Target PSD 到 FFT 频率轴
    interp_target_psd = np.interp(fft_freqs, f_axis, target_psd_linear)

    # 计算当前信号的粗略 PSD (为了计算缩放因子)
    # 这里为了严格对齐，我们比较的是 "当前信号的幅度谱平方" 和 "目标PSD"
    # 或者，更简单的方法是：Target Magnitude = sqrt(Target PSD * scale_factor)
    # 但 PSD 计算涉及窗口归一化，直接用 sqrt(PSD) 作为幅度谱的形状参考是最稳健的。

    # 获取当前信号在 FFT 频率轴上的平滑包络 (模拟 Welch 的效果以便比较) 可能会很慢。
    # 简化策略：直接修改幅度谱。
    # Scale Factor = sqrt( Target_PSD_Linear / Current_PSD_Estimate )
    # 但由于我们是对单顶 epoch 操作，Current_PSD_Estimate 最好就是它自己的 PSD30 (Linear)。

    # 重新计算当前 Epoch 的 PSD30 (Linear) 以获取当前风格
    f_curr, psd_curr_30 = calc_psd_30s_epoch(epoch_data, fs)

    # 插值当前 PSD30 到 FFT 频率轴
    interp_curr_psd = np.interp(fft_freqs, f_curr, psd_curr_30)

    # 3. 计算对齐 Scaling Factor (仅在 0.5-35Hz 区间生效)
    # Mask: 0.5 <= f <= 35
    mask = (fft_freqs >= low_cut) & (fft_freqs <= high_cut)

    scale_factor = np.ones_like(fft_freqs)

    # 避免除以0
    interp_curr_psd = np.maximum(interp_curr_psd, 1e-10)

    # 核心对齐公式： Target / Current
    # sqrt 因为 PSD 是功率 (幅度平方)
    scaling = np.sqrt(interp_target_psd[mask] / interp_curr_psd[mask])

    scale_factor[mask] = scaling

    # 4. 应用 Scaling 到原始幅度
    new_mag = original_mag * scale