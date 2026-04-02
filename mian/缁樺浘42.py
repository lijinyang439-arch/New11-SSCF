import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import wasserstein_distance
from scipy.interpolate import interp1d
from tqdm import tqdm
import warnings
import sys
from types import SimpleNamespace
import random
import pandas as pd

# 1. 导入模型
sys.path.append(os.getcwd())
try:
    from original.models.model import Model
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), 'wujiHDGDDP'))
    try:
        from original.models.model import Model
    except ImportError:
        print("[Error] Could not import Model.")

warnings.filterwarnings("ignore")

# ==========================================
# [配置区]
# ==========================================
CHECKPOINT_PATH = "/data/lijinyang/1_SleepHDG主要运行代码/wujiHDGDDP/results/SleepHDG_all数据集_5轮test一次_用8类算好的几何中心——test指标_2025-12-29_13-00-44/fold0/best_test_acc_a0.8320_f0.7750_ep44.pth"
DATA_ROOT = "/fdata/lijinyang/datasets_dir2_all_Merged"
SOURCE_DOMAIN = 'sleep-edfx'

TARGET_DOMAINS = [
    'ISRUC',
    'ABC',
    'MESA',
    'MROS1',
    'CFS',
    'CCSHS',
]

N_CLUSTERS = 8
FIXED_SEQ_LEN = 20
FIXED_CHANNELS = 2
FIXED_TIME_LEN = 3000
LATENT_DIM = 128


# ==========================================
# 数据处理
# ==========================================
def robust_standardize_input(x):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x.copy()).float()
    if x.ndim == 1:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
        x = x.shape[0] == FIXED_SEQ_LEN and x.unsqueeze(1) or x.shape[0] == FIXED_CHANNELS and x.unsqueeze(0) or \
            x.shape[1] == FIXED_CHANNELS and x.T.unsqueeze(0) or x.T.unsqueeze(1)
    if x.shape[0] != FIXED_SEQ_LEN:
        if x.shape[0] == 1:
            x = x.repeat(FIXED_SEQ_LEN, 1, 1)
        else:
            x = x.repeat((FIXED_SEQ_LEN // x.shape[0]) + 1, 1, 1)[:FIXED_SEQ_LEN]
    if x.shape[1] != FIXED_CHANNELS:
        if x.shape[1] == 1:
            x = x.repeat(1, FIXED_CHANNELS, 1)
        else:
            x = x[:, :FIXED_CHANNELS, :]
    if x.shape[2] != FIXED_TIME_LEN:
        x = x.reshape(-1, x.shape[1], x.shape[2])
        x = F.interpolate(x, size=FIXED_TIME_LEN, mode='linear', align_corners=False)
        x = x.view(FIXED_SEQ_LEN, FIXED_CHANNELS, FIXED_TIME_LEN)
    if x.shape == (FIXED_SEQ_LEN, FIXED_CHANNELS, FIXED_TIME_LEN): return x
    return None


def compute_spectral_feature(x, target_dim=128):
    if isinstance(x, np.ndarray): x = torch.from_numpy(x).float()
    x_flat = x.view(-1, x.shape[-1])
    x_flat = x_flat - x_flat.mean(dim=-1, keepdim=True)
    w = torch.hann_window(x_flat.shape[-1], device=x.device)
    fft = torch.fft.rfft(x_flat * w, dim=-1).abs() ** 2
    psd = fft.mean(dim=0)
    limit = max(32, psd.shape[0] // 4)
    psd = psd[:limit]
    psd = psd.view(1, 1, -1)
    psd_proj = F.interpolate(psd, size=target_dim, mode='linear', align_corners=False)
    return psd_proj.view(-1).numpy()


def extract_features(data_root, dataset_name, model, device, max_samples=800):
    print(f"[Info] Extracting {dataset_name}...")
    ds_path = os.path.join(data_root, dataset_name)
    files = []
    for root, _, fs in os.walk(ds_path):
        for f in fs:
            if f.endswith('.npy'): files.append(os.path.join(root, f))
    if len(files) > max_samples: files = random.sample(files, max_samples)

    raw_feats = []
    latent_feats = []
    model.eval()
    for f in tqdm(files, leave=False):
        try:
            raw = np.load(f, mmap_mode='r')
            x_std = robust_standardize_input(raw)
            if x_std is None: continue

            raw_f = compute_spectral_feature(x_std, target_dim=LATENT_DIM)
            raw_feats.append(raw_f)

            with torch.no_grad():
                batch_x = x_std.unsqueeze(0).to(device)
                _, mu = model.ae(batch_x)
                feat = mu.mean(dim=1).squeeze(0).cpu()
                if feat.shape[0] != LATENT_DIM:
                    feat = F.interpolate(
                        feat.view(1, 1, -1),
                        size=LATENT_DIM,
                        mode='linear',
                        align_corners=False
                    ).view(-1)
                latent_feats.append(feat.numpy())
        except:
            continue
    return np.array(raw_feats), np.array(latent_feats)


def fit_anchors(s_feats, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(s_feats)
    return kmeans.cluster_centers_


def compute_similarity_to_anchors(t_feats, anchors):
    sim_matrix = cosine_similarity(t_feats, anchors)
    max_sims = np.max(sim_matrix, axis=1)
    return max_sims


# ==========================================
# 核心功能：Bootstrap ECDF 平滑 & 绘图
# ==========================================
def bootstrap_smooth_ecdf(data, n_boot=100, grid_points=1000, x_min=None, x_max=None):
    """
    使用 Bootstrap 重采样计算平滑的 ECDF 曲线。
    原理：多次重采样 -> 计算阶梯ECDF -> 映射到统一网格 -> 取平均
    """
    data = np.asarray(data)
    if x_min is None: x_min = data.min()
    if x_max is None: x_max = data.max()  # 通常是 1.0

    # 建立统一的 X 轴网格
    x_grid = np.linspace(x_min, x_max, grid_points)
    y_accum = np.zeros_like(x_grid)

    # Bootstrap 循环
    for _ in range(n_boot):
        # 有放回重采样
        sample = np.random.choice(data, size=len(data), replace=True)
        sample.sort()

        # 构建当前样本的 ECDF 映射函数
        # y 坐标从 1/N 到 1
        y_vals = np.arange(1, len(sample) + 1) / len(sample)

        # 使用线性插值将当前样本的 ECDF 映射到统一网格 x_grid
        # bounds_error=False, fill_value=(0, 1) 保证网格超出样本范围时正确填充
        f = interp1d(sample, y_vals, kind='linear', bounds_error=False, fill_value=(0, 1))
        y_accum += f(x_grid)

    # 取平均得到平滑曲线
    y_smooth = y_accum / n_boot
    return x_grid, y_smooth


def plot_ecdf_with_colors(results, sim_src_ref, out_path="Figure4_Smooth_Bootstrap.pdf"):
    print("[Plotting] Bootstrap Smooth ECDF with Color-Coded Table...")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Liberation Serif']
    plt.rcParams['axes.linewidth'] = 0.8

    # 调色板
    palette = sns.color_palette("tab10", n_colors=max(10, len(results)))
    color_map = {r['name']: palette[i] for i, r in enumerate(results)}

    # 创建 Figure
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)

    # 确定 X 轴范围（为了平滑函数的网格）
    # 收集所有数据计算 percentile 避免极值影响
    all_vals = np.concatenate([sim_src_ref] + [r['before'] for r in results] + [r['after'] for r in results])
    xmin_global = np.percentile(all_vals, 1.0)  # 略去左边极端的离群点
    xmax_global = 1.0  # Cosine 相似度最大为 1

    # -------------------------------------------------------
    # 1. 绘制 Source Reference (灰色背景，平滑)
    # -------------------------------------------------------
    sx, sy = bootstrap_smooth_ecdf(sim_src_ref, x_min=xmin_global, x_max=xmax_global)
    ax.plot(sx, sy, color='gray', linestyle='-', linewidth=3, label='Source Reference', zorder=1, alpha=0.4)
    ax.fill_between(sx, sy, color='gray', alpha=0.1, zorder=0)

    # -------------------------------------------------------
    # 2. 绘制 Targets (Before/After) 并收集表格数据
    # -------------------------------------------------------
    table_data = []
    # 列结构: [ColorBlock, Dataset, WD(Pre), WD(Post), Gain]

    for r in results:
        name = r['name']
        sim_b = r['before']
        sim_a = r['after']
        c = color_map[name]

        # 计算指标
        wd_before = wasserstein_distance(sim_b, sim_src_ref)
        wd_after = wasserstein_distance(sim_a, sim_src_ref)
        improv = (wd_before - wd_after) / wd_before * 100.0 if wd_before > 0 else 0.0

        # 准备表格行：注意第一列是一个占位符，用来放颜色
        # 字符 '■' (U+25A0) 是实心方块
        table_data.append(["■", name, f"{wd_before:.4f}", f"{wd_after:.4f}", f"{improv:+.1f}%"])

        # 绘制 Before (虚线, 半透明)
        bx, by = bootstrap_smooth_ecdf(sim_b, x_min=xmin_global, x_max=xmax_global)
        ax.plot(bx, by, color=c, linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)

        # 绘制 After (实线, 鲜艳)
        ax_vals, ay_vals = bootstrap_smooth_ecdf(sim_a, x_min=xmin_global, x_max=xmax_global)
        ax.plot(ax_vals, ay_vals, color=c, linestyle='-', linewidth=2.5, alpha=0.9, zorder=3)

    # -------------------------------------------------------
    # 3. 设置轴和图例位置
    # -------------------------------------------------------
    # 调整 X 轴显示范围，聚焦在分布密集区
    display_xmin = np.percentile(all_vals, 5.0)
    ax.set_xlim(left=max(0.0, display_xmin - 0.01), right=1.005)
    ax.set_ylim(-0.02, 1.02)

    ax.set_xlabel("Cosine Similarity to Structural Center", fontsize=12)
    ax.set_ylabel("Cumulative Probability (Bootstrap ECDF)", fontsize=12)
    ax.grid(True, which='major', linestyle=':', linewidth=0.5, color='gray', alpha=0.3)

    # === 图例优化：放在左下角空白处 ===
    custom_lines = [
        plt.Line2D([0], [0], color='gray', lw=3, linestyle='-', alpha=0.4),
        plt.Line2D([0], [0], color='black', lw=1.2, linestyle='--', alpha=0.6),
        plt.Line2D([0], [0], color='black', lw=2.5, linestyle='-', alpha=0.9),
    ]
    # loc='lower left' 加上 bbox_to_anchor 微调位置 (x, y)
    # 放在 (0.02, 0.02) 处，避免遮挡曲线
    ax.legend(custom_lines, ['Source (Ref)', 'Target (Raw)', 'Target (Aligned)'],
              loc='lower left', bbox_to_anchor=(0.02, 0.02),
              frameon=True, fontsize=10, fancybox=False, edgecolor='k', framealpha=0.9)

    # -------------------------------------------------------
    # 4. 嵌入量化表格 (带颜色块)
    # -------------------------------------------------------
    # 计算均值行
    avg_gain = np.mean([float(x[4].strip('%')) for x in table_data])
    table_data.append(["", "MEAN", "-", "-", f"{avg_gain:+.1f}%"])  # MEAN 行没有颜色块

    col_labels = ["", "Dataset", "WD(Pre)", "WD(Post)↓", "Gain%"]

    # 表格位置：左上角 (Upper Left)
    the_table = ax.table(cellText=table_data,
                         colLabels=col_labels,
                         loc='upper left',
                         bbox=[0.02, 0.40, 0.48, 0.55])  # [x, y, width, height]

    the_table.auto_set_font_size(False)
    the_table.set_fontsize(8)
    # 调整列宽比例：颜色列要很窄
    # 遗憾的是 matplotlib table 自动布局很难精确控制列宽，只能通过 bbox 整体控制

    # === 核心：遍历单元格设置颜色 ===
    cells = the_table.get_celld()
    n_rows = len(table_data)
    n_cols = len(col_labels)

    for (row, col), cell in cells.items():
        cell.set_edgecolor('white')
        cell.set_facecolor('none')
        cell.set_text_props(color='black', ha='center')  # 默认居中

        # 表头处理
        if row == 0:
            cell.set_text_props(weight='bold', color='#333333')
            # 去掉第一列表头的下划线/边框，因为它只是占位
            if col == 0: cell.set_visible(False)

        # MEAN 行处理 (最后一行)
        elif row == n_rows:
            cell.set_text_props(weight='bold')
            if col == 4:  # Gain 列
                cell.set_text_props(color='darkgreen', weight='bold')

        # 数据行处理
        else:
            dataset_name = table_data[row - 1][1]
            # 获取该数据集的颜色
            curr_color = color_map.get(dataset_name, 'black')

            # 第一列：颜色块列
            if col == 0:
                cell.set_text_props(color=curr_color, fontsize=12, weight='bold')  # 让方块大一点
                # 也可以直接设置背景色，但文字比较好控制对齐
                # cell.set_facecolor(curr_color)

            # 第二列：数据集名称列 -> 设置颜色与曲线一致
            if col == 1:
                cell.set_text_props(color=curr_color, weight='bold', ha='left')

            # 最后一列：Gain
            if col == 4:
                cell.set_text_props(color='darkgreen', weight='bold')

    sns.despine(ax=ax, trim=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    print(f"[Success] Saved {out_path} with smooth lines and color legend.")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    params = SimpleNamespace(num_of_classes=5, d_model=128, nhead=4, num_layers=1, dim_feedforward=512, dropout=0.1,
                             activation="relu")
    try:
        model = Model(params).to(device)
        state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
    except Exception as e:
        print(f"[Error] {e}")
        return

    s_raw, s_lat = extract_features(DATA_ROOT, SOURCE_DOMAIN, model, device)
    if s_raw is None or s_lat is None or len(s_raw) == 0 or len(s_lat) == 0:
        print("[Error] Source features empty.")
        return

    print("[Metric] Fitting Source Anchors...")
    anchors_raw = fit_anchors(s_raw, N_CLUSTERS)
    anchors_lat = fit_anchors(s_lat, N_CLUSTERS)

    print("[Metric] Computing Source reference...")
    sim_src = compute_similarity_to_anchors(s_lat, anchors_lat)

    results = []
    for td in TARGET_DOMAINS:
        t_raw, t_lat = extract_features(DATA_ROOT, td, model, device)
        if t_raw is None or t_lat is None or len(t_raw) == 0 or len(t_lat) == 0:
            print(f"[Warn] Skip {td}: empty features.")
            continue

        print(f"[Metric] Computing {td}...")
        sim_before = compute_similarity_to_anchors(t_raw, anchors_raw)
        sim_after = compute_similarity_to_anchors(t_lat, anchors_lat)

        results.append({
            'name': td,
            'before': sim_before,
            'after': sim_after,
        })

    if len(results) == 0:
        print("[Error] No valid target results.")
        return

    plot_ecdf_with_colors(results, sim_src, out_path="Figure4_Smooth_Bootstrap.pdf")


if __name__ == "__main__":
    main()