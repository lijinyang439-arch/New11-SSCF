# utils/allutils.py
# (已更新 MetricsLogger 以包含 test_acc, test_f1 等)

import os
import random
import re
from datetime import datetime
from pathlib import Path
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import pandas as pd
from torch.utils.data import DataLoader, Subset
from collections import Counter
import torch.nn as nn
from tqdm import tqdm
def ensure_dir(path):
    """确保目录存在"""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now():
    """获取当前时间字符串"""
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


class MetricsLogger:
    """
    用于在训练期间记录和保存指标到 CSV 文件的实用工具。
    [已更新] 增加对 test_acc, test_f1, test_kappa 等的支持。
    """

    def __init__(self, log_dir: Path, fold_id: int):
        self.log_dir = ensure_dir(log_dir)
        self.fold_id = fold_id
        self.csv_path = self.log_dir / f"metrics.csv"

        # [修改] 添加新的测试指标列
        self.headers = [
            'time', 'epoch', 'lr',
            'train_loss', 'train_acc', 'train_f1',
            'val_acc', 'val_f1', 'val_kappa',
            'test_acc', 'test_f1', 'test_kappa',  # <-- 新增
            'wake_f1', 'n1_f1', 'n2_f1', 'n3_f1', 'rem_f1',
            'test_wake_f1', 'test_n1_f1', 'test_n2_f1', 'test_n3_f1', 'test_rem_f1'  # <-- 新增
        ]

        # 仅在文件不存在时写入表头
        if not self.csv_path.exists():
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
        else:
            # (可选) 检查表头是否匹配
            pass

    def log_epoch(self,
                  # 基础
                  time_str: str, epoch: int, lr: float,
                  # 训练
                  train_loss: float = None, train_acc: float = None, train_f1: float = None,
                  # 验证
                  val_acc: float = None, val_f1: float = None, val_kappa: float = None,
                  # [修改] 新增测试
                  test_acc: float = None, test_f1: float = None, test_kappa: float = None,
                  # 验证 F1
                  wake_f1: float = None, n1_f1: float = None, n2_f1: float = None,
                  n3_f1: float = None, rem_f1: float = None,
                  # [修改] 新增测试 F1
                  test_wake_f1: float = None, test_n1_f1: float = None, test_n2_f1: float = None,
                  test_n3_f1: float = None, test_rem_f1: float = None
                  ):
        """
        记录一个 epoch 的指标。
        """
        # [修改] 更新 row 字典以匹配新表头
        row = {
            'time': time_str, 'epoch': epoch, 'lr': lr,
            'train_loss': f"{train_loss:.5f}" if train_loss is not None else "",
            'train_acc': f"{train_acc:.5f}" if train_acc is not None else "",
            'train_f1': f"{train_f1:.5f}" if train_f1 is not None else "",
            'val_acc': f"{val_acc:.5f}" if val_acc is not None else "",
            'val_f1': f"{val_f1:.5f}" if val_f1 is not None else "",
            'val_kappa': f"{val_kappa:.5f}" if val_kappa is not None else "",
            'test_acc': f"{test_acc:.5f}" if test_acc is not None else "",
            'test_f1': f"{test_f1:.5f}" if test_f1 is not None else "",
            'test_kappa': f"{test_kappa:.5f}" if test_kappa is not None else "",
            'wake_f1': f"{wake_f1:.3f}" if wake_f1 is not None else "",
            'n1_f1': f"{n1_f1:.3f}" if n1_f1 is not None else "",
            'n2_f1': f"{n2_f1:.3f}" if n2_f1 is not None else "",
            'n3_f1': f"{n3_f1:.3f}" if n3_f1 is not None else "",
            'rem_f1': f"{rem_f1:.3f}" if rem_f1 is not None else "",
            'test_wake_f1': f"{test_wake_f1:.3f}" if test_wake_f1 is not None else "",
            'test_n1_f1': f"{test_n1_f1:.3f}" if test_n1_f1 is not None else "",
            'test_n2_f1': f"{test_n2_f1:.3f}" if test_n2_f1 is not None else "",
            'test_n3_f1': f"{test_n3_f1:.3f}" if test_n3_f1 is not None else "",
            'test_rem_f1': f"{test_rem_f1:.3f}" if test_rem_f1 is not None else "",
        }

        try:
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.headers)
                writer.writerow(row)
        except Exception as e:
            print(f"[ERROR] MetricsLogger failed to write row: {e}")
            print(f"      Row data: {row}")

    def path(self) -> Path:
        return self.csv_path


# ... (allutils.py 中的其余函数 build_ratio_loader, plot_curves_for_fold 等保持不变) ...
def build_ratio_loader(loader: DataLoader, ratio: float, seed: int = 42) -> DataLoader:
    """
    从现有 DataLoader 创建一个按比例缩小的 DataLoader。
    [修复] 必须传递原始 loader 的 collate_fn，否则遇到 None 数据会报错。
    """
    if ratio >= 1.0:
        return loader

    dataset = loader.dataset
    total_size = len(dataset)
    subset_size = int(total_size * ratio)

    # 确保我们至少有 1 个样本
    subset_size = max(1, subset_size)

    # 生成随机索引
    g = np.random.Generator(np.random.PCG64(seed))
    indices = g.permutation(total_size)[:subset_size]

    # 创建 Subset
    subset = Subset(dataset, indices)

    # 创建新的 DataLoader
    new_loader = DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=True,
        num_workers=loader.num_workers,
        pin_memory=loader.pin_memory,
        collate_fn=loader.collate_fn,  # <--- 【核心修复】：必须继承原有的 collate_fn
        persistent_workers=loader.persistent_workers, # 建议同时也继承这个参数
        prefetch_factor=loader.prefetch_factor        # 建议同时也继承这个参数
    )

    print(f"      [INFO] Building ratio loader: {subset_size} / {total_size} samples ({ratio * 100:.1f}%)")
    return new_loader

def plot_curves_for_fold(metrics_csv_path: Path, out_dir: Path):
    """
    读取 metrics.csv 文件并绘制训练/验证曲线图。
    [已更新] 增加对 test_acc 和 test_f1 的绘制。
    """
    try:
        df = pd.read_csv(metrics_csv_path)
    except FileNotFoundError:
        print(f"[WARN] plot_curves: {metrics_csv_path} not found. Skipping plot.")
        return
    except pd.errors.EmptyDataError:
        print(f"[WARN] plot_curves: {metrics_csv_path} is empty. Skipping plot.")
        return

    out_dir = ensure_dir(out_dir)
    plt.style.use('ggplot')

    # --- 1. 损失曲线 ---
    plt.figure(figsize=(10, 6))
    if 'train_loss' in df.columns and df['train_loss'].notna().any():
        plt.plot(df['epoch'], df['train_loss'], label='Train Loss', alpha=0.8)
    plt.title('Epoch vs. Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(out_dir / "plot_loss.png", dpi=150)
    plt.close()

    # --- 2. 准确率曲线 ---
    plt.figure(figsize=(10, 6))
    if 'val_acc' in df.columns and df['val_acc'].notna().any():
        plt.plot(df['epoch'], df['val_acc'], label='Validation Accuracy', marker='o', markersize=3, alpha=0.8)
    # [修改] 绘制测试准确率
    if 'test_acc' in df.columns and df['test_acc'].notna().any():
        plt.plot(df['epoch'], df['test_acc'], label='Test Accuracy', marker='x', markersize=3, alpha=0.8,
                 linestyle='--')

    plt.title('Epoch vs. Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(out_dir / "plot_accuracy.png", dpi=150)
    plt.close()

    # --- 3. F1 分数曲线 ---
    plt.figure(figsize=(10, 6))
    if 'val_f1' in df.columns and df['val_f1'].notna().any():
        plt.plot(df['epoch'], df['val_f1'], label='Validation Macro F1', marker='o', markersize=3, alpha=0.8)
    # [修改] 绘制测试 F1
    if 'test_f1' in df.columns and df['test_f1'].notna().any():
        plt.plot(df['epoch'], df['test_f1'], label='Test Macro F1', marker='x', markersize=3, alpha=0.8, linestyle='--')

    plt.title('Epoch vs. Macro F1-Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1-Score')
    plt.legend()
    plt.savefig(out_dir / "plot_f1.png", dpi=150)
    plt.close()


def extract_features_for_tsne(model: nn.Module, loader: DataLoader, device, take="mu_tilde"):
    """
    [FIXED]
    使用模型从数据加载器中提取特征 (mu_tilde) 和原始数据 (x) 以进行 t-SNE。
    """
    model.eval()

    all_raw_X = []
    all_rep_X = []
    all_y = []

    with torch.no_grad():
        for x, y, z in tqdm(loader, desc="t-SNE Feature Extraction", leave=False):
            x = x.to(device)
            y = y.to(device)
            z = z.to(device)

            # --- 提取原始数据 (展平) ---
            # x shape: (B, T, C, L) e.g., (B, 20, 2, 3000)
            # 展平为 (B, T*C*L)
            raw_flat = x.cpu().numpy().reshape(x.size(0), -1)
            all_raw_X.append(raw_flat)

            # --- 提取表征 ---
            try:
                # ==================================================================
                # [FIX]
                # 原始 'if' 条件 (hasattr(model, 'inference')) 过于宽泛，
                # 'original' 模型也有 'inference'，但没有 'ldp'。
                #
                # 新逻辑：
                # 1. 检查 'ldp' AND 'mu_tilde'，这是 'improved' 模型的特征。
                # 2. 否则，回退到 'forward' pass，这两种模型都支持。
                # ==================================================================
                if hasattr(model, 'ldp') and hasattr(model, 'ae') and take == "mu_tilde":
                    # --- 逻辑 1: 'improved' 模型 (take='mu_tilde') ---
                    # (B, T, C, L) -> (B, T, D) -> (B, D)
                    mu = model.ae.encoder(x).mean(dim=1)
                    # (B, D) -> (B, D)
                    mu_tilde, _ = model.ldp(mu, domain_ids=None, use_uni=True)
                    rep = mu_tilde
                else:
                    # --- 逻辑 2: 'original' 模型 (take='mu') 或 'improved' (take='mu') ---
                    # (B, T, C, L) -> (B, T, D) 和 (B, D)
                    _, _, mu, mu_tilde, _ = model(x, labels=y, domain_ids=z)

                    if take == "mu_tilde":
                        if mu_tilde is None:
                            # 'original' model forward() returns mu_tilde as None
                            raise AttributeError(
                                f"Model {type(model).__name__} does not produce 'mu_tilde'. Check 'take' argument.")
                        rep = mu_tilde  # (B, D)
                    else:  # take == "mu"
                        if mu is None:
                            raise ValueError("Model did not return 'mu' (3rd return value)")
                        rep = mu.mean(dim=1)  # (B, T, D) -> (B, D)
                # ==================================================================

            except Exception as e:
                print(f"[WARN] t-SNE feature extraction failed: {e}. Using dummy features.")
                # This is what causes the crash:
                rep = torch.zeros(x.size(0), 10)

            all_rep_X.append(rep.cpu().numpy())

            # 标签 (取序列的第一个标签)
            all_y.append(y[:, 0].cpu().numpy())

    RAW = np.concatenate(all_raw_X, axis=0)
    REP = np.concatenate(all_rep_X, axis=0)
    Y = np.concatenate(all_y, axis=0)

    return RAW, REP, Y


def tsne_compare_plot(raw_X, rep_X, y, out_dir, title_prefix, filename_prefix):
    """
    为原始数据和表征数据生成并排的 t-SNE 图。
    """
    out_dir = ensure_dir(out_dir)

    # --- t-SNE 设置 ---
    # (如果数据量太大，进行子采样)
    n_samples = 1500
    if len(y) > n_samples:
        print(f"[INFO] t-SNE: Subsampling data from {len(y)} to {n_samples}")
        indices = np.random.permutation(len(y))[:n_samples]
        raw_X = raw_X[indices]
        rep_X = rep_X[indices]
        y = y[indices]

    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    print("[INFO] t-SNE: Fitting raw data...")
    raw_tsne = tsne.fit_transform(raw_X)
    print("[INFO] t-SNE: Fitting representation data...")
    rep_tsne = tsne.fit_transform(rep_X)

    # --- 绘图 ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    cmap = plt.cm.get_cmap('tab10', 5)  # 5 个睡眠阶段

    # 图 1: 原始数据
    scatter1 = ax1.scatter(raw_tsne[:, 0], raw_tsne[:, 1], c=y, cmap=cmap, alpha=0.7, s=10)
    ax1.set_title(f'{title_prefix} - Raw Data (t-SNE)')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')

    # 图 2: 表征
    scatter2 = ax2.scatter(rep_tsne[:, 0], rep_tsne[:, 1], c=y, cmap=cmap, alpha=0.7, s=10)
    ax2.set_title(f'{title_prefix} - Representation (t-SNE)')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')

    # 添加图例
    legend_labels = ['W', 'N1', 'N2', 'N3', 'R']
    handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_labels[i],
                          markerfacecolor=cmap(i / 4.0), markersize=10) for i in range(5)]
    fig.legend(handles=handles, title='Sleep Stage', loc='upper right')

    plt.tight_layout(rect=[0, 0, 0.9, 1])  # 为图例留出空间

    out_path = out_dir / f"{filename_prefix}_comparison.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] t-SNE plot saved to {out_path}")
    plt.close(fig)


def write_aggregate_row(csv_path: Path, row: dict):
    """
    将单行聚合结果追加到 CSV 文件中。
    [已更新] 增加对 test_kappa 的支持。
    """
    ensure_dir(csv_path.parent)

    # [修改] 更新表头以匹配新指标
    headers = [
        "time", "run_id", "fold",
        "best_val_acc", "best_val_f1",
        "test_acc", "test_f1", "test_kappa",
        "wake_f1", "n1_f1", "n2_f1", "n3_f1", "rem_f1",
        "model_path"
    ]

    # 过滤 row 字典，只保留表头中的键
    filtered_row = {k: row.get(k, "N/A") for k in headers}

    file_exists = csv_path.exists()

    try:
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if not file_exists:
                writer.writeheader()  # 仅在文件不存在时写入表头
            writer.writerow(filtered_row)
    except Exception as e:
        print(f"[ERROR] Failed to write aggregate row to {csv_path}: {e}")