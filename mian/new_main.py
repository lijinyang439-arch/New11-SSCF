import os
import torch
import argparse
from types import SimpleNamespace
from pathlib import Path
import numpy as np

# 导入你现有的模块
from original.trainer import Trainer
from datasets.dataset import LoadDataset
from torch.utils.data import Subset


class TargetFineTuner(Trainer):
    def __init__(self, params):
        # 强制设置：每个 epoch 都测试以便观察指标
        params.eval_test_every_epoch = True
        super().__init__(params)

        # 截取指定数量的目标域训练样本
        if hasattr(params, 'target_sample_size') and params.target_sample_size is not None:
            self.limit_dataset_size(params.target_sample_size)

    def limit_dataset_size(self, sample_size):
        train_ds = self.data_loader['train'].dataset
        total_available = len(train_ds)
        actual_size = min(sample_size, total_available)

        indices = np.random.choice(total_available, actual_size, replace=False)
        new_dataset = Subset(train_ds, indices)

        print(f"[微调] 目标域样本总量: {total_available}, 实际使用样本: {actual_size}")

        self.data_loader['train'] = torch.utils.data.DataLoader(
            new_dataset,
            batch_size=self.params.batch_size,
            shuffle=True,
            num_workers=self.params.num_workers
        )

    # 核心：根据测试集指标调整 Loss
    def train(self) -> dict:
        # 你可以重写循环或在父类逻辑中通过 self.ce_loss 修改参数
        # 这里演示在微调开始前设置初始值
        return super().train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--target_dataset', type=str, default='P2018')
    parser.add_argument('--sample_size', type=int, default=200)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-6)
    args = parser.parse_args()

    # 补充模型必需的参数
    config = {
        "model_version": "original",
        "datasets_dir": "/fdata/lijinyang/datasets_dir2_all_Merged",
        "target_domains": args.target_dataset,
        "target_sample_size": args.sample_size,
        "batch_size": 256,
        "epochs": args.epochs,
        "lr": args.lr,
        "dropout": 0.1,  # <--- 修复 AttributeError 的关键：添加缺失的 dropout
        "label_smoothing": 0.1,  # 用于调整 Loss 的初始参数
        "num_workers": 4,
        "fold": 0,
        "model_dir": "./finetune_results",
        "lambda_ae": 1.0,
        "lambda_psd": 0.0,
        "lambda_coral": 0.0,
        "seed": 42,
        "clip_value": 5.0,  # 建议添加梯度裁剪
        "eval_test_every_epoch": True
    }

    params = SimpleNamespace(**config)

    # 实例化
    tuner = TargetFineTuner(params)

    # 加载预训练模型
    print(f"[信息] 加载权重: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location='cuda', weights_only=False)
    state_dict = checkpoint['model_state'] if 'model_state' in checkpoint else checkpoint

    # 过滤掉 DataParallel 的前缀（如果有）
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    tuner.model.load_state_dict(new_state_dict, strict=False)

    tuner.train()


if __name__ == "__main__":
    main()