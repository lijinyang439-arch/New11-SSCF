# evaluator.py
# (确保这些库已导入)
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)
from tqdm import tqdm
from types import SimpleNamespace  # 确保导入
import torch.nn as nn  # 确保导入


class Evaluator(object):
    def __init__(self, params: SimpleNamespace, data_loader):
        """
        初始化评估器。
        :param params: 包含配置的 SimpleNamespace 对象 (例如 params.num_of_classes)
        :param data_loader: 用于评估的 DataLoader (验证集或测试集)
        """
        self.params = params
        self.data_loader = data_loader
        self.num_classes = getattr(params, 'num_of_classes', 5)

    def get_accuracy(self, model: nn.Module):
        """
        计算模型在 self.data_loader 上的评估指标。

        :param model: 要评估的 PyTorch 模型 (可能是 nn.DataParallel 包装的)
        :return: (acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa, report)
        """
        # 将模型设置为评估模式
        model.eval()

        # 获取模型所在的设备
        try:
            device = next(model.parameters()).device
        except StopIteration:
            print("[WARN] 评估器：模型没有参数，默认为 'cuda'。")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_truths = []
        all_preds = []

        # 使用 tqdm 显示评估进度
        pbar_desc = "Evaluating..."
        if self.data_loader is None or len(self.data_loader) == 0:
            print("[WARN] 评估器：DataLoader 为空，跳过评估。")
            return (0.0,) * 9 + ("",)  # 返回 9 个 0.0 和一个空字符串

        pbar = tqdm(self.data_loader, mininterval=2, desc=pbar_desc, leave=False)

        # 禁用梯度计算
        with torch.no_grad():
            for x, y, z in pbar:
                # 将数据移动到 GPU
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()
                z = z.to(device, non_blocking=True).long()  # 域 ID

                # --- 模型前向传播 ---
                # 参照 trainer.py 的调用方式
                # 在 eval 模式下，labels 可以设为 None
                logits, _, _, _, _ = model(x, labels=None, domain_ids=z)

                # logits 形状预计为 (batch_size, 20, num_classes)
                # 沿类别维度 (dim=2) 找到最大值的索引
                preds = torch.argmax(logits, dim=2)

                # 收集 CPU 上的 numpy 结果
                all_truths.append(y.cpu().numpy())
                all_preds.append(preds.cpu().numpy())

        # 检查是否收集到了数据
        if not all_truths:
            print("[WARN] 评估器：未收集到任何评估结果。")
            return (0.0,) * 9 + ("",)

        # --- 这是关键修复点 ---

        # 1. 拼接所有批次的结果
        #    - truths_2d 形状: (总样本数, 20), 例如 (2040, 20)
        #    - preds_2d 形状: (总样本数, 20), 例如 (2040, 20)
        truths_2d = np.concatenate(all_truths)
        preds_2d = np.concatenate(all_preds)

        # 2. **【修复】** 将二维数组展平为一维数组
        #    - truths 形状: (总样本数 * 20,), 例如 (40800,)
        #    - preds 形状: (总样本数 * 20,), 例如 (40800,)
        truths = truths_2d.flatten()
        preds = preds_2d.flatten()

        # --- 修复结束 ---

        # 现在 truths 和 preds 都是一维且长度相同 (40800)，可以安全计算指标

        try:
            # 定义标签（确保与类别数量一致）
            labels = list(range(self.num_classes))  # [0, 1, 2, 3, 4]

            acc = accuracy_score(truths, preds)
            f1 = f1_score(truths, preds, average='macro', labels=labels, zero_division=0)
            kappa = cohen_kappa_score(truths, preds, labels=labels)
            cm = confusion_matrix(truths, preds, labels=labels)
            report = classification_report(truths, preds, labels=labels, digits=5, zero_division=0)

            # 计算每个类别的 F1 分数
            f1_per_class = f1_score(truths, preds, average=None, labels=labels, zero_division=0)

            # 确保 f1_per_class 数组长度为 5
            if len(f1_per_class) < self.num_classes:
                # 如果某个类别在 truth 和 pred 中都未出现，sklearn 可能返回较短的数组
                temp_f1s = np.zeros(self.num_classes)
                # 假设 f1_per_class[i] 对应 labels[i]
                # （这在 labels=... 被指定时是成立的）
                temp_f1s[:len(f1_per_class)] = f1_per_class
                f1_per_class = temp_f1s

            # 假设 5 个类别顺序为 [W, N1, N2, N3, R]
            wake_f1 = f1_per_class[0]
            n1_f1 = f1_per_class[1]
            n2_f1 = f1_per_class[2]
            n3_f1 = f1_per_class[3]
            rem_f1 = f1_per_class[4]

        except Exception as e:
            print(f"[ERROR] 计算评估指标时出错: {e}")
            # 返回空结果
            return (0.0,) * 9 + (f"Error: {e}",)

        # 返回 trainer.py 期望的所有指标
        return acc, f1, cm, wake_f1, n1_f1, n2_f1, n3_f1, rem_f1, kappa, report