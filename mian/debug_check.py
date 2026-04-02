import torch
import yaml
import os
import argparse
import inspect
import importlib
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description="Debug Model Structure & Checkpoint")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="权重文件路径")
    args = parser.parse_args()

    # 1. 加载配置
    print("\n" + "=" * 20 + " 1. 环境与配置检查 " + "=" * 20)
    if not os.path.exists(args.config):
        print(f"[Error] Config not found: {args.config}")
        return
    with open(args.config, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)
    params = SimpleNamespace(**config_dict)

    # 防止加载 centers 导致额外报错
    params.external_centers = None
    if hasattr(params, 'align'): params.align = {}

    # 2. 动态导入模型
    print("\n" + "=" * 20 + " 2. 代码源文件检查 " + "=" * 20)
    try:
        # 强制重新加载，防止 notebook 缓存
        module_path = 'original.models.model'
        if module_path in importlib.sys.modules:
            print(f"[Info] Reloading {module_path}...")
            model_lib = importlib.reload(importlib.sys.modules[module_path])
        else:
            model_lib = importlib.import_module(module_path)

        ModelClass = model_lib.Model
    except ImportError as e:
        print(f"[Fatal] 无法导入模型: {e}")
        print("请确保你在项目根目录下运行 (wujihdg2 目录)")
        return

    # 初始化模型
    model = ModelClass(params)
    epoch_encoder = model.ae.encoder.epoch_encoder

    # --- 核心：打印代码来源 ---
    print(f"[Check] Model 定义所在的物理路径:")
    print(f"  -> {inspect.getfile(ModelClass)}")
    print(f"[Check] EpochEncoder 定义所在的物理路径:")
    print(f"  -> {inspect.getfile(epoch_encoder.__class__)}")

    # 3. 打印内存中的模型结构
    print("\n" + "=" * 20 + " 3. 内存中模型结构 (print) " + "=" * 20)
    print(epoch_encoder)

    print("\n[Check] 属性存在性检测:")
    has_conv1 = hasattr(epoch_encoder, 'conv1')
    has_encoder_seq = hasattr(epoch_encoder, 'encoder')
    print(f"  -> hasattr(epoch_encoder, 'conv1')   = {has_conv1}")
    print(f"  -> hasattr(epoch_encoder, 'encoder') = {has_encoder_seq}")

    if has_conv1:
        print("\n[结论] 当前代码是【新版/Explicit】结构。")
    elif has_encoder_seq:
        print("\n[结论] 当前代码是【旧版/Sequential】结构。")
    else:
        print("\n[结论] 结构未知，既没有 conv1 也没有 encoder 序列。")

    # 4. 检查 Checkpoint 文件内容
    print("\n" + "=" * 20 + " 4. 权重文件内容检查 " + "=" * 20)
    print(f"Loading: {args.checkpoint}")
    try:
        # weights_only=False 解决 numpy 问题
        ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)

        state_dict = ckpt['model_state'] if 'model_state' in ckpt else ckpt

        # 筛选 epoch_encoder 相关的键
        keys = [k for k in state_dict.keys() if 'epoch_encoder' in k]

        print(f"\n[Info] 权重文件中包含 {len(keys)} 个与 epoch_encoder 相关的键。")
        print("前 5 个键名示例:")
        for k in keys[:5]:
            print(f"  - {k}")

        # 智能判定
        has_ckpt_conv1 = any('conv1' in k for k in keys)
        has_ckpt_seq = any('.encoder.0.' in k for k in keys)

        print("\n[Check] 权重文件结构判定:")
        print(f"  -> 包含 'conv1' 字样? {has_ckpt_conv1}")
        print(f"  -> 包含 '.encoder.0.' (Sequential) 字样? {has_ckpt_seq}")

        if has_ckpt_conv1:
            print("\n[结论] 权重文件匹配【新版/Explicit】结构。")
        elif has_ckpt_seq:
            print("\n[结论] 权重文件匹配【旧版/Sequential】结构。")

    except Exception as e:
        print(f"[Error] 读取权重文件失败: {e}")

    print("\n" + "=" * 20 + " 结束 " + "=" * 20)


if __name__ == "__main__":
    main()