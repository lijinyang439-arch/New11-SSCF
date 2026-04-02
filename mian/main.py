# main.py
import argparse
import os
import random
import shutil
from datetime import datetime
from pathlib import Path
import yaml
from types import SimpleNamespace
import numpy as np
import torch
import torch.distributed as dist  # 新增
import importlib

# 引入 DDP 工具
from utils.ddp_utils import setup_ddp, cleanup_ddp, is_main_process, setup_seed as ddp_setup_seed


# --- 辅助函数 ---
# 移除了原有的 setup_seed，使用 ddp_utils 中的

def backup_code(dst_root: Path):
    """备份当前项目中关键代码"""
    # 仅主进程备份
    if not is_main_process():
        return

    code_backup_dir = dst_root / 'mian'
    code_backup_dir.mkdir(parents=True, exist_ok=True)
    print(f"[信息] 正在备份代码到 {code_backup_dir}")

    # (保持原有备份逻辑不变)
    for item in os.listdir(''):
        source_path = Path(item)
        dest_path = code_backup_dir / source_path.name
        if source_path.is_file() and source_path.suffix == '.py':
            try:
                shutil.copy(str(source_path), str(dest_path))
            except Exception as e:
                print(f"[警告] 无法备份文件 {item}: {e}")
        elif source_path.is_dir() and item == 'configs':
            try:
                shutil.copytree(str(source_path), str(dest_path), dirs_exist_ok=True)
            except Exception as e:
                print(f"[警告] 无法备份目录 {item}: {e}")

    for dirname in ['original', 'original_sleepdg', 'improved', 'models', 'losses', 'datasets', 'utils']:
        source_dir = Path(dirname)
        if source_dir.is_dir():
            dst_dir = code_backup_dir / dirname
            try:
                shutil.copytree(str(source_dir), str(dst_dir), dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            except Exception as e:
                print(f"[警告] 无法备份目录 {dirname}: {e}")


def load_config(config_path: str) -> dict:
    if not Path(config_path).is_file():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def calculate_stats(results_list: list, key: str) -> tuple[float, float]:
    values = [res[key] for res in results_list if key in res and isinstance(res[key], (int, float))]
    if not values:
        return 0.0, 0.0
    mean_val = np.mean(values)
    std_val = np.std(values)
    return mean_val, std_val


try:
    from utils.allutils import write_aggregate_row
except ImportError:
    def write_aggregate_row(path, row):
        pass


def _fmt_metric(v, ndigits=5):
    if isinstance(v, (int, float)):
        return f"{v:.{ndigits}f}"
    return str(v)


def main():
    parser = argparse.ArgumentParser(description='SleepDG 运行器（支持 DDP）')
    parser.add_argument('--config', type=str, required=True, help='YAML 配置文件路径')
    # 不需要 --local_rank 参数，torchrun 会设置环境变量
    args = parser.parse_args()

    # -------- 1. DDP 初始化 --------
    is_ddp = setup_ddp()

    # -------- 2. 加载配置 --------
    try:
        config = load_config(args.config)
        if is_main_process():
            print("[信息] 配置文件加载成功。")
    except Exception as e:
        if is_main_process():
            print(f"[错误] 加载或解析配置文件失败 '{args.config}'：{e}")
        return

    # -------- 3. 环境设置 --------
    # DDP模式下，setup_ddp已经设置了device，config中的 'gpus' 选项将被忽略或作为参考
    # 设置种子
    ddp_setup_seed(config.get('seed', 42))
    torch.backends.cudnn.benchmark = True

    # -------- 4. 结果目录 --------
    # 只有主进程负责创建目录
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    results_basedir = Path(config.get('results_root', './results')) / f"{config.get('run_name', 'run')}_{ts}"

    if is_main_process():
        results_basedir.mkdir(parents=True, exist_ok=True)
        print(f"[信息] 结果将保存到：{results_basedir}")
        backup_code(results_basedir)

    # 等待主进程创建目录
    if is_ddp:
        dist.barrier()

    # -------- 5. 动态加载 Trainer --------
    model_version = config.get('model_version', 'original')
    try:
        if model_version == 'original':
            trainer_module = importlib.import_module('original.trainer')
            TrainerClass = trainer_module.Trainer
            if is_main_process():
                print("[信息] 使用 ORIGINAL 版本的 Trainer。")
    except Exception as e:
        if is_main_process():
            print(f"[错误] 无法导入 Trainer：{e}")
        return

    # -------- 6. 初始化结果存储 --------
    all_lodo_results = []
    # datasets_lodo_order = [
    #      'sleep-edfx','HMC', 'ISRUC', 'SHHS1', 'P2018',
    # ]
    datasets_lodo_order = ['MROS1', 'MROS2', 'MESA', 'SHHS1', 'P2018']


    num_total_datasets = len(datasets_lodo_order)
    expected_num_source = num_total_datasets - 1
    config['num_domains'] = expected_num_source

    # -------- 7. LODO 循环 --------
    for lodo_fold_index, target_dataset_name in enumerate(datasets_lodo_order):
        # DDP同步点：确保所有进程同时开始处理同一个 fold
        if is_ddp:
            dist.barrier()

        if is_main_process():
            print(
                f"\n{'=' * 15} LODO 折 {lodo_fold_index}/{num_total_datasets - 1}：目标域 = {target_dataset_name} {'=' * 15}")

        # -- 构建 params --
        fold_config = config.copy()
        fold_config['target_domains'] = target_dataset_name
        fold_config['model_dir'] = str(results_basedir)
        fold_config['fold'] = lodo_fold_index
        fold_config['run_name'] = f"{config.get('run_name', 'run')}_target_{target_dataset_name}_lodo{lodo_fold_index}"

        # 传递 DDP 标志给 Trainer
        fold_config['is_ddp'] = is_ddp

        params = SimpleNamespace(**fold_config)

        if is_main_process():
            print(f"===== Fold {params.fold} 参数: target={params.target_domains} =====")

        try:
            # 实例化 Trainer
            trainer = TrainerClass(params)

            # 训练
            lodo_result_dict = trainer.train()

            # 仅主进程收集和打印结果
            if is_main_process():
                if lodo_result_dict and isinstance(lodo_result_dict, dict):
                    all_lodo_results.append(lodo_result_dict)
                    acc = lodo_result_dict.get('test_acc', float('nan'))
                    f1 = lodo_result_dict.get('test_f1', float('nan'))
                    print(f"LODO 折 {lodo_fold_index} 结果 -> Acc：{_fmt_metric(acc)}，F1：{_fmt_metric(f1)}")
                else:
                    print(f"[警告] LODO 折 {lodo_fold_index} 无效结果。")

        except Exception as e:
            print(f"[Error in Fold {lodo_fold_index}] {e}")
            import traceback
            traceback.print_exc()
            continue

    # -------- 8. 总结 --------
    # 只有主进程做总结
    if is_main_process():
        print(f"\n======== LODO 整体均值结果 ========")
        aggregate_csv_path = results_basedir / "allfold" / "aggregate_results.csv"
        aggregate_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if all_lodo_results:
            avg_row = {
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "run_id": f"{config.get('run_name', 'run')}_LODO_avg",
                "fold": "mean",
                "model_path": "N/A"
            }
            metrics_to_average = ['test_acc', 'test_f1', 'test_kappa', 'wake_f1', 'n1_f1', 'n2_f1', 'n3_f1', 'rem_f1']

            for key in metrics_to_average:
                mean_val, std_val = calculate_stats(all_lodo_results, key)
                print(f"{key:<15} | {_fmt_metric(mean_val)} ± {_fmt_metric(std_val)}")
                avg_row[key] = f"{_fmt_metric(mean_val)} +/- {_fmt_metric(std_val)}"

            try:
                write_aggregate_row(aggregate_csv_path, row=avg_row)
            except Exception as e:
                print(f"写入 CSV 失败: {e}")

    # 清理 DDP
    cleanup_ddp()


if __name__ == '__main__':
    main()