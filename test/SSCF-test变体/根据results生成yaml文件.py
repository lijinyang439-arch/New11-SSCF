#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#python 根据results生成yaml文件.py --generate_all_testsets
import re
import yaml
import argparse
from pathlib import Path


CV_TARGET_ORDER = [
    "SHHS1",
    "P2018",
    "MROS1",
    "MROS2",
    "MESA",
]

INDEPENDENT_TEST_DATASETS = [
    "ABC",
    "CCSHS",
    "CFS",
    "HMC",
    "ISRUC",
    "sleep-edfx",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="批量扫描 results 目录并生成 test yaml 文件"
    )

    parser.add_argument(
        "--results_dir",
        type=str,
        default="/data/lijinyang/睡眠分期SSCF的test/人聚类人对齐结果Test/results",
        help="训练结果目录，里面每个一级子目录视为一次实验",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/lijinyang/睡眠分期SSCF的test/人聚类人对齐结果Test/SSCF-test变体/configs",
        help="生成的 yaml 输出目录",
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        default=None,
        help="指定单个测试集，如 MROS2；若不指定，则需配合 --generate_all_testsets 使用",
    )
    parser.add_argument(
        "--generate_all_testsets",
        action="store_true",
        help="是否为全部测试集生成 yaml",
    )

    parser.add_argument(
        "--model_version",
        type=str,
        default="original_sleepdg",
        help="yaml 中 model_version",
    )

    parser.add_argument(
        "--train_config_relpath",
        type=str,
        default="code_backup/configs/config_8_yes.yaml",
        help="每个实验目录内部用于读取 align 配置的相对路径",
    )
    parser.add_argument(
        "--relative_root",
        type=str,
        default="/root/autodl-tmp",
        help="将训练配置中的绝对路径转换为 ./ 相对路径时所依据的根目录",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="1",
        help='GPU 设置，例如 "0" / "1" / "0,1" / ""',
    )
    parser.add_argument(
        "--datasets_dir",
        type=str,
        default="/fdata/lijinyang/datasets_dir2_all_Merged",
        help="数据集根目录",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch_size",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="num_workers",
    )

    parser.add_argument(
        "--import_path",
        type=str,
        default="original.models.model",
        help="yaml 中 model.import_path",
    )
    parser.add_argument(
        "--class_name",
        type=str,
        default="Model",
        help="yaml 中 model.class_name",
    )
    parser.add_argument(
        "--ensemble_method",
        type=str,
        default="logits",
        choices=["logits", "probs"],
        help="yaml 中 model.ensemble_method",
    )
    parser.add_argument(
        "--strict_load",
        type=str,
        default="true",
        help='yaml 中 model.strict_load，传 "true" 或 "false"',
    )
    parser.add_argument(
        "--use_dataparallel",
        type=str,
        default="false",
        help='yaml 中 model.use_dataparallel，传 "true" 或 "false"',
    )

    parser.add_argument(
        "--model_type",
        type=str,
        default="LDP_CAA_Model",
        help="yaml 中 model_type",
    )
    parser.add_argument(
        "--num_of_classes",
        type=int,
        default=5,
        help="yaml 中 num_of_classes",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="yaml 中 dropout",
    )
    parser.add_argument(
        "--projection_type",
        type=str,
        default="diag",
        help="yaml 中 projection_type",
    )
    parser.add_argument(
        "--lowrank_rank",
        type=int,
        default=32,
        help="yaml 中 lowrank_rank",
    )
    parser.add_argument(
        "--enable_stats_alignment",
        type=int,
        default=1,
        help="yaml 中 enable_stats_alignment",
    )
    parser.add_argument(
        "--anchor_momentum",
        type=float,
        default=0.9,
        help="yaml 中 anchor_momentum",
    )
    parser.add_argument(
        "--num_domains",
        type=int,
        default=4,
        help="yaml 中 num_domains",
    )

    return parser.parse_args()


def str_to_bool(s):
    if isinstance(s, bool):
        return s
    s = str(s).strip().lower()
    if s in ["true", "1", "yes", "y"]:
        return True
    if s in ["false", "0", "no", "n"]:
        return False
    raise ValueError(f"无法解析布尔值: {s}")


def extract_ta_from_filename(filename):
    """
    从类似 topk_ep003_va0.8316_ta0.8488.pth 中提取 ta=0.8488
    """
    match = re.search(r"(?:^|_)ta([0-9]*\.?[0-9]+)(?:_|\.pth$)", filename)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except Exception:
        return None


def get_best_ckpt_in_fold(fold_dir):
    """
    在一个 fold 目录中找到 ta 最大的 .pth 文件
    返回: (best_path, best_ta)
    """
    pth_files = sorted([p for p in fold_dir.iterdir() if p.is_file() and p.suffix == ".pth"])
    if not pth_files:
        return None, None

    candidates = []
    for p in pth_files:
        ta = extract_ta_from_filename(p.name)
        if ta is not None:
            candidates.append((p, ta))

    if not candidates:
        return None, None

    candidates.sort(key=lambda x: (-x[1], x[0].name))
    best_path, best_ta = candidates[0]
    return str(best_path), best_ta


def is_valid_experiment_dir(exp_dir):
    """
    判断是否是合法实验目录：
    - 必须有 fold0 ~ fold4
    """
    for i in range(5):
        fold_dir = exp_dir / f"fold{i}"
        if not fold_dir.exists() or not fold_dir.is_dir():
            return False
    return True


def collect_best_ckpts_for_experiment(exp_dir):
    """
    收集某个实验目录下 fold0 ~ fold4 的最佳权重
    返回:
        ckpts(list[str]) 或 None
        info(list[tuple]) 记录每个fold的选择信息
        err_msg(str) 若失败则返回失败原因
    """
    ckpts = []
    info = []

    for i in range(5):
        fold_dir = exp_dir / f"fold{i}"
        best_ckpt, best_ta = get_best_ckpt_in_fold(fold_dir)
        if best_ckpt is None:
            return None, None, f"{fold_dir} 下没有可解析 ta 的 .pth 文件"
        ckpts.append(best_ckpt)
        info.append((f"fold{i}", best_ckpt, best_ta))

    return ckpts, info, None


def get_experiment_prefix(exp_name):
    """
    从实验目录名中取第一个 '_' 前面的前缀
    例如:
        Ours-k1_2026-03-16_08-57-53 -> Ours-k1
        Baseline_2026-03-18_09-00-00 -> Baseline
    """
    if "_" in exp_name:
        return exp_name.split("_")[0]
    return exp_name


def get_unique_output_path(output_dir, base_filename):
    """
    若文件存在，则自动追加 _1, _2 ...
    """
    output_path = output_dir / base_filename
    if not output_path.exists():
        return output_path

    stem = output_path.stem
    suffix = output_path.suffix
    idx = 1
    while True:
        candidate = output_dir / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def convert_abs_path_to_dot_relative(path_str, relative_root):
    """
    把类似:
        /root/autodl-tmp/cluster/cluster_output_direct1/ours1_centers.npy
    转成:
        ./cluster/cluster_output_direct1/ours1_centers.npy

    如果 path_str 不在 relative_root 下，则原样返回。
    """
    if path_str is None:
        return None

    path_str = str(path_str).strip()
    relative_root = str(relative_root).rstrip("/")

    prefix = relative_root + "/"
    if path_str.startswith(prefix):
        return "." + path_str[len(relative_root):]

    return path_str


def find_train_config_path(exp_dir, train_config_relpath):
    """
    默认按固定相对路径找训练配置文件
    """
    config_path = exp_dir / train_config_relpath
    if config_path.exists() and config_path.is_file():
        return config_path
    return None


def read_align_from_train_config(exp_dir, train_config_relpath, relative_root):
    """
    从每个实验自己的训练配置里读取:
        align.use_alignment
        align.cluster_centers_path
        align.cluster_map_path

    并把两个 cluster 路径转换成 ./ 开头的相对路径
    """
    config_path = find_train_config_path(exp_dir, train_config_relpath)
    if config_path is None:
        return None, f"未找到训练配置文件: {exp_dir / train_config_relpath}"

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            train_cfg = yaml.safe_load(f)
    except Exception as e:
        return None, f"读取训练配置失败: {config_path} | {e}"

    if not isinstance(train_cfg, dict):
        return None, f"训练配置内容不是字典结构: {config_path}"

    align_cfg = train_cfg.get("align", None)
    if not isinstance(align_cfg, dict):
        return None, f"训练配置中缺少 align 字段: {config_path}"

    use_alignment = align_cfg.get("use_alignment", True)
    cluster_centers_path = align_cfg.get("cluster_centers_path", None)
    cluster_map_path = align_cfg.get("cluster_map_path", None)

    if cluster_centers_path is None:
        return None, f"训练配置中缺少 align.cluster_centers_path: {config_path}"
    if cluster_map_path is None:
        return None, f"训练配置中缺少 align.cluster_map_path: {config_path}"

    align_out = {
        "use_alignment": bool(use_alignment),
        "cluster_centers_path": convert_abs_path_to_dot_relative(cluster_centers_path, relative_root),
        "cluster_map_path": convert_abs_path_to_dot_relative(cluster_map_path, relative_root),
    }

    return align_out, None


def build_yaml_config(args, exp_dir, test_dataset, ckpts, align_cfg):
    config = {
        "model_version": args.model_version,
        "results_log_dir": str(exp_dir),

        "test_dataset": test_dataset,
        "align": align_cfg,

        "seed": args.seed,
        "gpus": args.gpus,

        "datasets_dir": args.datasets_dir,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,

        "cv_target_order": CV_TARGET_ORDER,
        "independent_test_datasets": INDEPENDENT_TEST_DATASETS,

        "model": {
            "import_path": args.import_path,
            "class_name": args.class_name,
            "ckpts": ckpts,
            "ensemble_method": args.ensemble_method,
            "strict_load": str_to_bool(args.strict_load),
            "use_dataparallel": str_to_bool(args.use_dataparallel),
        },

        "model_type": args.model_type,
        "num_of_classes": args.num_of_classes,
        "dropout": args.dropout,

        "projection_type": args.projection_type,
        "lowrank_rank": args.lowrank_rank,
        "enable_stats_alignment": args.enable_stats_alignment,
        "anchor_momentum": args.anchor_momentum,

        "num_domains": args.num_domains,
    }
    return config


def dump_yaml(config, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        yaml.dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )


def get_target_test_datasets(args):
    if args.generate_all_testsets:
        return CV_TARGET_ORDER + INDEPENDENT_TEST_DATASETS

    if args.test_dataset is None:
        raise ValueError("未指定 --test_dataset，且未开启 --generate_all_testsets")

    all_valid = set(CV_TARGET_ORDER + INDEPENDENT_TEST_DATASETS)
    if args.test_dataset not in all_valid:
        raise ValueError(
            f"test_dataset={args.test_dataset} 不合法，可选为: {sorted(all_valid)}"
        )

    return [args.test_dataset]


def main():
    args = parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_dir.exists() or not results_dir.is_dir():
        raise FileNotFoundError(f"results_dir 不存在或不是目录: {results_dir}")

    target_test_datasets = get_target_test_datasets(args)
    exp_dirs = sorted([p for p in results_dir.iterdir() if p.is_dir()])

    print("=" * 100)
    print("开始批量生成 test yaml")
    print(f"results_dir          : {results_dir}")
    print(f"output_dir           : {output_dir}")
    print(f"train_config_relpath : {args.train_config_relpath}")
    print(f"relative_root        : {args.relative_root}")
    print(f"目标测试集            : {target_test_datasets}")
    print("=" * 100)

    total_exp = 0
    valid_exp = 0
    generated_yaml_count = 0
    skipped_exp_count = 0

    for exp_dir in exp_dirs:
        total_exp += 1
        exp_name = exp_dir.name
        print("\n" + "-" * 100)
        print(f"[实验] {exp_name}")

        if not is_valid_experiment_dir(exp_dir):
            skipped_exp_count += 1
            print(f"[跳过] 不是合法实验目录，缺少 fold0 ~ fold4: {exp_dir}")
            continue

        ckpts, info, err_msg = collect_best_ckpts_for_experiment(exp_dir)
        if ckpts is None:
            skipped_exp_count += 1
            print(f"[跳过] {err_msg}")
            continue

        align_cfg, align_err = read_align_from_train_config(
            exp_dir=exp_dir,
            train_config_relpath=args.train_config_relpath,
            relative_root=args.relative_root,
        )
        if align_cfg is None:
            skipped_exp_count += 1
            print(f"[跳过] {align_err}")
            continue

        valid_exp += 1

        print("[已选择的最佳权重]")
        for fold_name, best_ckpt, best_ta in info:
            print(f"  {fold_name}: ta={best_ta:.4f} | {best_ckpt}")

        print("[读取到的 align 配置]")
        print(f"  use_alignment      : {align_cfg['use_alignment']}")
        print(f"  cluster_centers    : {align_cfg['cluster_centers_path']}")
        print(f"  cluster_map        : {align_cfg['cluster_map_path']}")

        exp_prefix = get_experiment_prefix(exp_name)

        for test_dataset in target_test_datasets:
            config = build_yaml_config(
                args=args,
                exp_dir=exp_dir,
                test_dataset=test_dataset,
                ckpts=ckpts,
                align_cfg=align_cfg,
            )

            base_filename = f"Test-{exp_prefix}-{test_dataset}.yaml"
            save_path = get_unique_output_path(output_dir, base_filename)

            dump_yaml(config, save_path)
            generated_yaml_count += 1
            print(f"[生成成功] {save_path}")

    print("\n" + "=" * 100)
    print("生成完成")
    print(f"扫描实验目录总数: {total_exp}")
    print(f"合法实验目录数  : {valid_exp}")
    print(f"跳过实验目录数  : {skipped_exp_count}")
    print(f"生成 yaml 总数  : {generated_yaml_count}")
    print("=" * 100)


if __name__ == "__main__":
    main()