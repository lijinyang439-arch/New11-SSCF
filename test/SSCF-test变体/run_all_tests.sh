#!/usr/bin/env bash
set -u
#
#chmod +x run_all_tests.sh
#./run_all_tests.sh
# ==============================
# 你自己改这几个参数
# ==============================

PYTHON_BIN="python"
TEST_PY="test.py"
CONFIG_DIR="configs"
LOG_DIR="logs_batch_test"

# 要跑的 k
K_LIST=(2 3 4 5 6 7 8)

# 要跑的数据集
DATASETS=(
  "MESA"
  "SHHS1"
  "P2018"
  "MROS1"
  "MROS2"
  "ABC"
  "CCSHS"
  "CFS"
  "HMC"
  "ISRUC"
  "sleep-edfx"
)

# 可用 GPU
GPU_LIST=(0 1)

# 配置命名规则
# configs/Test-Ours-k2-MESA.yaml
CONFIG_PREFIX="Test-Ours-k"
CONFIG_SUFFIX=".yaml"

# 轮询间隔（秒）
SLEEP_SEC=2

# ==============================
# 初始化
# ==============================

mkdir -p "${LOG_DIR}"

SUCCESS_LOG="${LOG_DIR}/success_tasks.txt"
FAILED_LOG="${LOG_DIR}/failed_tasks.txt"
SKIPPED_LOG="${LOG_DIR}/skipped_tasks.txt"
ALL_LOG="${LOG_DIR}/all_tasks.txt"

: > "${SUCCESS_LOG}"
: > "${FAILED_LOG}"
: > "${SKIPPED_LOG}"
: > "${ALL_LOG}"

echo "=========================================="
echo "批量测试开始"
echo "PYTHON_BIN      = ${PYTHON_BIN}"
echo "TEST_PY         = ${TEST_PY}"
echo "CONFIG_DIR      = ${CONFIG_DIR}"
echo "LOG_DIR         = ${LOG_DIR}"
echo "K_LIST          = ${K_LIST[*]}"
echo "DATASETS        = ${DATASETS[*]}"
echo "GPU_LIST        = ${GPU_LIST[*]}"
echo "=========================================="

# ==============================
# 组装任务列表
# ==============================

TASK_CONFIGS=()
TASK_NAMES=()

TASK_COUNT=0
VALID_TASK_COUNT=0

for k in "${K_LIST[@]}"; do
    for dataset in "${DATASETS[@]}"; do
        TASK_COUNT=$((TASK_COUNT + 1))

        config_path="${CONFIG_DIR}/${CONFIG_PREFIX}${k}-${dataset}${CONFIG_SUFFIX}"
        task_name="k${k}-${dataset}"

        echo "${task_name} | CONFIG=${config_path}" >> "${ALL_LOG}"

        if [ ! -f "${config_path}" ]; then
            echo "[跳过] 配置不存在: ${config_path}"
            echo "${task_name} | CONFIG=${config_path}" >> "${SKIPPED_LOG}"
            continue
        fi

        TASK_CONFIGS+=("${config_path}")
        TASK_NAMES+=("${task_name}")
        VALID_TASK_COUNT=$((VALID_TASK_COUNT + 1))
    done
done

echo "总组合数   : ${TASK_COUNT}"
echo "有效任务数 : ${VALID_TASK_COUNT}"

if [ "${VALID_TASK_COUNT}" -eq 0 ]; then
    echo "[错误] 没有可运行的配置文件"
    exit 1
fi

# ==============================
# 每张 GPU 只跑 1 个任务
# 显式维护 GPU -> PID / TASK / LOG
# ==============================

declare -A GPU_TO_PID
declare -A GPU_TO_TASK
declare -A GPU_TO_LOG

for gpu in "${GPU_LIST[@]}"; do
    GPU_TO_PID["$gpu"]=""
    GPU_TO_TASK["$gpu"]=""
    GPU_TO_LOG["$gpu"]=""
done

NEXT_TASK_IDX=0
SUCCESS_COUNT=0
FAILED_COUNT=0

# ==============================
# 启动一个任务到指定 GPU
# ==============================
launch_task_on_gpu() {
    local gpu="$1"
    local task_idx="$2"

    local task_name="${TASK_NAMES[$task_idx]}"
    local config_path="${TASK_CONFIGS[$task_idx]}"
    local log_file="${LOG_DIR}/${task_name}.log"

    echo "[启动] ${task_name} -> GPU ${gpu}"
    echo "        config: ${config_path}"
    echo "        log   : ${log_file}"

    (
        export CUDA_VISIBLE_DEVICES="${gpu}"
        export OMP_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

        "${PYTHON_BIN}" "${TEST_PY}" --config "${config_path}"
    ) > "${log_file}" 2>&1 &

    local pid=$!

    GPU_TO_PID["$gpu"]="${pid}"
    GPU_TO_TASK["$gpu"]="${task_name}"
    GPU_TO_LOG["$gpu"]="${log_file}"
}

# ==============================
# 检查某张 GPU 上的任务是否结束
# 结束了就 wait 一次并释放 GPU
# ==============================
check_gpu_job() {
    local gpu="$1"
    local pid="${GPU_TO_PID[$gpu]}"

    if [ -z "${pid}" ]; then
        return 1
    fi

    if kill -0 "${pid}" 2>/dev/null; then
        return 1
    fi

    local task_name="${GPU_TO_TASK[$gpu]}"
    local log_file="${GPU_TO_LOG[$gpu]}"

    wait "${pid}"
    local exit_code=$?

    if [ "${exit_code}" -eq 0 ]; then
        echo "[完成] ${task_name} | GPU ${gpu}"
        echo "${task_name} | GPU=${gpu} | LOG=${log_file}" >> "${SUCCESS_LOG}"
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        echo "[失败] ${task_name} | GPU ${gpu} | exit_code=${exit_code}"
        echo "${task_name} | GPU=${gpu} | LOG=${log_file} | EXIT_CODE=${exit_code}" >> "${FAILED_LOG}"
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi

    GPU_TO_PID["$gpu"]=""
    GPU_TO_TASK["$gpu"]=""
    GPU_TO_LOG["$gpu"]=""

    return 0
}

# ==============================
# 主调度循环
# 逻辑：
# 1. 先尽量把空闲 GPU 填满
# 2. 再轮询检查哪些 GPU 跑完了
# 3. 跑完就继续塞新任务
# ==============================

while true; do
    # 先给空闲 GPU 塞任务
    for gpu in "${GPU_LIST[@]}"; do
        if [ -z "${GPU_TO_PID[$gpu]}" ] && [ "${NEXT_TASK_IDX}" -lt "${VALID_TASK_COUNT}" ]; then
            launch_task_on_gpu "${gpu}" "${NEXT_TASK_IDX}"
            NEXT_TASK_IDX=$((NEXT_TASK_IDX + 1))
        fi
    done

    # 判断是否全部结束
    all_done=1

    if [ "${NEXT_TASK_IDX}" -lt "${VALID_TASK_COUNT}" ]; then
        all_done=0
    fi

    for gpu in "${GPU_LIST[@]}"; do
        if [ -n "${GPU_TO_PID[$gpu]}" ]; then
            all_done=0
            break
        fi
    done

    if [ "${all_done}" -eq 1 ]; then
        break
    fi

    # 检查已完成任务
    for gpu in "${GPU_LIST[@]}"; do
        if [ -n "${GPU_TO_PID[$gpu]}" ]; then
            check_gpu_job "${gpu}" || true
        fi
    done

    sleep "${SLEEP_SEC}"
done

echo "=========================================="
echo "全部任务处理结束"
echo "总组合数       : ${TASK_COUNT}"
echo "有效任务数     : ${VALID_TASK_COUNT}"
echo "成功任务数     : ${SUCCESS_COUNT}"
echo "失败任务数     : ${FAILED_COUNT}"
echo "成功任务日志   : ${SUCCESS_LOG}"
echo "失败任务日志   : ${FAILED_LOG}"
echo "跳过任务日志   : ${SKIPPED_LOG}"
echo "=========================================="