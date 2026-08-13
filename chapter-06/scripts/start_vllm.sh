#!/usr/bin/env bash
# ch06/scripts/start_vllm.sh
# 启动 vLLM 推理服务
#
# 依赖版本：vLLM 0.6.3
# 硬件预设：2×A100 80GB
# 模型：Qwen2.5-7B-Instruct-AWQ
#
# 用法：
#   bash ch06/scripts/start_vllm.sh              # 前台运行
#   bash ch06/scripts/start_vllm.sh --daemon     # 后台运行，日志写入 logs/vllm.log
#
# 可通过环境变量覆盖关键参数：
#   VLLM_MAX_NUM_SEQS=320 bash start_vllm.sh
#   VLLM_MODEL_DIR=/path/to/local/model bash start_vllm.sh

set -euo pipefail

# ---------- 参数配置（可通过环境变量覆盖）----------
MODEL="${VLLM_MODEL_DIR:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
TENSOR_PARALLEL="${VLLM_TP_SIZE:-2}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.90}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_BATCHED_TOKENS="${VLLM_MAX_BATCHED_TOKENS:-8192}"
PORT="${VLLM_PORT:-8000}"
LOG_DIR="${CH06_ROOT:-$HOME/ch06}/logs"
# ---------------------------------------------------

mkdir -p "${LOG_DIR}"

CMD=(
    vllm serve "${MODEL}"
    --quantization awq
    --dtype float16
    --tensor-parallel-size "${TENSOR_PARALLEL}"
    --gpu-memory-utilization "${GPU_MEM_UTIL}"
    --max-num-seqs "${MAX_NUM_SEQS}"
    --max-model-len "${MAX_MODEL_LEN}"
    --max-num-batched-tokens "${MAX_BATCHED_TOKENS}"
    --port "${PORT}"
    --host 0.0.0.0
    --enable-prefix-caching
    --disable-log-requests
    --served-model-name Qwen2.5-7B-Instruct-AWQ
)

echo "=== 启动 vLLM 服务 ==="
echo "模型:            ${MODEL}"
echo "张量并行度:      ${TENSOR_PARALLEL}"
echo "最大并发序列数:  ${MAX_NUM_SEQS}"
echo "显存利用率:      ${GPU_MEM_UTIL}"
echo "最大序列长度:    ${MAX_MODEL_LEN}"
echo "端口:            ${PORT}"
echo ""

if [ "${1:-}" = "--daemon" ]; then
    LOG_FILE="${LOG_DIR}/vllm_$(date +%Y%m%d_%H%M%S).log"
    echo "后台模式，日志: ${LOG_FILE}"
    nohup "${CMD[@]}" > "${LOG_FILE}" 2>&1 &
    echo "进程 PID: $!"
else
    exec "${CMD[@]}"
fi
