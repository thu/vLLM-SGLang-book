#!/bin/bash
# scripts/chapter8/start_sglang_server.sh
# 启动 SGLang Runtime，加载 ChatGLM-3-6B
#
# 测试环境: 单张 A100 80GB, CUDA 12.1, PyTorch 2.1.0, SGLang v0.1.17
# 用法: bash scripts/chapter8/start_sglang_server.sh [模型路径]
#
# 参数说明:
#   --mem-fraction-static 0.85
#       默认值 0.9 在高并发时因 CUDA 内核临时缓冲区不足可能 OOM。
#       0.85 为 12GB 权重 + 约 55GB KV Cache 预留空间，实测稳定。
#   --context-length 8192
#       ChatGLM-3 支持 32K，但本项目 8 轮对话不超过 4K token。
#       限制为 8192 可将 KV Cache 预分配规模减半，提高并发能力。
#   --trust-remote-code
#       ChatGLM-3 分词器含自定义代码，此参数必须添加。
#   --enable-flashinfer
#       启用 FlashInfer 内核加速注意力计算，约降低 15% 推理延迟。

set -euo pipefail

# ── 配置 ─────────────────────────────────────────────────────────────
MODEL_PATH="${1:-${HOME}/models/chatglm3-6b}"
HOST="127.0.0.1"
PORT=30000
LOG_LEVEL="info"
LOG_FILE="logs/sglang_server.log"

# ── 检查 ─────────────────────────────────────────────────────────────
echo "=============================================="
echo "  SGLang Runtime 启动脚本"
echo "  模型路径: ${MODEL_PATH}"
echo "  服务地址: http://${HOST}:${PORT}"
echo "=============================================="

if [ ! -d "${MODEL_PATH}" ]; then
    echo "✗ 模型目录不存在: ${MODEL_PATH}"
    echo "  请先运行: python scripts/chapter8/download_model.py"
    exit 1
fi

if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "✗ 模型文件不完整，缺少 config.json"
    exit 1
fi

mkdir -p logs

echo ""
echo "[1/3] 检查 GPU 状态..."
nvidia-smi --query-gpu=name,memory.total,memory.free \
    --format=csv,noheader,nounits 2>/dev/null \
    | awk -F',' '{printf "  GPU: %s | 总显存: %s MB | 空闲: %s MB\n", $1, $2, $3}' \
    || echo "  警告: 无法获取 GPU 信息"

echo ""
echo "[2/3] 检查端口 ${PORT} 是否空闲..."
if lsof -i ":${PORT}" >/dev/null 2>&1; then
    echo "  警告: 端口 ${PORT} 已被占用，尝试终止旧进程..."
    lsof -ti ":${PORT}" | xargs kill -9 2>/dev/null || true
    sleep 2
fi

echo ""
echo "[3/3] 启动 SGLang Runtime..."
echo "  日志文件: ${LOG_FILE}"
echo "  （首次加载模型约需 60-90 秒）"
echo ""

# 启动服务，日志同时写文件和终端
python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --tp 1 \
    --mem-fraction-static 0.85 \
    --context-length 8192 \
    --trust-remote-code \
    --enable-flashinfer \
    --log-level "${LOG_LEVEL}" \
    2>&1 | tee "${LOG_FILE}"
