#!/bin/bash
# 文件名: scripts/vllm/start_server.sh
# 功能: 以 OpenAI 兼容模式启动 vLLM 服务器
# 用法: bash start_server.sh [端口号，默认8000]
# 注意: 启动后服务器在前台运行，测试完成后 Ctrl+C 关闭
#
# 关键参数说明:
#   --dtype float16           与 SGLang 对齐的精度设置
#   --enable-prefix-caching   启用 APC，与 SGLang 的 RadixAttention 对等
#   --max-model-len 4096      测试最长输入约 2500 tokens，4096 足够
#   --gpu-memory-utilization   预分配 90% 显存给 KV Cache + 模型权重
#   --served-model-name       统一模型名称，两框架共用同一测试脚本

set -e

MODEL_PATH=~/llm-benchmark/models/llama-3.1-8b-instruct
PORT=${1:-8000}
LOG_DIR=~/llm-benchmark/logs

mkdir -p "$LOG_DIR"

echo "========================================"
echo "启动 vLLM 服务器"
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "监听端口: $PORT"
echo "日志文件: $LOG_DIR/vllm_server.log"
echo ""

# 检查模型目录是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "✗ 模型目录不存在: $MODEL_PATH"
    echo "  请先运行 download_model.sh"
    exit 1
fi

# 检查端口是否被占用
if lsof -i :"$PORT" &> /dev/null; then
    echo "⚠ 端口 $PORT 已被占用"
    echo "  占用进程:"
    lsof -i :"$PORT" | head -3
    echo "  请关闭占用进程后重试，或指定其他端口"
    exit 1
fi

echo "启动中... （首次加载模型约需 30–90 秒）"
echo "就绪后将显示 'Uvicorn running on ...'"
echo ""

vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --dtype float16 \
    --enable-prefix-caching \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --served-model-name "test-model"
