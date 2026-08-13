#!/bin/bash
# 文件名: scripts/sglang/start_server.sh
# 功能: 启动 SGLang OpenAI 兼容推理服务器
# 用法: bash start_server.sh [端口号，默认30000]
# 注意: 启动后服务器在前台运行，测试完成后 Ctrl+C 关闭
#
# 关键参数说明:
#   --enable-flashinfer        使用 FlashInfer 高性能注意力后端
#   --enable-prefix-caching    启用 RadixAttention 前缀缓存
#   --chunked-prefill-size 512 分块预填充，降低长输入对解码的阻塞
#   --mem-fraction-static 0.88 88% 显存给 KV Cache（留 2% 给树结构元数据）
#   --served-model-name        统一模型名称，与 vLLM 保持一致
#
# 版本注意:
#   以下参数基于 SGLang 0.5.9 验证。不同版本的参数名可能有差异，
#   建议通过 python -m sglang.launch_server --help 确认当前版本参数。

set -e

MODEL_PATH=~/llm-benchmark/models/llama-3.1-8b-instruct
PORT=${1:-30000}
LOG_DIR=~/llm-benchmark/logs

mkdir -p "$LOG_DIR"

echo "========================================"
echo "启动 SGLang 服务器"
echo "========================================"
echo "模型路径: $MODEL_PATH"
echo "监听端口: $PORT"
echo "日志文件: $LOG_DIR/sglang_server.log"
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

# 检查 FlashInfer 是否可用
echo "检查 FlashInfer..."
if python3 -c "import flashinfer" 2>/dev/null; then
    echo "✓ FlashInfer 可用"
    FLASHINFER_FLAG="--enable-flashinfer"
else
    echo "⚠ FlashInfer 不可用，将使用 Triton 后端（性能约低 10-15%）"
    FLASHINFER_FLAG="--disable-flashinfer"
fi

echo ""
echo "启动中... （首次加载模型约需 30–90 秒）"
echo ""

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --port "$PORT" \
    --dtype float16 \
    $FLASHINFER_FLAG \
    --enable-prefix-caching \
    --chunked-prefill-size 512 \
    --mem-fraction-static 0.88 \
    --served-model-name "test-model"
