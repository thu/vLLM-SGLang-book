#!/bin/bash
# 文件名: scripts/setup/env_manager.sh
# 功能: 快速切换 vLLM / SGLang 测试环境
# 用法: source env_manager.sh [vllm|sglang|info]
# 注意: 必须使用 source 执行（而非 bash），否则环境变量不会在当前
#       shell 中生效

PROJECT_ROOT=~/llm-benchmark

_activate_vllm() {
    # 激活 vLLM 虚拟环境并设置框架标识变量
    if [ ! -d "$PROJECT_ROOT/envs/vllm" ]; then
        echo "✗ vLLM 环境不存在，请先运行 install_vllm.sh"
        return 1
    fi
    source "$PROJECT_ROOT/envs/vllm/bin/activate"
    export ACTIVE_FRAMEWORK="vllm"
    export ACTIVE_PORT=8000
    echo "✓ vLLM 环境已激活（端口 $ACTIVE_PORT）"
}

_activate_sglang() {
    # 激活 SGLang 虚拟环境并设置框架标识变量
    if [ ! -d "$PROJECT_ROOT/envs/sglang" ]; then
        echo "✗ SGLang 环境不存在，请先运行 install_sglang.sh"
        return 1
    fi
    source "$PROJECT_ROOT/envs/sglang/bin/activate"
    export ACTIVE_FRAMEWORK="sglang"
    export ACTIVE_PORT=30000
    echo "✓ SGLang 环境已激活（端口 $ACTIVE_PORT）"
}

_show_info() {
    # 打印当前激活环境的关键信息
    echo "========================================"
    echo "当前环境信息"
    echo "========================================"

    if [ -z "$VIRTUAL_ENV" ]; then
        echo "虚拟环境 : 未激活"
    else
        echo "虚拟环境 : $VIRTUAL_ENV"
    fi

    echo "Python   : $(python3 --version 2>&1)"
    echo "框架     : ${ACTIVE_FRAMEWORK:-未知}"
    echo "端口     : ${ACTIVE_PORT:-未设置}"

    if command -v nvidia-smi &> /dev/null; then
        gpu_name=$(nvidia-smi --query-gpu=name \
            --format=csv,noheader | head -1)
        gpu_mem=$(nvidia-smi --query-gpu=memory.used,memory.total \
            --format=csv,noheader | head -1)
        echo "GPU      : $gpu_name"
        echo "显存     : $gpu_mem"
    fi
}

case "$1" in
    vllm)    _activate_vllm   ;;
    sglang)  _activate_sglang ;;
    info)    _show_info        ;;
    *)
        echo "用法: source env_manager.sh [vllm|sglang|info]"
        echo ""
        echo "  vllm   - 激活 vLLM 测试环境"
        echo "  sglang - 激活 SGLang 测试环境"
        echo "  info   - 显示当前环境信息"
        ;;
esac
