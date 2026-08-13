#!/bin/bash
# 文件名: scripts/run_full_benchmark.sh
# 功能: 依次启动 vLLM 和 SGLang 服务器并执行全场景基准测试
# 用法: bash run_full_benchmark.sh
# 说明: 脚本会在后台启动服务器进程，测试完成后自动关闭
#       全流程包含：vLLM 测试 → GPU 冷却 → SGLang 测试 → 对比分析
#       预计总耗时约 1–2 小时（取决于 GPU 型号和样本数量）

set -e

PROJECT_ROOT=~/llm-benchmark
SCENARIOS="short_short short_long long_short long_long single_turn multi_turn"
NUM_SAMPLES=100
COOLDOWN_SECONDS=300

echo "========================================"
echo "完整基准测试流水线"
echo "开始时间: $(date)"
echo "项目目录: $PROJECT_ROOT"
echo "测试场景: $SCENARIOS"
echo "样本数量: $NUM_SAMPLES"
echo "========================================"

# ------------------------------------------------------------------
# 辅助函数
# ------------------------------------------------------------------

wait_for_server() {
    # 等待服务器 /health 端点可用
    local url="$1"
    local max_wait=120
    local elapsed=0
    echo -n "等待服务器启动（最多 ${max_wait}s）"
    while ! curl -sf "${url}" > /dev/null 2>&1; do
        sleep 1
        elapsed=$((elapsed + 1))
        echo -n "."
        if [ "$elapsed" -ge "$max_wait" ]; then
            echo ""
            echo "✗ 服务器启动超时，请检查日志:"
            echo "  tail -50 $2"
            exit 1
        fi
    done
    echo ""
    echo "✓ 服务器已就绪（耗时 ${elapsed}s）"
}

cleanup_server() {
    # 安全关闭服务器进程
    local pid=$1
    local name=$2
    if kill -0 "$pid" 2>/dev/null; then
        echo "关闭 $name 服务器（PID: $pid）..."
        kill "$pid" 2>/dev/null || true
        # 等待进程退出，最多 30 秒
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ "$waited" -lt 30 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "  进程未响应，强制终止..."
            kill -9 "$pid" 2>/dev/null || true
        fi
        echo "✓ $name 服务器已关闭"
    fi
}

check_gpu_idle() {
    # 检查 GPU 是否空闲（无其他进程占用）
    local gpu_procs=$(nvidia-smi --query-compute-apps=pid \
        --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -gt 0 ]; then
        echo "⚠ 检测到 $gpu_procs 个 GPU 进程正在运行:"
        nvidia-smi --query-compute-apps=pid,name,used_memory \
            --format=csv,noheader
        echo "  建议关闭其他 GPU 进程后再执行测试"
        read -p "  是否继续？(y/N) " confirm
        if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
            exit 0
        fi
    fi
}

# ------------------------------------------------------------------
# 预检查
# ------------------------------------------------------------------

echo -e "\n[预检查] 验证环境..."

# 检查虚拟环境
[ -d "$PROJECT_ROOT/envs/vllm" ] \
    || { echo "✗ vLLM 环境不存在"; exit 1; }
[ -d "$PROJECT_ROOT/envs/sglang" ] \
    || { echo "✗ SGLang 环境不存在"; exit 1; }

# 检查测试数据
[ -d "$PROJECT_ROOT/datasets/processed" ] \
    || { echo "✗ 测试数据不存在，请先运行 preprocess_datasets.py"; exit 1; }

# 检查 GPU 状态
check_gpu_idle

echo "✓ 预检查通过"

# ------------------------------------------------------------------
# 第一阶段：vLLM 测试
# ------------------------------------------------------------------

echo -e "\n========================================"
echo "[阶段 1/3] vLLM 测试"
echo "========================================"

source "$PROJECT_ROOT/envs/vllm/bin/activate"

# 记录 GPU 温度基线
gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu \
    --format=csv,noheader | head -1)
echo "GPU 起始温度: ${gpu_temp}°C"

# 后台启动 vLLM 服务器
VLLM_LOG="$PROJECT_ROOT/logs/vllm_server.log"
bash "$PROJECT_ROOT/scripts/vllm/start_server.sh" 8000 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "vLLM 服务器 PID: $VLLM_PID"

wait_for_server "http://127.0.0.1:8000/health" "$VLLM_LOG"

# 对每个场景执行测试
for scenario in $SCENARIOS; do
    echo -e "\n  ── 测试场景: $scenario ──"
    python "$PROJECT_ROOT/scripts/vllm/benchmark_vllm.py" \
        --scenario "$scenario" \
        --num-samples "$NUM_SAMPLES" \
        --port 8000 \
        --monitor
    # 场景间短暂停顿，让 KV Cache 驱逐归零
    sleep 5
done

# 关闭 vLLM 服务器
cleanup_server $VLLM_PID "vLLM"
deactivate

# ------------------------------------------------------------------
# GPU 冷却
# ------------------------------------------------------------------

gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu \
    --format=csv,noheader | head -1)
echo -e "\n当前 GPU 温度: ${gpu_temp}°C"
echo "等待 $(($COOLDOWN_SECONDS / 60)) 分钟让 GPU 降温..."

sleep $COOLDOWN_SECONDS

gpu_temp=$(nvidia-smi --query-gpu=temperature.gpu \
    --format=csv,noheader | head -1)
echo "冷却后 GPU 温度: ${gpu_temp}°C"

# ------------------------------------------------------------------
# 第二阶段：SGLang 测试
# ------------------------------------------------------------------

echo -e "\n========================================"
echo "[阶段 2/3] SGLang 测试"
echo "========================================"

source "$PROJECT_ROOT/envs/sglang/bin/activate"

SGLANG_LOG="$PROJECT_ROOT/logs/sglang_server.log"
bash "$PROJECT_ROOT/scripts/sglang/start_server.sh" 30000 \
    > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
echo "SGLang 服务器 PID: $SGLANG_PID"

wait_for_server "http://127.0.0.1:30000/health" "$SGLANG_LOG"

for scenario in $SCENARIOS; do
    echo -e "\n  ── 测试场景: $scenario ──"
    python "$PROJECT_ROOT/scripts/sglang/benchmark_sglang.py" \
        --scenario "$scenario" \
        --num-samples "$NUM_SAMPLES" \
        --port 30000 \
        --monitor
    sleep 5
done

cleanup_server $SGLANG_PID "SGLang"
deactivate

# ------------------------------------------------------------------
# 第三阶段：对比分析
# ------------------------------------------------------------------

echo -e "\n========================================"
echo "[阶段 3/3] 生成对比报告"
echo "========================================"

# 对比分析不需要 GPU，使用任一环境均可
source "$PROJECT_ROOT/envs/vllm/bin/activate"

python "$PROJECT_ROOT/scripts/common/compare_results.py"
python "$PROJECT_ROOT/scripts/common/visualize_results.py"

deactivate

# ------------------------------------------------------------------
# 完成
# ------------------------------------------------------------------

echo -e "\n========================================"
echo "✓ 全部测试完成"
echo "========================================"
echo "结束时间: $(date)"
echo ""
echo "结果目录:"
echo "  vLLM 原始数据  : $PROJECT_ROOT/results/vllm/"
echo "  SGLang 原始数据 : $PROJECT_ROOT/results/sglang/"
echo "  对比报告       : $PROJECT_ROOT/results/comparison/"
echo "  可视化图表     : $PROJECT_ROOT/results/figures/"
echo ""
echo "查看对比摘要:"
echo "  cat $PROJECT_ROOT/results/comparison/summary.json"
