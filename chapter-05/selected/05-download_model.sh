#!/bin/bash
# 文件名: scripts/setup/download_model.sh
# 功能: 下载 Meta-Llama-3.1-8B-Instruct 模型
# 用法: bash download_model.sh
# 前提: 已在 huggingface.co 申请 Llama 3.1 访问权限
#       并通过 huggingface-cli login 完成认证

set -e

MODEL_ID="meta-llama/Meta-Llama-3.1-8B-Instruct"
MODEL_DIR=~/llm-benchmark/models/llama-3.1-8b-instruct

echo "========================================"
echo "下载 $MODEL_ID"
echo "========================================"

# 创建模型目录
mkdir -p "$MODEL_DIR"

# 检查 huggingface-cli 是否可用
if ! command -v huggingface-cli &> /dev/null; then
    echo "安装 huggingface_hub..."
    pip install huggingface_hub
fi

# 检查是否已登录
echo -e "\n[1/3] 检查 HuggingFace 认证..."
if ! huggingface-cli whoami &> /dev/null; then
    echo "请先执行 huggingface-cli login 完成认证"
    echo "需要在 https://huggingface.co/settings/tokens 创建 token"
    exit 1
fi
echo "✓ 已通过认证"

# 下载模型
echo -e "\n[2/3] 开始下载模型（约 16 GB）..."
echo "目标路径: $MODEL_DIR"
echo "请耐心等待..."

huggingface-cli download \
    "$MODEL_ID" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

# 验证关键文件存在
echo -e "\n[3/3] 验证模型完整性..."

required_files=(
    "config.json"
    "tokenizer.json"
    "tokenizer_config.json"
    "generation_config.json"
)

all_ok=true
for f in "${required_files[@]}"; do
    if [ -f "$MODEL_DIR/$f" ]; then
        echo "  ✓ $f"
    else
        echo "  ✗ $f 缺失"
        all_ok=false
    fi
done

# 检查权重文件
weight_count=$(find "$MODEL_DIR" \
    \( -name "*.safetensors" -o -name "pytorch_model*.bin" \) \
    | wc -l)
echo "  ✓ 找到 $weight_count 个权重文件"

if [ "$weight_count" -eq 0 ]; then
    echo "✗ 未找到权重文件，下载可能不完整"
    all_ok=false
fi

# 统计模型总大小
total_size=$(du -sh "$MODEL_DIR" | cut -f1)
echo "  模型总大小: $total_size"

if $all_ok; then
    echo -e "\n✓ 模型下载验证通过"
    echo "模型路径: $MODEL_DIR"
else
    echo -e "\n✗ 模型验证失败，请检查下载是否完整"
    exit 1
fi
