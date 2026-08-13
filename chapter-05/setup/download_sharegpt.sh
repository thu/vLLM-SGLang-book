#!/bin/bash
# 文件名: scripts/setup/download_sharegpt.sh
# 功能: 下载 ShareGPT 对话数据集
# 用法: bash download_sharegpt.sh

set -e

DATASET_DIR=~/llm-benchmark/datasets/raw

echo "========================================"
echo "下载 ShareGPT 数据集"
echo "========================================"

mkdir -p "$DATASET_DIR"
cd "$DATASET_DIR"

# 检查是否已下载
if [ -f "sharegpt.json" ]; then
    echo "sharegpt.json 已存在，跳过下载"
    echo "若需重新下载，请先删除该文件"
else
    echo "[1/2] 下载 ShareGPT_V3_unfiltered_cleaned_split.json..."
    wget -O sharegpt.json \
        "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
    echo "✓ 下载完成"
fi

# 验证 JSON 格式及基本统计
echo -e "\n[2/2] 验证数据集..."
python3 - << 'PYEOF'
import json
import sys

try:
    with open("sharegpt.json", encoding="utf-8") as f:
        data = json.load(f)
except json.JSONDecodeError as e:
    print(f"✗ JSON 格式错误: {e}")
    sys.exit(1)

if not isinstance(data, list):
    print(f"✗ 预期为列表，实际类型: {type(data)}")
    sys.exit(1)

# 基本统计
turns = [len(d.get("conversations", [])) for d in data]
human_starts = sum(
    1 for d in data
    if d.get("conversations")
    and d["conversations"][0].get("from") == "human"
)

print(f"对话总数        : {len(data)}")
print(f"以 human 开头   : {human_starts}")
print(f"平均消息轮次    : {sum(turns)/len(turns):.1f}")
print(f"最大消息轮次    : {max(turns)}")
print(f"最小消息轮次    : {min(turns)}")
print(f"文件大小        : {sys.getsizeof(data) / 1024 / 1024:.1f} MB (内存)")
print("✓ ShareGPT 数据集格式验证通过")
PYEOF

echo -e "\n========================================"
echo "ShareGPT 数据集准备完成"
echo "文件路径: $DATASET_DIR/sharegpt.json"
echo "========================================"
