#!/usr/bin/env bash
# ch06/scripts/download_model.sh
# 下载 Qwen2.5-7B-Instruct-AWQ 模型权重
# 依赖：huggingface_hub（pip install huggingface_hub）
# 运行：bash ch06/scripts/download_model.sh
#
# 说明：--local-dir-use-symlinks False 确保文件以实体形式存储，
#       避免 symlink 在 Docker 容器或跨挂载点环境中失效。

set -euo pipefail

MODEL_ID="Qwen/Qwen2.5-7B-Instruct-AWQ"
LOCAL_DIR="${CH06_MODEL_DIR:-$HOME/models/Qwen2.5-7B-Instruct-AWQ}"

echo "=== 下载模型 ==="
echo "来源: ${MODEL_ID}"
echo "目标: ${LOCAL_DIR}"
echo ""

# 检查是否已存在
if [ -f "${LOCAL_DIR}/config.json" ]; then
    echo "模型目录已存在，跳过下载。"
    echo "如需重新下载，请先删除: ${LOCAL_DIR}"
    exit 0
fi

mkdir -p "${LOCAL_DIR}"

huggingface-cli download \
    "${MODEL_ID}" \
    --local-dir "${LOCAL_DIR}" \
    --local-dir-use-symlinks False

# 验证关键文件存在
for f in config.json tokenizer.json tokenizer_config.json; do
    if [ ! -f "${LOCAL_DIR}/${f}" ]; then
        echo "错误：缺少文件 ${f}，下载可能不完整"
        exit 1
    fi
done

echo ""
echo "=== 下载完成 ==="
echo "模型路径: ${LOCAL_DIR}"
echo "文件列表:"
ls -lh "${LOCAL_DIR}"
