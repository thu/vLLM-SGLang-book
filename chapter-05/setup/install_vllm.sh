#!/bin/bash
# 文件名: scripts/setup/install_vllm.sh
# 功能: 在独立虚拟环境中安装 vLLM 0.8.x 及全部辅助依赖
# 用法: bash install_vllm.sh

set -e

PROJECT_ROOT=~/llm-benchmark

echo "========================================"
echo "安装 vLLM 0.8.x"
echo "========================================"

# 创建独立虚拟环境
echo -e "\n[1/4] 创建虚拟环境..."
python3 -m venv "$PROJECT_ROOT/envs/vllm"
source "$PROJECT_ROOT/envs/vllm/bin/activate"
pip install --upgrade pip setuptools wheel

# 安装 vLLM
# pip 会自动解析并安装匹配的 PyTorch 版本
# 若网络较慢，可添加 --index-url 指向国内镜像
echo -e "\n[2/4] 安装 vLLM..."
pip install vllm==0.8.5

# 安装测试与分析辅助库
echo -e "\n[3/4] 安装辅助依赖..."
pip install \
    numpy \
    pandas \
    matplotlib \
    seaborn \
    requests \
    aiohttp \
    tqdm \
    psutil \
    nvidia-ml-py \
    transformers \
    openai

# 锁定依赖版本以保证可重复性
pip freeze > "$PROJECT_ROOT/configs/vllm_requirements.txt"
echo "依赖清单已写入 configs/vllm_requirements.txt"

# 验证安装
echo -e "\n[4/4] 验证安装..."
python3 << 'PYEOF'
import sys
import torch
import vllm
from vllm import LLM, SamplingParams

print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"vLLM    : {vllm.__version__}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA 不可用，vLLM 将无法执行 GPU 推理")
    sys.exit(1)

print("✓ vLLM 安装验证通过")
PYEOF

deactivate

echo -e "\n========================================"
echo "vLLM 安装完成"
echo "激活环境: source $PROJECT_ROOT/envs/vllm/bin/activate"
echo "========================================"
