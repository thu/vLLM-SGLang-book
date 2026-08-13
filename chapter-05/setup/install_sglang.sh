#!/bin/bash
# 文件名: scripts/setup/install_sglang.sh
# 功能: 在独立虚拟环境中安装 SGLang 0.5.x 及 FlashInfer
# 用法: bash install_sglang.sh

set -e

PROJECT_ROOT=~/llm-benchmark

echo "========================================"
echo "安装 SGLang 0.5.x"
echo "========================================"

# 创建独立虚拟环境
echo -e "\n[1/5] 创建虚拟环境..."
python3 -m venv "$PROJECT_ROOT/envs/sglang"
source "$PROJECT_ROOT/envs/sglang/bin/activate"
pip install --upgrade pip setuptools wheel

# 安装 SGLang（含全部可选依赖）
echo -e "\n[2/5] 安装 SGLang..."
pip install "sglang[all]==0.5.9"

# 安装 FlashInfer
# FlashInfer 预编译包与 CUDA 版本和 PyTorch 版本严格绑定
# 此处使用 CUDA 12.4 + PyTorch 2.5 的组合
# 若版本不匹配会导致运行时 ImportError 或段错误
echo -e "\n[3/5] 安装 FlashInfer（CUDA 12.4 / PyTorch 2.5）..."
pip install flashinfer-python \
    -i https://flashinfer.ai/whl/cu124/torch2.5/

# 安装辅助库
echo -e "\n[4/5] 安装辅助依赖..."
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

# 锁定依赖版本
pip freeze > "$PROJECT_ROOT/configs/sglang_requirements.txt"
echo "依赖清单已写入 configs/sglang_requirements.txt"

# 验证安装
echo -e "\n[5/5] 验证安装..."
python3 << 'PYEOF'
import sys
import torch
import sglang

print(f"Python  : {sys.version.split()[0]}")
print(f"PyTorch : {torch.__version__}")
print(f"CUDA    : {torch.version.cuda}")
print(f"SGLang  : {sglang.__version__}")

if torch.cuda.is_available():
    print(f"GPU     : {torch.cuda.get_device_name(0)}")
else:
    print("⚠ CUDA 不可用")
    sys.exit(1)

# 验证核心模块可导入
from sglang import Engine
print("✓ sglang.Engine 可导入")

# 验证 FlashInfer 可导入
try:
    import flashinfer
    print(f"FlashInfer: {flashinfer.__version__}")
    print("✓ FlashInfer 可导入")
except ImportError as e:
    print(f"⚠ FlashInfer 导入失败: {e}")
    print("  可使用 --disable-flashinfer 回退到 Triton 后端")

print("✓ SGLang 安装验证通过")
PYEOF

deactivate

echo -e "\n========================================"
echo "SGLang 安装完成"
echo "激活环境: source $PROJECT_ROOT/envs/sglang/bin/activate"
echo "========================================"
