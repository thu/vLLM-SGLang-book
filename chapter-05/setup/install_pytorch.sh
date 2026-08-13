#!/bin/bash
# 文件名: scripts/setup/install_pytorch.sh
# 功能: 在临时虚拟环境中安装 PyTorch 2.5 并验证 GPU 计算功能
# 用法: bash install_pytorch.sh
# 说明: 此脚本用于在安装 vLLM/SGLang 之前验证 CUDA 环境可用性。
#       vLLM 和 SGLang 各自的虚拟环境会独立安装各自所需的 PyTorch 版本。

set -e

PROJECT_ROOT=~/llm-benchmark

echo "========================================"
echo "安装 PyTorch 2.5（CUDA 12.4 验证环境）"
echo "========================================"

# 创建临时验证环境
echo -e "\n[1/3] 创建验证用虚拟环境..."
python3 -m venv "$PROJECT_ROOT/envs/base"
source "$PROJECT_ROOT/envs/base/bin/activate"
pip install --upgrade pip setuptools wheel

# 安装 PyTorch
echo -e "\n[2/3] 安装 PyTorch 2.5（CUDA 12.4 编译版本）..."
pip install torch==2.5.1 --index-url \
    https://download.pytorch.org/whl/cu124

# 验证安装
echo -e "\n[3/3] 执行 GPU 计算验证..."
python3 << 'PYEOF'
import sys
import torch

print(f"Python    : {sys.version.split()[0]}")
print(f"PyTorch   : {torch.__version__}")
print(f"CUDA 可用 : {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("✗ CUDA 不可用，请检查驱动与 CUDA 安装")
    sys.exit(1)

print(f"CUDA 版本 : {torch.version.cuda}")
print(f"cuDNN 版本: {torch.backends.cudnn.version()}")
print(f"GPU 数量  : {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    name = torch.cuda.get_device_name(i)
    props = torch.cuda.get_device_properties(i)
    mem_gb = props.total_memory / 1024**3
    print(f"GPU {i}    : {name}，显存 {mem_gb:.1f} GB")

# 验证 GPU 矩阵运算
x = torch.rand(512, 512, device="cuda")
y = torch.rand(512, 512, device="cuda")
z = torch.matmul(x, y)
assert z.shape == (512, 512), "矩阵乘法结果形状不正确"
print("✓ GPU 矩阵运算测试通过")

# 验证半精度计算（FP16 是测试中使用的精度）
x_fp16 = x.half()
y_fp16 = y.half()
z_fp16 = torch.matmul(x_fp16, y_fp16)
assert z_fp16.dtype == torch.float16, "FP16 计算结果类型不正确"
print("✓ FP16 半精度计算测试通过")

print("\n✓ 全部验证通过，CUDA 环境就绪")
PYEOF

deactivate

echo -e "\n========================================"
echo "PyTorch 验证完成"
echo "后续 vLLM 和 SGLang 将在各自环境中独立安装 PyTorch"
echo "========================================"
