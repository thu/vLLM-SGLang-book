#!/bin/bash
# 文件名: scripts/setup/install_cuda.sh
# 功能: 在 Ubuntu 22.04 上安装 CUDA 12.4
# 用法: sudo bash install_cuda.sh
# 注意: 若系统已安装 CUDA 12.4 或更高版本，可跳过此脚本

set -e

echo "========================================"
echo "安装 CUDA 12.4"
echo "========================================"

# 检查是否已安装 CUDA 12.4
if command -v nvcc &> /dev/null; then
    current_cuda=$(nvcc --version | grep "release" \
        | sed 's/.*release //' | sed 's/,.*//')
    echo "当前 CUDA 版本: $current_cuda"
    if [[ "$current_cuda" == 12.4* ]]; then
        echo "✓ CUDA 12.4 已安装，无需重复安装"
        exit 0
    else
        echo "当前版本非 12.4，继续安装..."
        echo "提示: CUDA 支持多版本共存，安装后通过"
        echo "      CUDA_HOME 环境变量切换版本"
    fi
fi

# 添加 NVIDIA 软件包仓库
echo -e "\n[1/4] 添加 NVIDIA 软件包仓库..."
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
rm -f cuda-keyring_1.1-1_all.deb
sudo apt-get update

# 安装 CUDA Toolkit
echo -e "\n[2/4] 安装 CUDA Toolkit 12.4..."
sudo apt-get install -y cuda-toolkit-12-4

# 写入环境变量
echo -e "\n[3/4] 配置环境变量..."
CUDA_ENV_BLOCK='
# CUDA 12.4 环境变量（由 install_cuda.sh 添加）
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
'

# 避免重复写入
if ! grep -q "CUDA 12.4 环境变量" ~/.bashrc; then
    echo "$CUDA_ENV_BLOCK" >> ~/.bashrc
    echo "✓ 环境变量已写入 ~/.bashrc"
else
    echo "✓ 环境变量已存在，跳过写入"
fi

# 使环境变量在当前 shell 生效
export CUDA_HOME=/usr/local/cuda-12.4
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证安装
echo -e "\n[4/4] 验证安装..."
if nvcc --version | grep -q "12.4"; then
    echo "✓ CUDA 12.4 安装成功"
    nvcc --version
else
    echo "✗ CUDA 安装验证失败，请检查安装日志"
    exit 1
fi

echo -e "\n========================================"
echo "CUDA 12.4 安装完成"
echo "请执行 source ~/.bashrc 或重新登录以使环境变量生效"
echo "========================================"
