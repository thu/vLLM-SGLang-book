#!/bin/bash
# 文件名: scripts/setup/check_system.sh
# 功能: 检查系统环境是否满足测试要求
# 用法: bash check_system.sh

set -e

echo "========================================"
echo "系统环境检查"
echo "========================================"

# 检查操作系统版本
echo -e "\n[1/7] 检查操作系统..."
os_version=$(lsb_release -d | cut -f2)
echo "操作系统: $os_version"
if [[ "$os_version" == *"Ubuntu 22.04"* ]]; then
    echo "✓ 操作系统符合要求"
else
    echo "⚠ 推荐使用 Ubuntu 22.04 LTS"
fi

# 检查 NVIDIA GPU 及驱动
echo -e "\n[2/7] 检查 GPU..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ 未检测到 NVIDIA 驱动，请先安装驱动"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version \
    --format=csv,noheader
echo "✓ GPU 检测正常"

# 检查驱动版本（CUDA 12.4 要求驱动 >= 550）
driver_version=$(nvidia-smi --query-gpu=driver_version \
    --format=csv,noheader | head -1)
major=$(echo "$driver_version" | cut -d'.' -f1)
if [ "$major" -ge 550 ]; then
    echo "✓ 驱动版本 $driver_version 满足 CUDA 12.4 要求"
else
    echo "⚠ CUDA 12.4 需要驱动版本 >= 550，当前为 $driver_version"
fi

# 检查 CPU 核心数
echo -e "\n[3/7] 检查 CPU..."
cpu_cores=$(nproc)
echo "CPU 核心数: $cpu_cores"
[ "$cpu_cores" -ge 16 ] \
    && echo "✓ CPU 核心数充足" \
    || echo "⚠ 推荐至少 16 核 CPU"

# 检查内存
echo -e "\n[4/7] 检查内存..."
total_mem=$(free -g | awk '/^Mem:/{print $2}')
echo "总内存: ${total_mem} GB"
[ "$total_mem" -ge 128 ] \
    && echo "✓ 内存充足" \
    || echo "⚠ 推荐至少 128 GB 内存"

# 检查磁盘空间
echo -e "\n[5/7] 检查磁盘空间..."
available_disk=$(df -BG "$HOME" \
    | awk 'NR==2 {gsub("G","",$4); print $4}')
echo "可用空间: ${available_disk} GB"
[ "$available_disk" -ge 200 ] \
    && echo "✓ 磁盘空间充足" \
    || echo "⚠ 推荐至少 200 GB 可用空间"

# 检查 Python 版本
echo -e "\n[6/7] 检查 Python..."
if ! command -v python3 &> /dev/null; then
    echo "✗ 未安装 Python3"
    exit 1
fi
python_version=$(python3 --version | cut -d' ' -f2)
echo "Python 版本: $python_version"
if [[ "$python_version" == 3.10.* ]] \
    || [[ "$python_version" == 3.11.* ]] \
    || [[ "$python_version" == 3.12.* ]]; then
    echo "✓ Python 版本符合要求"
else
    echo "⚠ 推荐使用 Python 3.10–3.12"
fi

# 检查 pip
echo -e "\n[7/7] 检查 pip..."
if ! command -v pip3 &> /dev/null; then
    echo "✗ 未安装 pip3"
    exit 1
fi
echo "pip 版本: $(pip3 --version | cut -d' ' -f2)"
echo "✓ pip 已安装"

echo -e "\n========================================"
echo "系统环境检查完成"
echo "========================================"
