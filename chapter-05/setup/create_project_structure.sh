#!/bin/bash
# 文件名: scripts/setup/create_project_structure.sh
# 功能: 建立项目目录骨架
# 用法: bash create_project_structure.sh

PROJECT_ROOT=~/llm-benchmark

echo "========================================"
echo "创建项目目录结构"
echo "========================================"

# 虚拟环境目录
mkdir -p "$PROJECT_ROOT"/envs

# 模型权重目录
mkdir -p "$PROJECT_ROOT"/models

# 数据集目录：原始数据与预处理后数据分开存放
mkdir -p "$PROJECT_ROOT"/datasets/raw
mkdir -p "$PROJECT_ROOT"/datasets/processed

# 测试脚本目录：公共模块 + 框架特定脚本
mkdir -p "$PROJECT_ROOT"/scripts/common
mkdir -p "$PROJECT_ROOT"/scripts/vllm
mkdir -p "$PROJECT_ROOT"/scripts/sglang
mkdir -p "$PROJECT_ROOT"/scripts/setup

# 结果目录：各框架结果 + 对比结果 + 可视化图表
mkdir -p "$PROJECT_ROOT"/results/vllm
mkdir -p "$PROJECT_ROOT"/results/sglang
mkdir -p "$PROJECT_ROOT"/results/comparison
mkdir -p "$PROJECT_ROOT"/results/figures

# 日志与配置
mkdir -p "$PROJECT_ROOT"/logs
mkdir -p "$PROJECT_ROOT"/configs

echo "项目目录已创建："
find "$PROJECT_ROOT" -maxdepth 2 -type d | sort

echo -e "\n✓ 目录结构创建完成"
echo "项目根目录: $PROJECT_ROOT"
