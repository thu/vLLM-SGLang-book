# 第11章配套代码：法律文书批量分析（前缀缓存深度应用）

## 环境要求

- Python 3.11+
- SGLang v0.5.x（通过 `pip install sglang[all]` 安装）
- vLLM v0.8.x（通过 `pip install vllm` 安装）
- CUDA 12.1 / cuDNN 8.9
- 4× NVIDIA A100 80GB（张量并行）

## 模型

Qwen/Qwen2-72B-Instruct-AWQ（HuggingFace Hub 官方量化版本）

## 目录结构

```
ch11_legal_doc_analysis/
├── config/
│   ├── __init__.py
│   ├── model_config.py        # 模型与硬件参数
│   └── prompt_config.py       # 提示词模板（字节级固定）
├── data/
│   ├── __init__.py
│   ├── dataset_builder.py     # 合成数据集构建（ch11_code_01）
│   └── text_normalizer.py     # 文本标准化流水线（ch11_code_07）
├── inference/
│   ├── __init__.py
│   ├── memory_budget.py       # KV Cache容量计算（ch11_code_02）
│   ├── sglang_server.py       # SGLang服务启动与管理（ch11_code_04）
│   ├── vllm_server.py         # vLLM服务启动与管理（ch11_code_05）
│   └── base_client.py         # 统一推理客户端接口
├── pipeline/
│   ├── __init__.py
│   ├── scheduler.py           # 两阶段流水线调度（ch11_code_10）
│   ├── cache_scheduler.py     # 自适应缓存调度（ch11_code_08）
│   └── resilient_client.py    # 降级策略客户端（ch11_code_09）
├── monitoring/
│   ├── __init__.py
│   ├── metrics.py             # Prometheus指标定义（ch11_code_06）
│   └── grafana_panels.yaml    # Grafana面板查询配置
├── utils/
│   ├── __init__.py
│   └── prefix_validator.py    # 前缀一致性验证（ch11_code_01片段）
├── scripts/
│   ├── run_baseline.py        # 基线测试入口（ch11_code_03）
│   ├── run_benchmark.py       # 性能对比测试入口（ch11_code_11）
│   └── run_extended.py        # 延伸场景演示
├── tests/
│   ├── test_normalizer.py
│   ├── test_memory_budget.py
│   └── test_prefix_validator.py
├── requirements.txt
└── README.md
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 SGLang 服务
python -m inference.sglang_server

# 3. 运行基线测试（无缓存）
python scripts/run_baseline.py --framework sglang --no-cache

# 4. 运行完整两阶段流水线（含缓存）
python scripts/run_benchmark.py --framework sglang --docs 180
```

## 版本基线

| 组件 | 版本 |
|------|------|
| SGLang | v0.5.x |
| vLLM | v0.8.x |
| Pydantic | v2.5+ |
| PyTorch | 2.3.0 |
| Transformers | 4.40+ |
