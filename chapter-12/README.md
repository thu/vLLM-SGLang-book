# 第12章 代码仓库：基于 vLLM 的 DeepSeek V4 长上下文推理服务实战

## 目录结构

```
chapter-12/
├── server/
│   └── launch_v4_flash.py        # vLLM 服务启动脚本（含配置摘要打印）
├── client/
│   └── doc_qa_server.py          # 长文档问答 API 服务（aiohttp）
├── benchmark/
│   ├── latency_benchmark.py      # TTFT/TPOT/吞吐量基准测试
│   └── chunk_size_sweep.py       # chunked prefill 参数扫描
└── utils/
    └── kv_cache_estimator.py     # KV cache 显存估算工具
```

## 版本基线

- vLLM：v0.8.x
- DeepSeek-V4-Flash-AWQ（AWQ Marlin 量化）
- Python：3.11+
- 硬件：4×NVIDIA A100 80G，NVLink 互联

## 快速开始

### 1. 启动 vLLM 服务（128K 上下文）

```bash
python server/launch_v4_flash.py \
    --max-model-len 131072 \
    --chunk-size 4096 \
    --gpu-mem-util 0.90
```

### 2. 估算 KV cache 显存

```bash
python utils/kv_cache_estimator.py \
    --model v4-flash \
    --seq-len 131072 \
    --tp 4
```

### 3. 启动文档问答 API

```bash
python client/doc_qa_server.py \
    --port 9000 \
    --max-concurrency 16
```

### 4. 运行性能基准测试

```bash
python benchmark/latency_benchmark.py \
    --context-lengths 4096 16384 32768 65536 131072 \
    --concurrency 1 4 8 16 \
    --repeat 5
```

## 注意事项

- 运行基准测试前请确保 vLLM 服务已完成预热（发送 2-3 条预热请求）
- `chunk_size_sweep.py` 每个 chunk_size 测试需手动重启服务
- 长上下文（>64K）测试时建议降低并发数（≤8），避免 OOM
