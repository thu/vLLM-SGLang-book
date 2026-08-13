# 第6章　高并发 API 服务：vLLM 生产部署实战

## 版本信息

| 组件 | 版本 |
|------|------|
| vLLM | 0.6.3 |
| FastAPI | 0.115.x |
| Python | 3.11+ |
| 模型 | Qwen/Qwen2.5-7B-Instruct-AWQ |
| CUDA | 12.1+（A100 验证） |

---

## 目录结构

```
ch06/
├── scripts/
│   ├── download_model.sh          # 下载模型权重
│   ├── start_vllm.sh              # 启动 vLLM 服务
│   ├── validate_service.py        # 服务健康验证（3项检查）
│   ├── tune_max_num_seqs.py       # max_num_seqs 参数扫描
│   └── compare_configurations.py  # 基线 vs 优化配置对比
├── gateway/
│   ├── main.py                    # FastAPI 网关主文件
│   └── truncation.py              # 输入截断模块
├── load_test/
│   └── locustfile.py              # Locust 压测脚本
├── monitoring/
│   ├── prometheus.yml             # Prometheus 采集配置
│   ├── alert_rules.yml            # 告警规则（5条核心规则）
│   └── docker-compose.yml         # 监控栈一键启动
├── data/
│   └── customer_service_queries.jsonl  # 压测数据集（30条样本）
├── extensions/
│   └── routing/
│       └── canary.py              # 灰度路由模块
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r ch06/requirements.txt
```

> vLLM 的安装需要匹配 CUDA 版本，如遇问题请参考：
> https://docs.vllm.ai/en/v0.6.3/getting_started/installation.html

### 2. 下载模型

```bash
bash ch06/scripts/download_model.sh
```

默认下载到 `~/models/Qwen2.5-7B-Instruct-AWQ`。
通过环境变量覆盖路径：`CH06_MODEL_DIR=/your/path bash download_model.sh`

### 3. 启动 vLLM 服务

```bash
# 前台运行（开发调试）
bash ch06/scripts/start_vllm.sh

# 后台运行（生产）
bash ch06/scripts/start_vllm.sh --daemon
```

### 4. 验证服务

```bash
python ch06/scripts/validate_service.py
```

全部通过输出：
```
=== 全部检查通过，服务就绪 ===
```

### 5. 启动网关

```bash
uvicorn ch06.gateway.main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 6. 启动监控栈

```bash
docker compose -f ch06/monitoring/docker-compose.yml up -d
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
```

### 7. 运行基线压测

```bash
locust -f ch06/load_test/locustfile.py \
       --host http://localhost:8001 \
       --headless -u 500 -r 50 -t 5m \
       --csv results/baseline
```

---

## 6.4 节参数调优流程

### 步骤一：扫描 max_num_seqs

```bash
python ch06/scripts/tune_max_num_seqs.py
```

扫描完成后查看 `results/tune_max_num_seqs.json`，
选择 QPS 接近峰值且 P99 ≤ 1000ms 的最小 `max_num_seqs` 值。

### 步骤二：将最优值填入对比脚本

编辑 `ch06/scripts/compare_configurations.py`，
将 `OPTIMIZED_CONFIG` 中的 `max_num_seqs` 和 `gpu_memory_utilization`
替换为实测最优值。

### 步骤三：运行完整对比

```bash
python ch06/scripts/compare_configurations.py
```

对比报告保存到 `results/comparison_report.json`。

---

## 常见问题

**Q：vLLM 启动时报 OOM**
A：检查 `--gpu-memory-utilization` 是否过高，或其他进程占用了 GPU 显存（`nvidia-smi` 确认）。

**Q：validate_service.py 提示 max_model_len 低于预期**
A：vLLM 在显存不足时会静默降低此值。尝试降低 `--gpu-memory-utilization` 或减小 `--max-model-len`。

**Q：Locust 压测时大量 504 超时**
A：首先检查 Grafana 中的 `vllm:num_requests_waiting`，
   如果等待队列持续增长，说明已超过系统吞吐上限；
   如果队列为空但超时率高，检查网关连接池配置（`max_connections`）。

**Q：如何将压测数据集替换为真实客服数据**
A：将数据整理为 `{"question": "..."}` 格式，每行一条，
   保存到 `ch06/data/customer_service_queries.jsonl` 覆盖默认样本文件。
