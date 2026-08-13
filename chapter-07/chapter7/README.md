# Chapter 7 — 结构化数据提取系统（SGLang 优势场景）

本仓库为《LLM 推理系统实战》第7章配套代码。

## 版本基线

| 组件 | 版本 |
|------|------|
| SGLang | v0.4.x |
| Pydantic | v2.5+ |
| Python | 3.10+ |
| 模型 | Llama-3.1-8B-Instruct |
| CUDA | 12.1 / 12.4 |

---

## 目录结构

```
chapter7/
├── src/
│   ├── models/           # Pydantic 数据模型（Resume、Contract、MedicalReport、FinancialNews）
│   ├── parsers/          # 文档解析层（PDF、DOCX、TXT）
│   ├── extractors/       # SGLang 约束解码提取器 + 批处理器
│   ├── prompts/          # Few-shot 示例管理
│   └── evaluators/       # 准确率评估框架
├── scripts/
│   ├── verify_env.py                  # 环境验证
│   ├── verify_schema.py               # JSON Schema 验证
│   ├── verify_constrained_decoding.py # 约束解码功能验证
│   ├── verify_radix_attention.py      # RadixAttention 效果验证
│   ├── benchmark_throughput.py        # 吞吐量基准测试
│   ├── monitor_cache.py               # 缓存命中率实时监测
│   ├── evaluate_accuracy.py           # 准确率批量评估
│   └── run_batch.py                   # 批量处理入口
├── data/
│   ├── raw/              # 原始文档（PDF/DOCX）
│   ├── extracted/        # 解析后的纯文本
│   └── ground_truth/     # 人工标注 JSON
├── results/              # 实验结果输出
├── requirements.txt
└── README.md
```

---

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境
python3 -m venv ~/llm-envs/chapter7
source ~/llm-envs/chapter7/bin/activate

# 安装 PyTorch（CUDA 12.1）
pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu121

# 安装其他依赖
pip install -r requirements.txt
```

### 2. 下载模型

```bash
huggingface-cli login    # 需要先在 HuggingFace 接受 Llama-3.1 许可协议
huggingface-cli download meta-llama/Llama-3.1-8B-Instruct \
    --local-dir ~/models/Llama-3.1-8B-Instruct \
    --local-dir-use-symlinks False
```

### 3. 启动 SGLang 服务

```bash
python -m sglang.launch_server \
    --model-path ~/models/Llama-3.1-8B-Instruct \
    --host 127.0.0.1 \
    --port 30000 \
    --mem-fraction-static 0.85 \
    --tp 1 \
    --enable-metrics
```

### 4. 验证环境

```bash
python scripts/verify_env.py
python scripts/verify_schema.py
python scripts/verify_constrained_decoding.py
```

### 5. 批量处理

```bash
# 处理 data/raw/ 目录下所有文档
python scripts/run_batch.py \
    --input data/raw/ \
    --output results/extracted/ \
    --concurrency 20

# 使用 Few-shot 示例（推荐）
python scripts/run_batch.py \
    --input data/raw/ \
    --output results/extracted/ \
    --few-shot date_normalization current_job \
    --concurrency 20
```

### 6. 评估准确率（需要 ground truth 标注）

```bash
python scripts/evaluate_accuracy.py \
    --predictions results/extracted/ \
    --ground-truth data/ground_truth/ \
    --output results/accuracy_report.json
```

### 7. 性能测试

```bash
# 吞吐量基准测试（遍历多个并发数）
python scripts/benchmark_throughput.py --n 200

# RadixAttention 效果验证
python scripts/verify_radix_attention.py

# 缓存命中率实时监测（在批量任务运行时同步执行）
python scripts/monitor_cache.py --duration 300
```

---

## 场景延伸

本章的提取器和批处理器支持任意 Pydantic v2 模型，
切换场景只需替换模型和系统提示：

```python
import asyncio
import aiohttp
from src.extractors.sglang_extractor import extract_structured
from src.models.contract import Contract

CONTRACT_PROMPT = """你是一个合同审查助手。
从合同文本中提取关键信息并识别风险条款。
风险判断标准：
- 单方面免责条款 → 高风险
- 赔偿上限低于合同金额10% → 中等风险
- 缺少争议解决条款 → 高风险"""

async def extract_contract(text: str):
    async with aiohttp.ClientSession() as session:
        contract = await extract_structured(
            text=text,
            session=session,
            pydantic_model=Contract,
            system_prompt=CONTRACT_PROMPT,
            schema_name="contract_extraction",
        )
    return contract
```

其他场景（医疗报告、金融舆情）的用法完全相同，
只需替换 `Contract` 为 `MedicalReport` 或 `FinancialNews`。

---

## 常见问题

**Q：SGLang 服务启动后健康检查返回 404？**
A：服务可能仍在加载模型（约30-60秒），继续等待后重试。

**Q：约束解码报错 `response_format not supported`？**
A：确认 SGLang 版本 `~=0.4.0`，旧版本不支持 `json_schema` 类型。

**Q：`/flush_cache` 返回 404？**
A：执行 `curl http://127.0.0.1:30000/` 查看实际可用端点列表。

**Q：指标采样全部返回 -1？**
A：执行 `curl http://127.0.0.1:30000/metrics` 查看实际指标名称，
   对照修改 `monitor_cache.py` 中的 `WATCHED_METRICS`。

**Q：ValidationError 手机号验证失败？**
A：通常是简历原始数据不规范（实际位数不足），
   系统会保存为 `*_validation_error.json` 供人工审核，不影响其他简历。
