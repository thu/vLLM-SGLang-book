# config/model_config.py
# 模型架构参数与硬件部署配置
# 版本基线：SGLang v0.5.x / vLLM v0.8.x
# 模型：Qwen2-72B-Instruct-AWQ（HuggingFace 官方量化版本）

# ── Qwen2-72B 架构参数 ──────────────────────────────────────────
# 来源：Qwen2 技术报告（arXiv:2407.10671）
MODEL_CONFIG = {
    # Transformer 层数
    "num_layers": 80,
    # GQA KV 头数（Grouped Query Attention，Query 头数=64）
    "num_kv_heads": 8,
    # 注意力头维度
    "head_dim": 128,
    # KV Cache 精度：FP16（量化仅作用于权重，不作用于 KV Cache）
    "kv_dtype_bytes": 2,
    # 模型最大上下文长度
    "max_context_len": 32768,
    # 模型 HuggingFace Hub 路径
    "model_path": "Qwen/Qwen2-72B-Instruct-AWQ",
    # 量化类型
    "quantization": "awq",
    # AWQ 量化 group size（官方版本固定值）
    "awq_group_size": 128,
}

# ── 硬件与部署配置 ───────────────────────────────────────────────
DEPLOY_CONFIG = {
    # 张量并行卡数
    "tensor_parallel_size": 4,
    # 单卡显存（GB）：A100 SXM4 80GB
    "total_vram_per_card_gb": 80,
    # AWQ-4bit + TP=4 下单卡权重显存占用（GB）
    # 计算：72B params × 4bit / 8 / 4卡 ≈ 9 GB
    "weight_per_card_gb": 9.0,
    # 峰值激活值缓冲区预留（GB/卡）
    # 在 max_num_seqs=32、长序列场景下的保守估计
    "activation_buffer_per_card_gb": 3.0,
    # SGLang：静态 KV Cache 预分配比例
    # 0.88 × (80 - 9 - 3) GB = 0.88 × 68 ≈ 59.8 GB/卡
    "sglang_mem_fraction_static": 0.88,
    # vLLM：GPU 显存总利用率（权重 + KV Cache 之和）
    # 0.90 × 80 GB = 72 GB/卡，减去权重 9 GB，剩余 63 GB 用于 KV Cache
    "vllm_gpu_memory_utilization": 0.90,
    # vLLM 块大小（tokens/块）：32 是长文本场景的推荐值
    # 须为 2 的幂次；块大小越大，哈希表维护开销越低
    "vllm_block_size": 32,
    # 最大并发序列数
    "max_num_seqs": 32,
    # SGLang 服务地址
    "sglang_host": "0.0.0.0",
    "sglang_port": 30000,
    # vLLM 服务地址
    "vllm_host": "0.0.0.0",
    "vllm_port": 30001,
}

# ── KV Cache 单 token 占用计算 ───────────────────────────────────
# 公式：num_layers × num_kv_heads × head_dim × 2（K+V）× dtype_bytes
# 结果：80 × 8 × 128 × 2 × 2 = 327,680 bytes ≈ 320 KB/token
#
# 注意：TP=4 下 KV Cache 的存储方式取决于框架实现：
#   SGLang：每卡存储 num_kv_heads/tp_size = 2 个头的 KV，
#           4 卡合计覆盖全部 8 个 KV 头，逻辑上等价于单一缓存单元。
#   vLLM：与 SGLang 相同，KV Cache 按头维度切分到各卡。
#   因此"单 token KV Cache 占用"指 4 卡合计的逻辑占用，
#   实际每卡存储量为此值的 1/4。
KV_BYTES_PER_TOKEN = (
    MODEL_CONFIG["num_layers"]
    * MODEL_CONFIG["num_kv_heads"]
    * MODEL_CONFIG["head_dim"]
    * 2  # K + V
    * MODEL_CONFIG["kv_dtype_bytes"]
)  # = 327,680 bytes
