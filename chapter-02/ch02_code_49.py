import torch


# 问题1：量化后精度下降严重
# 解决：检查校准数据是否representative
def validate_calibration_data(model, tokenizer, calib_data):
    """
    验证校准数据质量
    功能：检查校准数据是否具有代表性
    """
    ppls = []
    for text in calib_data[:10]:
        inputs = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs.input_ids)
            ppl = torch.exp(outputs.loss)
            ppls.append(ppl.item())

    avg_ppl = sum(ppls) / len(ppls)
    print(f"校准数据平均perplexity:{avg_ppl:.2f}")

    # 如果PPL异常高，>50，说明校准数据不合适
    if avg_ppl > 50:
        print("警告：校准数据质量差，建议更换！")


# 问题2：量化后推理速度没提升
# 解决：检查是否使用了优化的kernel
def check_kernel_optimization():
    """
    检查kernel优化状态
    功能：验证硬件和软件环境是否支持量化加速
    """
    import vllm

    print(f"vLLM版本：{vllm.__version__}")
    print(f"CUDA可用：{torch.cuda.is_available()}")
    print(f"Compute Capability:{torch.cuda.get_device_capability()}")

    # Ampere架构，8.0+，才能充分利用INT4加速
    if torch.cuda.get_device_capability()[0] < 8:
        print("警告：GPU架构较老，INT4加速有限！")


# 问题3：显存占用仍然很高
# 解决：检查KV Cache和激活值占用
def analyze_memory_usage(llm):
    """
    分析显存占用
    功能：定位内存使用的瓶颈
    """
    torch.cuda.reset_peak_memory_stats()

    # 推理一个batch
    llm.generate(["Test prompt"], SamplingParams(max_tokens=100))

    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    model_memory = llm.get_model_memory() / 1e9
    kv_cache_memory = llm.get_kv_cache_memory() / 1e9

    print(f"总显存峰值：{peak_memory:.2f} GB")
    print(f"模型权重：{model_memory:.2f} GB")
    print(f"KV Cache:{kv_cache_memory:.2f} GB")
    print(f"其他，激活等：{peak_memory - model_memory - kv_cache_memory:.2f} GB")


# 使用示例
if __name__ == "__main__":
    print("=== 问题排查工具 ===")

    # 检查kernel优化
    check_kernel_optimization()

    # 如果有模型和数据，可以运行其他检查
    # validate_calibration_data(model, tokenizer, calib_data)
    # analyze_memory_usage(llm)