from vllm import LLM, SamplingParams

# 方法1：使用预量化的GPTQ模型
llm = LLM(
    model="TheBloke/Llama-2-70B-GPTQ",  # HuggingFace上的GPTQ模型
    quantization="gptq",  # 指定量化方法
    dtype="float16",  # 激活值仍用FP16
    gpu_memory_utilization=0.9,
    max_model_len=4096
)

# 正常推理
prompts = ["Explain the theory of relativity in simple terms."]
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"生成文本：{output.outputs[0].text}")

# 使用示例主函数
if __name__ == "__main__":
    print("加载GPTQ量化模型...")
    llm = LLM(
        model="TheBloke/Llama-2-70B-GPTQ",
        quantization="gptq",
        dtype="float16"
    )

    print("执行推理...")
    prompts = [
        "What is machine learning?",
        "Explain neural networks.",
        "What are transformers in AI?"
    ]

    outputs = llm.generate(prompts, SamplingParams(max_tokens=100))

    for i, output in enumerate(outputs):
        print(f"\n--- 请求 {i + 1} ---")
        print(f"提示：{output.prompt}")
        print(f"生成：{output.outputs[0].text[:200]}...")