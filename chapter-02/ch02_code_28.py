from vllm import LLM, SamplingParams

# 配置1：单节点4卡，纯TP
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,
    pipeline_parallel_size=1,
    dtype="float16",
    max_num_seqs=256
)

# 配置2:8卡，混合并行
llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 每个PP stage内部TP=4
    pipeline_parallel_size=2,  # 2个PP stage
    dtype="float16",
    max_num_seqs=128  # PP会增加延迟，减少批大小
)

# 推理
outputs = llm.generate(
    ["Hello, how are you?"],
    SamplingParams(temperature=0.7, max_tokens=100)
)

# 使用示例
if __name__ == "__main__":
    # 单节点纯TP配置
    print("配置1：单节点4卡纯TP")
    llm1 = LLM(
        model="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=4,
        pipeline_parallel_size=1
    )

    # 混合并行配置
    print("配置2:8卡混合并行")
    llm2 = LLM(
        model="meta-llama/Llama-2-70b-hf",
        tensor_parallel_size=4,
        pipeline_parallel_size=2
    )