from vllm import LLM, SamplingParams

# 使用预量化的AWQ模型
llm = LLM(
    model="TheBloke/Llama-2-70B-AWQ",
    quantization="awq",
    dtype="float16",
    max_model_len=4096
)

# 推理
prompts = ["Explain the concept of machine learning."]
outputs = llm.generate(prompts, SamplingParams(max_tokens=200))

for output in outputs:
    print(f"生成：{output.outputs[0].text}")


# 或者自己量化，使用AutoAWQ库
def quantize_with_autoawq(model_name: str, output_dir: str):
    """
    使用AutoAWQ量化模型
    功能：将FP16模型量化为AWQ格式
    """
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    # 量化
    print(f"加载模型：{model_name}...")
    model = AutoAWQForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # AWQ量化，自动搜索最优缩放因子
    print("执行AWQ量化...")
    model.quantize(
        tokenizer,
        quant_config={
            "zero_point": True,
            "q_group_size": 128,
            "w_bit": 4,
            "version": "GEMM"  # 使用优化的GEMM kernel
        }
    )

    # 保存
    print(f"保存到：{output_dir}")
    model.save_quantized(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("量化完成！")


# 使用示例
if __name__ == "__main__":
    # 方法1：使用预量化模型
    print("=== 使用预量化AWQ模型 ===")
    llm = LLM(model="TheBloke/Llama-2-70B-AWQ", quantization="awq")

    # 方法2：自己量化
    print("\n=== 自己量化模型 ===")
    quantize_with_autoawq(
        model_name="meta-llama/Llama-2-70b-hf",
        output_dir="./Llama-2-70b-AWQ"
    )

    # 加载自己量化的模型
    llm2 = LLM(model="./Llama-2-70b-AWQ", quantization="awq")