from datasets import load_dataset
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def complete_quantization_workflow():
    """
    完整的量化工作流程
    功能：从数据准备到模型部署的端到端流程
    """

    # ========== 步骤1：准备校准数据 ==========
    print("=== 步骤1：准备校准数据 ===")

    def prepare_calibration_data(dataset_name="c4", num_samples=128):
        """准备校准数据集"""
        dataset = load_dataset(dataset_name, split="train", streaming=True)

        calibration_samples = []
        for i, sample in enumerate(dataset):
            if i >= num_samples:
                break
            calibration_samples.append(sample["text"])

        return calibration_samples

    calibration_data = prepare_calibration_data()
    print(f"校准数据准备完成，样本数：{len(calibration_data)}")

    # ========== 步骤2：量化模型 ==========
    print("\n=== 步骤2：量化模型 ===")

    model_name = "meta-llama/Llama-2-70b-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 配置量化参数
    quantize_config = BaseQuantizeConfig(
        bits=4,  # 4-bit量化
        group_size=128,  # 分组大小
        desc_act=True,  # 降序激活
        sym=False,  # 非对称量化
        damp_percent=0.01,  # Hessian阻尼
        true_sequential=True,  # 顺序量化，更慢但更准
    )

    # 加载模型
    print(f"加载模型：{model_name}...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config,
        device_map="auto"  # 自动分配GPU
    )

    # 执行量化
    print("执行量化...")
    model.quantize(
        calibration_data,
        batch_size=1,  # 70B模型建议batch_size=1
    )

    # ========== 步骤3：保存量化模型 ==========
    print("\n=== 步骤3：保存量化模型 ===")

    output_dir = "Llama-2-70b-GPTQ-4bit"
    model.save_quantized(output_dir, use_safetensors=True)
    tokenizer.save_pretrained(output_dir)

    print(f"量化模型已保存到：{output_dir}")

    # ========== 步骤4：验证量化质量 ==========
    print("\n=== 步骤4：验证量化质量 ===")

    from lm_eval import evaluator

    # 在标准benchmark上评估
    results = evaluator.simple_evaluate(
        model="hf-causal",
        model_args=f"pretrained={output_dir}",
        tasks=["hellaswag", "winogrande", "arc_easy"],
        batch_size=8
    )

    print("评估结果：")
    for task, metrics in results["results"].items():
        print(f"{task}:{metrics['acc']:.4f}")

    # ========== 步骤5：在vLLM中部署 ==========
    print("\n=== 步骤5：在vLLM中部署 ===")

    llm = LLM(
        model=output_dir,
        quantization="gptq",
        dtype="float16",
        tensor_parallel_size=1,  # 量化后单卡即可
        max_model_len=4096,
        gpu_memory_utilization=0.9
    )

    # 测试推理
    prompts = [
        "Explain quantum computing in simple terms:",
        "Write a Python function to sort a list:"
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=256
    )

    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print(f"\nPrompt:{prompt}")
        print(f"Generated:{output.outputs[0].text[:200]}...")
        print("-" * 80)


# 使用示例
if __name__ == "__main__":
    complete_quantization_workflow()