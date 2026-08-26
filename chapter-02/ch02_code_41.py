from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from transformers import AutoTokenizer


def quantize_model_with_autogptq(
        model_name: str,
        output_dir: str,
        bits: int = 4,
        group_size: int = 128
):
    """
    使用AutoGPTQ量化模型
    功能：将FP16模型量化为GPTQ格式
    """
    # 量化配置
    quantize_config = BaseQuantizeConfig(
        bits=bits,  # 4-bit量化
        group_size=group_size,  # 分组大小
        desc_act=True,  # 激活值降序排序
        sym=False,  # 非对称量化
        damp_percent=0.01  # Hessian阻尼
    )

    # 加载并量化模型
    print(f"加载模型：{model_name}...")
    model = AutoGPTQForCausalLM.from_pretrained(
        model_name,
        quantize_config=quantize_config
    )

    # 准备校准数据，c4数据集，128-512个样本
    print("准备校准数据...")
    calibration_data = load_calibration_dataset("c4", num_samples=128)

    # 执行量化
    print("执行量化...")
    model.quantize(calibration_data)

    # 保存量化模型
    print(f"保存量化模型到：{output_dir}")
    model.save_quantized(output_dir)

    # 保存tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)

    print("量化完成！")


def load_calibration_dataset(dataset_name: str, num_samples: int = 128):
    """
    加载校准数据集
    功能：准备用于量化的代表性样本
    """
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split="train", streaming=True)

    calibration_samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        calibration_samples.append(sample["text"])

    return calibration_samples


# 使用示例
if __name__ == "__main__":
    # 量化LLaMA-70B模型
    quantize_model_with_autogptq(
        model_name="meta-llama/Llama-2-70b-hf",
        output_dir="./Llama-2-70b-GPTQ-4bit",
        bits=4,
        group_size=128
    )

    # 在vLLM中加载量化模型
    from vllm import LLM

    print("\n在vLLM中加载量化模型...")
    llm = LLM(
        model="./Llama-2-70b-GPTQ-4bit",
        quantization="gptq"
    )

    print("量化模型加载成功！")