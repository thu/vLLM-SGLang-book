from vllm import LLM, SamplingParams
from typing import Dict


class MultiModelServer:
    """
    多模型服务器
    功能：在单进程中管理多个模型实例
    """

    def __init__(self):
        self.models: Dict[str, LLM] = {}

    def load_model(self, model_name: str, model_path: str, **kwargs):
        """
        加载模型
        功能：初始化并缓存模型实例
        """
        print(f"加载模型：{model_name}...")
        self.models[model_name] = LLM(
            model=model_path,
            **kwargs
        )
        print(f"模型{model_name}加载完成")

    def generate(self, model_name: str, prompts: list, **sampling_kwargs):
        """
        生成文本
        功能：使用指定模型进行推理
        """
        if model_name not in self.models:
            raise ValueError(f"模型{model_name}未加载")

        model = self.models[model_name]
        sampling_params = SamplingParams(**sampling_kwargs)
        return model.generate(prompts, sampling_params)


# 使用示例
if __name__ == "__main__":
    # 创建多模型服务器
    server = MultiModelServer()

    # 加载多个模型
    server.load_model(
        "llama-7b",
        "meta-llama/Llama-2-7b-hf",
        tensor_parallel_size=1
    )

    server.load_model(
        "codellama-7b",
        "codellama/CodeLlama-7b-hf",
        tensor_parallel_size=1
    )

    # 使用不同模型生成
    outputs1 = server.generate(
        "llama-7b",
        ["Explain quantum computing."],
        max_tokens=100
    )
    print(f"LLaMA输出：{outputs1[0].outputs[0].text}")

    outputs2 = server.generate(
        "codellama-7b",
        ["def factorial(n):"],
        max_tokens=100
    )
    print(f"CodeLLaMA输出：{outputs2[0].outputs[0].text}")