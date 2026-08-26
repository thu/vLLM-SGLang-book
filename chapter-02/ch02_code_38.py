import gc
import torch


class DynamicModelManager:
    """
    动态模型管理器
    功能：支持运行时加载、卸载和切换模型
    """

    def __init__(self, max_models: int = 3):
        self.max_models = max_models
        self.models: Dict[str, LLM] = {}
        self.model_usage: Dict[str, int] = {}  # 使用次数统计

    def load_model(self, model_name: str, model_path: str, **kwargs):
        """
        加载模型，如果达到上限则卸载最少使用的模型
        功能：动态管理模型缓存
        """
        # 如果模型已加载，直接返回
        if model_name in self.models:
            self.model_usage[model_name] += 1
            return self.models[model_name]

        # 如果达到上限，卸载最少使用的模型
        if len(self.models) >= self.max_models:
            least_used = min(self.model_usage, key=self.model_usage.get)
            self.unload_model(least_used)

        # 加载新模型
        print(f"加载模型：{model_name}...")
        self.models[model_name] = LLM(model=model_path, **kwargs)
        self.model_usage[model_name] = 1

        return self.models[model_name]

    def unload_model(self, model_name: str):
        """
        卸载模型并释放显存
        功能：清理不常用的模型以节省资源
        """
        if model_name not in self.models:
            return

        print(f"卸载模型：{model_name}...")

        # 删除模型实例
        del self.models[model_name]
        del self.model_usage[model_name]

        # 强制垃圾回收
        gc.collect()
        torch.cuda.empty_cache()

        print(f"模型{model_name}已卸载")

    def get_model(self, model_name: str):
        """获取模型实例"""
        if model_name in self.models:
            self.model_usage[model_name] += 1
            return self.models[model_name]
        return None


# 使用示例
if __name__ == "__main__":
    manager = DynamicModelManager(max_models=2)

    # 加载模型1
    model1 = manager.load_model("llama-7b", "meta-llama/Llama-2-7b-hf")

    # 加载模型2
    model2 = manager.load_model("mistral-7b", "mistralai/Mistral-7B-v0.1")

    # 加载模型3，会自动卸载最少使用的模型1
    model3 = manager.load_model("codellama-7b", "codellama/CodeLlama-7b-hf")

    print(f"当前加载的模型：{list(manager.models.keys())}")