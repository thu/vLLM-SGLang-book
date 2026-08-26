def configure_multi_model_deployment(models: list) -> dict:
    """
    配置多模型部署
    功能：根据模型大小自动分配GPU资源
    """
    total_gpus = torch.cuda.device_count()
    total_memory = torch.cuda.get_device_properties(0).total_memory

    configs = {}

    for model in models:
        model_name = model['name']
        model_size_gb = model['size_gb']

        # 计算需要的GPU数量
        gpus_needed = max(1, int(model_size_gb / (total_memory / 1e9 * 0.9)))

        # 分配配置
        configs[model_name] = {
            'tensor_parallel_size': gpus_needed,
            'gpu_memory_utilization': 0.9 if gpus_needed == 1 else 0.95,
            'max_num_seqs': 256 // gpus_needed,  # 大模型减少并发数
        }

    return configs


# 使用示例
if __name__ == "__main__":
    models = [
        {'name': 'llama-7b', 'size_gb': 14},
        {'name': 'llama-70b', 'size_gb': 140},
        {'name': 'codellama-34b', 'size_gb': 68},
    ]

    configs = configure_multi_model_deployment(models)
    for model_name, config in configs.items():
        print(f"{model_name}配置：{config}")