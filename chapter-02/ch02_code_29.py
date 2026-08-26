# 启动vLLM API服务器
from vllm.entrypoints.openai.api_server import run_server


def start_vllm_server():
    """
    启动vLLM OpenAI兼容API服务器
    功能：提供与OpenAI相同的HTTP接口
    """
    # 基础配置
    server_config = {
        "host": "0.0.0.0",
        "port": 8000,
        "model": "meta-llama/Llama-2-70b-hf",
        "tensor_parallel_size": 4,
        "dtype": "float16",
        "max_model_len": 4096,
        "gpu_memory_utilization": 0.9,
    }

    # 高级配置
    advanced_config = {
        "enable_prefix_caching": True,  # 启用前缀缓存
        "disable_log_requests": False,  # 记录请求日志
        "max_num_seqs": 256,  # 最大并发序列数
        "max_num_batched_tokens": 8192,  # 最大批处理token数
    }

    # 合并配置
    config = {**server_config, **advanced_config}

    # 启动服务器
    run_server(**config)


# 使用示例
if __name__ == "__main__":
    print("启动vLLM OpenAI兼容API服务器...")
    start_vllm_server()