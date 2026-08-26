# 启动时启用前缀缓存
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-70b-hf \
    --enable-prefix-caching

# 在代码中使用
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# 多个请求共享相同的系统提示
system_prompt = "You are an expert Python programmer. " * 100  # 长系统提示

for i in range(10):
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[
            {"role": "system", "content": system_prompt},  # 共享前缀
            {"role": "user", "content": f"Write function {i}"}
        ],
        max_tokens=100
    )
    print(f"请求{i}完成")

# 使用示例
if __name__ == "__main__":
    print("测试前缀缓存...")
    # 第一次请求会计算完整的系统提示
    # 后续请求会复用缓存的KV Cache