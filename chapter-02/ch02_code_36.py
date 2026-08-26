from openai import OpenAI

# 通过反向代理访问不同模型
client = OpenAI(base_url="http://localhost:8000", api_key="EMPTY")

# 使用LLaMA-70B
response = client.chat.completions.create(
    model="llama-70b",  # 会路由到localhost:8001
    messages=[{"role": "user", "content": "Explain AI."}],
    max_tokens=100
)

# 使用CodeLLaMA-34B
response = client.chat.completions.create(
    model="codellama-34b",  # 会路由到localhost:8002
    messages=[{"role": "user", "content": "Write a sorting function."}],
    max_tokens=100
)