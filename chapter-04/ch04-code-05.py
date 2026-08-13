# 文件：ch04_openai_compatible_tool_call.py
# 说明：演示通过 OpenAI 兼容接口发送工具调用请求
#       仅修改 base_url 端口号即可在 vLLM 与 SGLang 之间切换，应用层代码零改动
# 依赖：pip install openai>=1.0.0
# 前置：已启动 vLLM 或 SGLang 服务端，例如：
#   vLLM:   vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
#   SGLang: python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
#             --host 0.0.0.0 --port 30000
# 运行：python ch04_openai_compatible_tool_call.py

import openai

# 通过修改 base_url 在两个框架之间切换，应用代码零改动
# vLLM:   base_url = "http://localhost:8000/v1"
# SGLang: base_url = "http://localhost:30000/v1"
client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token"
)

# 带工具调用的请求
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
                },
                "required": ["city"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {"role": "user", "content": "北京今天天气怎么样？"}
    ],
    tools=tools,
    tool_choice="auto",
    stream=False
)

# 处理工具调用响应
msg = response.choices[0].message
if msg.tool_calls:
    call = msg.tool_calls[0]
    print(f"工具名称: {call.function.name}")
    print(f"调用参数: {call.function.arguments}")