from openai import OpenAI

# 配置客户端指向vLLM服务器
client = OpenAI(
    api_key="EMPTY",  # vLLM不需要API key
    base_url="http://localhost:8000/v1",
)


def test_chat_completion():
    """
    测试聊天补全功能
    功能：使用OpenAI SDK进行聊天对话
    """
    # 发送聊天请求
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Explain quantum computing in simple terms."}
        ],
        temperature=0.7,
        max_tokens=256,
        top_p=0.9,
    )

    # 打印响应
    print(f"响应：{response.choices[0].message.content}")
    print(f"使用的tokens:{response.usage.total_tokens}")


def test_text_completion():
    """
    测试文本补全功能
    功能：使用OpenAI SDK进行文本续写
    """
    response = client.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        prompt="Once upon a time in a distant galaxy,",
        max_tokens=100,
        temperature=0.8,
    )

    print(f"补全文本：{response.choices[0].text}")


def test_streaming():
    """
    测试流式输出
    功能：实时接收生成的token
    """
    print("流式输出：")
    stream = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[
            {"role": "user", "content": "Write a short poem about AI."}
        ],
        max_tokens=200,
        stream=True,  # 启用流式输出
    )

    for chunk in stream:
        if chunk.choices[0].delta.content is not None:
            print(chunk.choices[0].delta.content, end="", flush=True)
    print()


# 使用示例
if __name__ == "__main__":
    print("=== 测试聊天补全 ===")
    test_chat_completion()

    print("\n=== 测试文本补全 ===")
    test_text_completion()

    print("\n=== 测试流式输出 ===")
    test_streaming()