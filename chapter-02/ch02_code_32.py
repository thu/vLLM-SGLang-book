from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


def test_advanced_sampling():
    """
    测试高级采样参数
    功能：展示各种采样策略的效果
    """
    # 策略1：确定性生成，temperature=0
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[{"role": "user", "content": "What is 2+2?"}],
        temperature=0.0,  # 完全确定性
        max_tokens=50
    )
    print(f"确定性生成：{response.choices[0].message.content}")

    # 策略2：创意生成，高temperature + top_p
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[{"role": "user", "content": "Write a creative story."}],
        temperature=1.5,  # 高随机性
        top_p=0.95,  # 核采样
        max_tokens=200
    )
    print(f"创意生成：{response.choices[0].message.content[:100]}...")

    # 策略3：束搜索，vLLM特有
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[{"role": "user", "content": "Translate to French: Hello"}],
        use_beam_search=True,  # vLLM扩展参数
        best_of=5,  # 生成5个候选
        max_tokens=50
    )
    print(f"束搜索结果：{response.choices[0].message.content}")

    # 策略4：重复惩罚
    response = client.chat.completions.create(
        model="meta-llama/Llama-2-70b-hf",
        messages=[{"role": "user", "content": "List 10 colors:"}],
        presence_penalty=0.6,  # 避免重复主题
        frequency_penalty=0.3,  # 避免重复词汇
        max_tokens=100
    )
    print(f"重复惩罚结果：{response.choices[0].message.content}")


# 使用示例
if __name__ == "__main__":
    test_advanced_sampling()