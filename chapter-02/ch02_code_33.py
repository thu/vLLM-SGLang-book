import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")


async def process_batch_requests():
    """
    批量处理请求
    功能：展示vLLM的高并发处理能力
    """
    # 创建100个并发请求
    tasks = []
    for i in range(100):
        task = client.chat.completions.create(
            model="meta-llama/Llama-2-70b-hf",
            messages=[
                {"role": "user", "content": f"What is {i} + {i}?"}
            ],
            max_tokens=50
        )
        tasks.append(task)

    # 并发执行
    import time
    start = time.time()
    responses = await asyncio.gather(*tasks)
    end = time.time()

    print(f"处理100个请求耗时：{end - start:.2f}秒")
    print(f"平均延迟：{(end - start) / 100 * 1000:.1f}ms/请求")
    print(f"吞吐量：{100 / (end - start):.1f}请求/秒")


# 使用示例
if __name__ == "__main__":
    asyncio.run(process_batch_requests())