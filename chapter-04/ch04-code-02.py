# 文件：ch04_vllm_guided_decoding_http.py
# 说明：通过 OpenAI 兼容 HTTP 接口向 vLLM 服务端发送 JSON Schema 约束请求
# 依赖：pip install openai>=1.0.0
# 前置：已启动 vLLM 服务端，例如：
#   vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000
# 运行：python ch04_vllm_guided_decoding_http.py

import openai
import json

# JSON Schema 定义（与离线推理示例保持一致）
schema = {
    "type": "object",
    "properties": {
        "product_name": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100
        },
        "category": {
            "type": "string",
            "enum": ["electronics", "clothing", "food", "books", "other"]
        },
        "price": {
            "type": "number",
            "minimum": 0.0,
            "exclusiveMinimum": True
        },
        "in_stock": {
            "type": "boolean"
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 5
        }
    },
    "required": ["product_name", "category", "price", "in_stock"]
}

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-token"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "你是一个精确的商品信息结构化助手，"
                       "请严格按照指定格式输出 JSON。"
        },
        {
            "role": "user",
            "content": (
                "请根据以下描述生成商品信息：\n"
                "一款售价 299 元的无线蓝牙耳机，目前库存充足，"
                "属于电子产品，标签包含'音频'和'无线'。"
            )
        }
    ],
    response_format={
        "type": "json_schema",
        "json_schema": {
            "name":   "ProductInfo",
            "schema": schema,
            "strict": True
        }
    },
    temperature=0.0
)

result = json.loads(response.choices[0].message.content)
# result 是一个已验证符合 schema 的 Python 字典
print(result["category"])   # "electronics"
print(result["price"])      # 299.0