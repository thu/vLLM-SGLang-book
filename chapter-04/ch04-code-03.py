# 文件：ch04_sglang_guided_decoding_http.py
# 说明：通过 OpenAI 兼容 HTTP 接口向 SGLang 服务端发送 JSON Schema 约束请求
#       与 ch04_vllm_guided_decoding_http.py 仅 base_url 端口号不同，应用层代码零改动
# 依赖：pip install openai>=1.0.0
# 前置：已启动 SGLang 服务端，例如：
#   python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
#     --host 0.0.0.0 --port 30000
# 运行：python ch04_sglang_guided_decoding_http.py

import openai
import json

# JSON Schema 定义（与 vLLM 示例保持一致）
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
    base_url="http://localhost:30000/v1",
    api_key="None"
)

response = client.chat.completions.create(
    model="Qwen/Qwen2.5-7B-Instruct",
    messages=[
        {
            "role": "system",
            "content": "你是一个精确的商品信息结构化助手。"
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
print(result["category"])   # "electronics"
print(result["price"])      # 299.0
