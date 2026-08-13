# 文件：ch04_sglang_guided_decoding_pydantic.py
# 说明：使用 Pydantic 模型作为约束目标，向 SGLang 服务端发送结构化输出请求
#       SGLang 服务端自动将 Pydantic 模型转换为 JSON Schema，响应直接解析为类型安全对象
# 依赖：pip install openai>=1.0.0 pydantic>=2.0.0
# 前置：已启动 SGLang 服务端，例如：
#   python -m sglang.launch_server --model-path Qwen/Qwen2.5-7B-Instruct \
#     --host 0.0.0.0 --port 30000
# 运行：python ch04_sglang_guided_decoding_pydantic.py

import openai
import json
from pydantic import BaseModel, Field
from typing import Literal, List, Optional

class ProductInfo(BaseModel):
    product_name: str = Field(min_length=1, max_length=100)
    category: Literal["electronics", "clothing", "food", "books", "other"]
    price: float = Field(gt=0.0)
    in_stock: bool
    tags: Optional[List[str]] = Field(default=None, max_length=5)

client = openai.OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="None"
)

# SGLang 服务端在接收请求时自动将 Pydantic 模型转换为 JSON Schema
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
            "schema": ProductInfo.model_json_schema(),
            "strict": True
        }
    },
    temperature=0.0
)

# 直接解析为 Pydantic 对象，类型安全有保证
product = ProductInfo.model_validate_json(
    response.choices[0].message.content
)
print(product.category)     # Literal 类型，IDE 自动补全
print(product.price > 0)    # True，类型系统保证