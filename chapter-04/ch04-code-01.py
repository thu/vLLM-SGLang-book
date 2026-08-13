# 文件：ch04_vllm_guided_decoding_offline.py
# 说明：使用 vLLM 离线推理接口 + GuidedDecodingParams 实现 JSON Schema 约束解码
# 依赖：pip install vllm>=0.18.0
# 运行：python ch04_vllm_guided_decoding_offline.py

from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct")

# 定义包含多种约束类型的 JSON Schema
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

guided = GuidedDecodingParams(json=schema)
params = SamplingParams(
    temperature=0.0,    # 结构化抽取通常使用贪婪解码以保证确定性
    max_tokens=256,
    guided_decoding=guided
)

prompt = (
    "请根据以下商品描述生成结构化的商品信息 JSON：\n"
    "一款售价 299 元的无线蓝牙耳机，目前库存充足，"
    "属于电子产品类别，标签包含'音频'和'无线'。\n\n"
    "JSON 输出："
)

outputs = llm.generate([prompt], params)
# 输出保证完全符合 schema，例如：
# {"product_name": "无线蓝牙耳机", "category": "electronics",
#  "price": 299.0, "in_stock": true, "tags": ["音频", "无线"]}
print(outputs[0].outputs[0].text)