# 1. 安装SqueezeLLM
pip install squeezellm

# 2. 量化模型
python -m squeezellm.quant \
    --model meta-llama/Llama-2-70b-hf \
    --output Llama-2-70b-SqueezeLLM \
    --wbits 4 \
    --groupsize 128 \
    --sensitive-percent 0.01  # 1%权重保持FP16

# 3. 在vLLM中加载
from vllm import LLM

llm = LLM(
    model="Llama-2-70b-SqueezeLLM",
    quantization="squeezellm",
    dtype="float16"
)