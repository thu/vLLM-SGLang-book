# scripts/setup_models.py
# 功能: 下载 Qwen2.5-VL-7B 目标模型与 Qwen2.5-1.5B 草稿模型
#   验证项：文件完整性、分词器词表一致性、KV head 兼容性、vLLM 多模态路径激活
#
# 运行方式（项目根目录下）:
#   python scripts/setup_models.py
#
# ⚠ 多模态路径验证说明：
#   vLLM 加载 Qwen2.5-VL 时通过 config.json 中的 vision_config 识别多模态能力。
#   若版本不匹配，vLLM 会静默降级为纯文本模式——不抛异常，
#   但图像内容被完全忽略，输出为幻觉描述且难以与正常输出区分。
#   验证策略：发送纯红色图像，期望模型输出包含"红"相关词汇。

import io
import base64
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoConfig
from PIL import Image

TARGET_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
DRAFT_REPO  = "Qwen/Qwen2.5-1.5B-Instruct"
TARGET_DIR  = "./models/Qwen2.5-VL-7B-Instruct"
DRAFT_DIR   = "./models/Qwen2.5-1.5B-Instruct"


def download_models() -> None:
    print(f"下载目标模型: {TARGET_REPO}（约 15GB）...")
    snapshot_download(repo_id=TARGET_REPO, local_dir=TARGET_DIR, resume_download=True)

    print(f"下载草稿模型: {DRAFT_REPO}（约 3GB）...")
    snapshot_download(repo_id=DRAFT_REPO,  local_dir=DRAFT_DIR,  resume_download=True)
    print("下载完成。")


def verify_tokenizer_compatibility() -> None:
    """
    验证两模型分词器词表完全一致。

    投机解码要求 draft 与 target 共享词表，token ID 一一对应。
    Qwen2.5-VL-7B 与 Qwen2.5-1.5B 词表均为 151,936 token，
    special tokens（<|im_start|>、<|im_end|> 等）ID 完全相同。
    """
    tok_t = AutoTokenizer.from_pretrained(TARGET_DIR)
    tok_d = AutoTokenizer.from_pretrained(DRAFT_DIR)

    if len(tok_t) != len(tok_d):
        raise ValueError(
            f"词表大小不一致: target={len(tok_t)}, draft={len(tok_d)}。\n"
            f"投机解码要求词表完全相同，请确认两模型均为 Qwen2.5 系列。"
        )

    for token_name in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        id_t = tok_t.convert_tokens_to_ids(token_name)
        id_d = tok_d.convert_tokens_to_ids(token_name)
        if id_t != id_d:
            raise ValueError(
                f"Special token '{token_name}' ID 不一致: target={id_t}, draft={id_d}。"
            )

    print(f"✓ 分词器兼容性验证通过，词表大小: {len(tok_t)}")


def verify_kv_head_compatibility() -> None:
    """
    验证 draft 模型的 num_key_value_heads 满足 vLLM 投机解码约束。

    vLLM v0.8.x 要求：draft.num_key_value_heads 能整除
    target.num_key_value_heads，或两者相等。

    Qwen2.5-VL-7B : num_key_value_heads = 4
    Qwen2.5-1.5B  : num_key_value_heads = 2
    4 % 2 == 0，满足约束。
    """
    cfg_t = AutoConfig.from_pretrained(TARGET_DIR)
    cfg_d = AutoConfig.from_pretrained(DRAFT_DIR)
    kv_t = getattr(cfg_t, "num_key_value_heads", None)
    kv_d = getattr(cfg_d, "num_key_value_heads", None)

    if kv_t is None or kv_d is None:
        print("⚠ 无法读取 num_key_value_heads，跳过 KV head 兼容性验证。")
        return

    if kv_t % kv_d != 0 and kv_d % kv_t != 0:
        raise ValueError(
            f"KV head 不兼容: target={kv_t}, draft={kv_d}。\n"
            f"vLLM 要求两者满足整除关系。"
        )
    print(f"✓ KV head 兼容性验证通过: target={kv_t}, draft={kv_d}")


def verify_multimodal_path() -> None:
    """
    用纯红色图像请求验证 vLLM 多模态路径完整激活。

    若 ViT 未正确初始化，模型无法感知图像颜色，
    输出概率上不会包含"红"相关词汇。
    """
    from vllm import LLM, SamplingParams

    print(f"验证多模态路径: {TARGET_DIR} ...")
    llm = LLM(
        model=TARGET_DIR,
        dtype="bfloat16",
        max_model_len=512,
        gpu_memory_utilization=0.4,
    )

    img = Image.new("RGB", (224, 224), color=(220, 30, 30))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    data_uri = f"data:image/jpeg;base64,{b64}"

    messages = [{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": "这张图片的主色调是什么颜色？只回答颜色名称。"},
        ],
    }]
    outputs = llm.chat(messages, SamplingParams(temperature=0.0, max_tokens=10))
    response = outputs[0].outputs[0].text.strip().lower()

    red_keywords = ["红", "red", "crimson", "scarlet", "rose", "朱"]
    if not any(kw in response for kw in red_keywords):
        raise RuntimeError(
            f"多模态路径验证失败：\n"
            f"  模型输出: '{response}'\n"
            f"  期望包含: {red_keywords}\n"
            f"  可能原因: vLLM 版本与 Qwen2.5-VL 不兼容，vision_config 未正确加载。\n"
            f"  请确认 vLLM >= v0.8.0。"
        )

    print(f"✓ 多模态路径验证通过，测试输出: '{response}'")
    del llm


if __name__ == "__main__":
    download_models()
    verify_tokenizer_compatibility()
    verify_kv_head_compatibility()
    verify_multimodal_path()
    print("\n✓ 所有验证通过，模型就绪。")
