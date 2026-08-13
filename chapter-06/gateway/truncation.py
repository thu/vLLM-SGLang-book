#!/usr/bin/env python3
# ch06/gateway/truncation.py
# 输入截断模块
#
# 目的：在请求进入 vLLM 队列前截断超长输入，
#       防止少数长请求在 prefill 阶段阻塞批处理队列，推高 P99 延迟。
#
# 截断策略：保留系统提示词，对用户消息从末尾截断（保留最后 N 个 token）。
# 保留末尾而非开头，是因为客服问题的关键意图通常在末尾。

from functools import lru_cache
from transformers import PreTrainedTokenizerBase

MAX_USER_INPUT_TOKENS = 1024  # 超出此长度则截断

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct-AWQ"


@lru_cache(maxsize=1)
def _get_tokenizer() -> PreTrainedTokenizerBase:
    """
    懒加载 tokenizer，仅用于 token 计数，不用于解码完整序列。
    lru_cache 确保多进程/多请求下只初始化一次。
    """
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)


def truncate_messages(messages: list[dict]) -> list[dict]:
    """
    检查并截断消息列表中最后一条用户消息。

    只截断最后一条 user 消息，原因：
    - 系统提示词（system）不截断，它定义了模型行为边界
    - 历史对话消息已经通过了前一轮的截断检查
    - 当前轮的用户输入是最可能超长的部分
    """
    if not messages:
        return messages

    # 找到最后一条 user 消息的索引
    last_user_idx = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "user":
            last_user_idx = i
            break

    if last_user_idx is None:
        return messages

    content = messages[last_user_idx].get("content", "")
    if not isinstance(content, str):
        # content 为列表格式（多模态）时跳过截断
        return messages

    tokenizer = _get_tokenizer()
    token_ids = tokenizer.encode(content, add_special_tokens=False)

    if len(token_ids) <= MAX_USER_INPUT_TOKENS:
        return messages

    # 截断：保留最后 MAX_USER_INPUT_TOKENS 个 token
    truncated_ids = token_ids[-MAX_USER_INPUT_TOKENS:]
    truncated_content = tokenizer.decode(truncated_ids, skip_special_tokens=True)

    # 浅拷贝消息列表，避免修改原始请求对象
    messages = list(messages)
    messages[last_user_idx] = dict(messages[last_user_idx])
    messages[last_user_idx]["content"] = truncated_content

    return messages
