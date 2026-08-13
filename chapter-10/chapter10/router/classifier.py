"""
文件: chapter10/router/classifier.py

请求特征分类器：提取请求体特征向量并输出路由决策。

路由决策优先级（从高到低）:
  1. 携带 JSON Schema 约束（response_format.type = json_schema/json_object）
     → SGLang（利用原生 constrained decoding，保证输出格式合规）
  2. 携带工具调用定义（tools 字段非空）
     → SGLang（利用函数调用编排能力）
  3. 系统提示超过 2048 tokens（估算值）
     → SGLang（利用 RadixAttention 前缀缓存复用，减少重复计算）
  4. 其余请求
     → vLLM（高并发短文本，充分利用 continuous batching 吞吐优势）

性能说明:
  系统提示长度估算采用字符数近似方案，而非 tiktoken 精确编码。
  tiktoken.encode() 对2048-token级别的文本耗时约1-3ms；
  在QPS达到数百时，同步调用会成为路由中间件的明显瓶颈。
  字符数估算（中文约1.5字符/token，英文约4字符/token）精度足够用于
  2048 token 阈值的粗判断，P99 误判率对路由质量的影响可接受。
  若业务对路由精度有极高要求，可将 USE_EXACT_TOKEN_COUNT 设为 True，
  但需评估对路由服务吞吐量的影响。
"""

from __future__ import annotations

import os
from enum import Enum
from typing import Optional

from pydantic import BaseModel

# 是否使用精确token计数（默认False，使用字符数估算）
# 精确模式适用于QPS较低（<100）且对路由准确性要求极高的场景
USE_EXACT_TOKEN_COUNT = os.getenv("ROUTER_EXACT_TOKEN_COUNT", "false").lower() == "true"

# 系统提示长度阈值（token数），超过此值优先路由至SGLang利用前缀缓存
SYSTEM_PROMPT_TOKEN_THRESHOLD = int(os.getenv("ROUTER_SYSTEM_PROMPT_THRESHOLD", "2048"))


# ── 枚举与数据模型 ─────────────────────────────────────────────────────────

class RouteTarget(str, Enum):
    """路由目标框架"""
    VLLM = "vllm"
    SGLANG = "sglang"


class RoutingDecision(BaseModel):
    """路由决策结果"""
    target: RouteTarget
    reason: str          # 触发路由的原因标识，用于监控和调试
    confidence: float    # 决策置信度（0.0-1.0），用于路由质量监控


class RequestFeatures(BaseModel):
    """从请求体提取的特征向量"""
    has_response_format: bool    # 是否携带 JSON Schema 或 json_object 约束
    has_tool_calls: bool         # 是否携带工具调用定义（tools 字段）
    system_prompt_tokens: int    # 系统提示估算 token 数
    max_tokens: int              # 请求指定的最大输出 token 数
    estimated_complexity: float  # 综合复杂度评分（0.0-1.0）


# ── Token 数量估算 ──────────────────────────────────────────────────────────

def _estimate_tokens_by_chars(text: str) -> int:
    """
    通过字符数估算 token 数量（近似方案，避免 tiktoken 的同步编码开销）。

    估算规则:
      - ASCII 字符（英文、数字、标点）：平均约4字符/token → 乘以0.25
      - 非ASCII字符（中文、日文、韩文等）：平均约1.5字符/token → 乘以0.67
    
    实测误差范围：对于混合中英文文本，与 tiktoken 精确值偏差通常在 ±15% 以内，
    足够支撑以2048 token为阈值的路由分支决策。
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    non_ascii_chars = len(text) - ascii_chars
    return int(ascii_chars * 0.25 + non_ascii_chars * 0.67)


def _count_tokens_exact(text: str) -> int:
    """
    使用 tiktoken 精确计算 token 数（cl100k_base 编码）。
    仅在 USE_EXACT_TOKEN_COUNT=True 时调用。
    注意: 首次调用会触发编码器加载（约50-100ms），后续调用约1-3ms。
    """
    import tiktoken  # 延迟导入，精确模式未启用时不引入依赖
    enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))


def count_system_prompt_tokens(text: str) -> int:
    """
    根据配置选择 token 计数方式。
    默认使用字符数估算（高性能），可通过环境变量切换为精确模式。
    """
    if USE_EXACT_TOKEN_COUNT:
        return _count_tokens_exact(text)
    return _estimate_tokens_by_chars(text)


# ── 特征提取 ───────────────────────────────────────────────────────────────

def extract_features(payload: dict) -> RequestFeatures:
    """
    从 OpenAI 格式请求体中提取路由决策所需的特征向量。
    支持 /v1/chat/completions 标准格式。

    Args:
        payload: 已解析的请求体字典

    Returns:
        RequestFeatures: 特征向量，包含所有路由决策所需维度
    """
    # ── 特征1: JSON Schema 约束检测 ────────────────────────────────────────
    # response_format.type 为 "json_schema" 或 "json_object" 时触发
    response_format = payload.get("response_format", {})
    has_schema = (
        isinstance(response_format, dict)
        and response_format.get("type") in ("json_schema", "json_object")
    )

    # ── 特征2: 工具调用检测 ────────────────────────────────────────────────
    tools = payload.get("tools", [])
    has_tools = isinstance(tools, list) and len(tools) > 0

    # ── 特征3: 系统提示长度 ────────────────────────────────────────────────
    # 遍历 messages 列表，找到第一条 role=system 的消息并估算其 token 数
    # 系统提示通常在多轮对话中被复用，是 RadixAttention 前缀缓存的主要命中对象
    system_tokens = 0
    messages = payload.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                system_tokens = count_system_prompt_tokens(content)
            break  # 只检查第一条系统提示

    # ── 特征4: 最大输出长度 ────────────────────────────────────────────────
    max_tokens = int(payload.get("max_tokens", 512))

    # ── 综合复杂度评分（0.0-1.0）─────────────────────────────────────────
    # 各特征的权重反映其对计算复杂度的影响程度：
    #   Schema约束（0.6）> 工具调用（0.3）> 长前缀（0.2）> 长输出（0.1）
    complexity = 0.0
    if has_schema:
        complexity += 0.6
    if has_tools:
        complexity += 0.3
    if system_tokens > SYSTEM_PROMPT_TOKEN_THRESHOLD:
        complexity += 0.2
    if max_tokens > 2048:
        complexity += 0.1
    complexity = min(complexity, 1.0)

    return RequestFeatures(
        has_response_format=has_schema,
        has_tool_calls=has_tools,
        system_prompt_tokens=system_tokens,
        max_tokens=max_tokens,
        estimated_complexity=complexity,
    )


# ── 路由决策 ───────────────────────────────────────────────────────────────

def make_routing_decision(features: RequestFeatures) -> RoutingDecision:
    """
    基于特征向量输出路由决策。

    决策逻辑采用优先级规则链（短路求值），确保高确定性规则优先执行：
      P1: Schema约束  → SGLang（必须，无例外）
      P2: 工具调用    → SGLang（强烈偏好）
      P3: 长前缀      → SGLang（缓存收益明确）
      P4: 默认        → vLLM（通用短文本最优）

    Args:
        features: 由 extract_features() 生成的特征向量

    Returns:
        RoutingDecision: 包含目标框架、触发原因和置信度
    """
    # 规则P1: JSON Schema 约束 → 必须路由至 SGLang
    # SGLang v0.5.x 原生支持 constrained decoding（基于 FSM 的 token 掩码）。
    # vLLM v0.8.x 对 json_schema 的支持依赖外部库，在复杂 Schema 下可能输出不合规。
    if features.has_response_format:
        return RoutingDecision(
            target=RouteTarget.SGLANG,
            reason="json_schema_constraint",
            confidence=1.0,
        )

    # 规则P2: 工具调用 → 路由至 SGLang
    # SGLang 的函数调用编排能力更完善，支持并行工具调用和多步骤结果汇聚。
    if features.has_tool_calls:
        return RoutingDecision(
            target=RouteTarget.SGLANG,
            reason="tool_call_workflow",
            confidence=0.95,
        )

    # 规则P3: 长系统提示 → 优先路由至 SGLang
    # 系统提示超过阈值的请求具有高前缀复用潜力（多轮对话场景）。
    # SGLang 的 RadixAttention 可将相同前缀的 KV Cache 在多个请求间共享，
    # 避免重复计算，提升吞吐并降低 TTFT。
    if features.system_prompt_tokens > SYSTEM_PROMPT_TOKEN_THRESHOLD:
        return RoutingDecision(
            target=RouteTarget.SGLANG,
            reason="long_prefix_cache_reuse",
            confidence=0.85,
        )

    # 规则P4: 默认路由 → vLLM
    # 高并发短文本请求由 vLLM 的 continuous batching 机制处理。
    # vLLM 的调度开销低于 SGLang，在短请求场景下 P99 延迟更低。
    return RoutingDecision(
        target=RouteTarget.VLLM,
        reason="default_general_request",
        confidence=0.90,
    )


# ── 公共接口 ───────────────────────────────────────────────────────────────

def classify(payload: dict) -> RoutingDecision:
    """
    分类器公共接口：接收请求体字典，返回路由决策。

    供 server.py 的 FastAPI 端点和测试用例直接调用。
    """
    features = extract_features(payload)
    return make_routing_decision(features)
