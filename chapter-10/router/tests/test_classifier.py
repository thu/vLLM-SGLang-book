"""
文件: chapter10/router/tests/test_classifier.py

路由分类器单元测试。

运行方式:
  # 在 chapter10/ 目录下执行
  pytest router/tests/test_classifier.py -v
"""

import pytest
import sys
import os

# 将 chapter10/ 加入 Python 路径，使 router 包可被导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from router.classifier import (
    classify,
    extract_features,
    make_routing_decision,
    count_system_prompt_tokens,
    RouteTarget,
    RequestFeatures,
    SYSTEM_PROMPT_TOKEN_THRESHOLD,
)


# ── Token 估算测试 ─────────────────────────────────────────────────────────

class TestTokenEstimation:
    def test_empty_string(self):
        assert count_system_prompt_tokens("") == 0

    def test_short_ascii(self):
        # "hello world" = 11字符，估算约2-3 token，应远低于阈值
        result = count_system_prompt_tokens("hello world")
        assert 1 <= result <= 5

    def test_long_ascii_exceeds_threshold(self):
        # 构造一段肯定超过2048 token的英文文本（约8200字符 ÷ 4 ≈ 2050 tokens）
        long_text = "A" * 8500
        result = count_system_prompt_tokens(long_text)
        assert result > SYSTEM_PROMPT_TOKEN_THRESHOLD

    def test_chinese_text_estimation(self):
        # 中文字符：每字约0.67 token，3000字 ≈ 2010 tokens（接近阈值）
        chinese_text = "你好" * 1500   # 3000个汉字
        result = count_system_prompt_tokens(chinese_text)
        # 估算值应在合理范围内（不要求与tiktoken精确一致）
        assert 1500 <= result <= 3000


# ── 特征提取测试 ──────────────────────────────────────────────────────────

class TestExtractFeatures:
    def test_simple_chat_no_special_features(self):
        payload = {
            "model": "qwen2.5-7b",
            "messages": [{"role": "user", "content": "hello"}],
            "max_tokens": 100,
        }
        features = extract_features(payload)
        assert features.has_response_format is False
        assert features.has_tool_calls is False
        assert features.system_prompt_tokens == 0
        assert features.max_tokens == 100
        assert features.estimated_complexity < 0.1

    def test_json_schema_detected(self):
        payload = {
            "messages": [{"role": "user", "content": "extract info"}],
            "response_format": {"type": "json_schema", "json_schema": {"name": "result"}},
        }
        features = extract_features(payload)
        assert features.has_response_format is True
        assert features.estimated_complexity >= 0.6

    def test_json_object_also_detected(self):
        payload = {
            "messages": [{"role": "user", "content": "output json"}],
            "response_format": {"type": "json_object"},
        }
        features = extract_features(payload)
        assert features.has_response_format is True

    def test_tools_detected(self):
        payload = {
            "messages": [{"role": "user", "content": "call tool"}],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }
        features = extract_features(payload)
        assert features.has_tool_calls is True
        assert features.estimated_complexity >= 0.3

    def test_empty_tools_not_detected(self):
        payload = {
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [],
        }
        features = extract_features(payload)
        assert features.has_tool_calls is False

    def test_long_system_prompt_detected(self):
        # 构造超过阈值的系统提示（约8500英文字符 ≈ 2125 tokens > 2048阈值）
        long_system = "You are a helpful assistant. " * 300
        payload = {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "hello"},
            ],
        }
        features = extract_features(payload)
        assert features.system_prompt_tokens > SYSTEM_PROMPT_TOKEN_THRESHOLD

    def test_short_system_prompt_not_threshold(self):
        payload = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
        }
        features = extract_features(payload)
        assert features.system_prompt_tokens < SYSTEM_PROMPT_TOKEN_THRESHOLD

    def test_default_max_tokens(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        features = extract_features(payload)
        assert features.max_tokens == 512   # 默认值

    def test_complexity_capped_at_one(self):
        # Schema + 工具 + 长前缀 + 长输出 → 复杂度不超过1.0
        long_system = "system " * 1500
        payload = {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "x"},
            ],
            "response_format": {"type": "json_schema"},
            "tools": [{"type": "function", "function": {"name": "f"}}],
            "max_tokens": 4096,
        }
        features = extract_features(payload)
        assert features.estimated_complexity == 1.0


# ── 路由决策测试 ─────────────────────────────────────────────────────────

class TestMakeRoutingDecision:
    def _features(self, **kwargs) -> RequestFeatures:
        defaults = dict(
            has_response_format=False,
            has_tool_calls=False,
            system_prompt_tokens=0,
            max_tokens=512,
            estimated_complexity=0.0,
        )
        defaults.update(kwargs)
        return RequestFeatures(**defaults)

    def test_schema_routes_to_sglang_with_full_confidence(self):
        features = self._features(has_response_format=True)
        decision = make_routing_decision(features)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "json_schema_constraint"
        assert decision.confidence == 1.0

    def test_tools_routes_to_sglang(self):
        features = self._features(has_tool_calls=True)
        decision = make_routing_decision(features)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "tool_call_workflow"
        assert decision.confidence >= 0.9

    def test_long_prefix_routes_to_sglang(self):
        features = self._features(system_prompt_tokens=SYSTEM_PROMPT_TOKEN_THRESHOLD + 1)
        decision = make_routing_decision(features)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "long_prefix_cache_reuse"

    def test_default_routes_to_vllm(self):
        features = self._features()
        decision = make_routing_decision(features)
        assert decision.target == RouteTarget.VLLM
        assert decision.reason == "default_general_request"

    def test_schema_priority_over_tools(self):
        # Schema约束优先级高于工具调用
        features = self._features(has_response_format=True, has_tool_calls=True)
        decision = make_routing_decision(features)
        assert decision.reason == "json_schema_constraint"

    def test_tools_priority_over_long_prefix(self):
        # 工具调用优先级高于长前缀
        features = self._features(
            has_tool_calls=True,
            system_prompt_tokens=SYSTEM_PROMPT_TOKEN_THRESHOLD + 100,
        )
        decision = make_routing_decision(features)
        assert decision.reason == "tool_call_workflow"


# ── 端到端 classify 接口测试 ─────────────────────────────────────────────

class TestClassifyEndToEnd:
    def test_plain_chat(self):
        decision = classify({
            "model": "qwen2.5-7b",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert decision.target == RouteTarget.VLLM

    def test_structured_output(self):
        decision = classify({
            "messages": [{"role": "user", "content": "parse"}],
            "response_format": {"type": "json_schema"},
        })
        assert decision.target == RouteTarget.SGLANG

    def test_empty_payload_defaults_to_vllm(self):
        decision = classify({})
        assert decision.target == RouteTarget.VLLM

    def test_malformed_response_format_ignored(self):
        # response_format 不是字典时不应崩溃
        decision = classify({
            "messages": [{"role": "user", "content": "x"}],
            "response_format": "invalid_string",
        })
        assert decision.target == RouteTarget.VLLM

    def test_malformed_tools_ignored(self):
        # tools 不是列表时不应崩溃
        decision = classify({
            "messages": [{"role": "user", "content": "x"}],
            "tools": "not_a_list",
        })
        assert decision.target == RouteTarget.VLLM
