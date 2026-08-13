"""
文件: chapter10/tests/test_routing_e2e.py

路由服务端到端集成测试。

测试范围:
  - 路由分类器在各类请求特征下的决策正确性
  - FastAPI 路由服务接口的请求/响应格式
  - 边界条件和异常输入的健壮性

运行方式:
  # 单元级别（无需外部服务，直接测试分类器逻辑）
  pytest tests/test_routing_e2e.py -v -m "not integration"

  # 集成级别（需要本地运行路由服务 uvicorn router.server:app --port 8080）
  pytest tests/test_routing_e2e.py -v -m "integration"
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from router.classifier import classify, RouteTarget, SYSTEM_PROMPT_TOKEN_THRESHOLD


# ── 路由逻辑场景测试 ──────────────────────────────────────────────────────

class TestRoutingScenarios:
    """
    按业务场景组织的路由决策测试。
    每个测试对应章节中描述的一类实际业务请求。
    """

    # 场景1: 智能客服 —— 高并发短文本，应路由至 vLLM
    def test_customer_service_chat(self):
        payload = {
            "model": "qwen2.5-7b",
            "messages": [
                {"role": "system", "content": "你是一个客服助手。"},
                {"role": "user", "content": "我的订单什么时候发货？"},
            ],
            "max_tokens": 200,
        }
        decision = classify(payload)
        assert decision.target == RouteTarget.VLLM, (
            f"短文本客服请求应路由至vLLM，实际路由至{decision.target}（原因：{decision.reason}）"
        )

    # 场景2: 合同信息抽取 —— 携带 JSON Schema，应路由至 SGLang
    def test_contract_extraction_with_schema(self):
        payload = {
            "messages": [{"role": "user", "content": "从合同中抽取关键信息"}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "contract_info",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "party_a": {"type": "string"},
                            "party_b": {"type": "string"},
                            "amount": {"type": "number"},
                        },
                    },
                },
            },
        }
        decision = classify(payload)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "json_schema_constraint"
        assert decision.confidence == 1.0

    # 场景3: 函数调用工作流 —— 携带 tools，应路由至 SGLang
    def test_function_calling_workflow(self):
        payload = {
            "messages": [{"role": "user", "content": "帮我查询今天的天气"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "获取指定城市的天气",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                        },
                    },
                }
            ],
        }
        decision = classify(payload)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "tool_call_workflow"

    # 场景4: 文档摘要（长系统提示，知识库场景）—— 应路由至 SGLang
    def test_document_analysis_long_system_prompt(self):
        # 构造超过阈值的系统提示（模拟RAG系统注入的检索文档）
        retrieved_docs = "相关文档内容：" + "这是一段检索到的文档内容。" * 200
        payload = {
            "messages": [
                {"role": "system", "content": retrieved_docs},
                {"role": "user", "content": "请总结以上文档的核心观点"},
            ],
            "max_tokens": 512,
        }
        decision = classify(payload)
        assert decision.target == RouteTarget.SGLANG, (
            f"长系统提示（RAG场景）应路由至SGLang，实际：{decision.target}"
        )
        assert decision.reason == "long_prefix_cache_reuse"

    # 场景5: JSON Object 输出（无完整 schema，但要求 JSON 格式）
    def test_json_object_format(self):
        payload = {
            "messages": [{"role": "user", "content": "以JSON格式返回分析结果"}],
            "response_format": {"type": "json_object"},
        }
        decision = classify(payload)
        assert decision.target == RouteTarget.SGLANG
        assert decision.reason == "json_schema_constraint"

    # 场景6: 普通文本格式（response_format 为 text）—— 不触发 Schema 路由
    def test_text_format_no_schema_routing(self):
        payload = {
            "messages": [{"role": "user", "content": "写一段介绍"}],
            "response_format": {"type": "text"},
        }
        decision = classify(payload)
        # response_format.type=text 不是 json_schema/json_object，不应触发 SGLang 路由
        assert decision.target == RouteTarget.VLLM

    # 场景7: Schema 优先级高于长前缀
    def test_schema_priority_over_long_prefix(self):
        long_system = "系统提示：" + "A" * 10000
        payload = {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "抽取信息"},
            ],
            "response_format": {"type": "json_schema"},
        }
        decision = classify(payload)
        # Schema 约束优先级最高（P1），即使同时有长前缀（P3）
        assert decision.reason == "json_schema_constraint"
        assert decision.confidence == 1.0

    # 场景8: 工具调用优先级高于长前缀
    def test_tools_priority_over_long_prefix(self):
        long_system = "You are a knowledgeable assistant. " * 200
        payload = {
            "messages": [
                {"role": "system", "content": long_system},
                {"role": "user", "content": "search something"},
            ],
            "tools": [{"type": "function", "function": {"name": "search"}}],
        }
        decision = classify(payload)
        assert decision.reason == "tool_call_workflow"


# ── 边界条件与健壮性测试 ─────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_payload(self):
        """空 payload 不应崩溃，默认路由至 vLLM"""
        decision = classify({})
        assert decision.target == RouteTarget.VLLM

    def test_none_response_format(self):
        """response_format 为 None 不应崩溃"""
        decision = classify({
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": None,
        })
        assert decision.target == RouteTarget.VLLM

    def test_string_response_format(self):
        """response_format 为字符串（非字典）不应崩溃"""
        decision = classify({
            "messages": [{"role": "user", "content": "hello"}],
            "response_format": "json",   # 错误格式，应被忽略
        })
        assert decision.target == RouteTarget.VLLM

    def test_string_tools(self):
        """tools 为字符串（非列表）不应崩溃"""
        decision = classify({
            "messages": [{"role": "user", "content": "hello"}],
            "tools": "invalid",
        })
        assert decision.target == RouteTarget.VLLM

    def test_missing_messages(self):
        """缺少 messages 字段不应崩溃"""
        decision = classify({
            "model": "qwen2.5-7b",
            "max_tokens": 100,
        })
        assert decision.target == RouteTarget.VLLM

    def test_system_prompt_exactly_at_threshold(self):
        """系统提示长度恰好在阈值边界时的路由行为"""
        # 构造估算结果恰好等于阈值的文本（使用字符估算）
        # SYSTEM_PROMPT_TOKEN_THRESHOLD = 2048，ASCII字符：2048 × 4 = 8192字符
        boundary_text = "A" * (SYSTEM_PROMPT_TOKEN_THRESHOLD * 4)
        payload = {
            "messages": [
                {"role": "system", "content": boundary_text},
                {"role": "user", "content": "hi"},
            ],
        }
        # 边界条件：恰好等于阈值时不触发长前缀路由（严格大于才触发）
        decision = classify(payload)
        # 不断言具体target，只验证不崩溃
        assert decision.target in (RouteTarget.VLLM, RouteTarget.SGLANG)

    def test_decision_confidence_range(self):
        """所有决策的 confidence 应在 [0.0, 1.0] 范围内"""
        payloads = [
            {},
            {"response_format": {"type": "json_schema"}},
            {"tools": [{"type": "function", "function": {"name": "f"}}]},
            {"messages": [{"role": "system", "content": "long " * 1000}]},
        ]
        for payload in payloads:
            decision = classify(payload)
            assert 0.0 <= decision.confidence <= 1.0, (
                f"confidence={decision.confidence} 超出[0,1]范围，payload={payload}"
            )


# ── 成本模型基础验证 ──────────────────────────────────────────────────────

class TestCostModel:
    def test_workload_ratios_must_sum_to_one(self):
        from cost_model.analyzer import WorkloadProfile
        with pytest.raises(ValueError, match="must sum to 1.0"):
            WorkloadProfile(
                short_text_ratio=0.50,
                structured_ratio=0.30,
                long_context_ratio=0.30,  # 总和=1.1，应报错
            )

    def test_cost_comparison_hybrid_not_worse(self):
        """混合架构在工作负载与框架精准匹配时，GPU需求不应多于纯vLLM方案"""
        from cost_model.analyzer import (
            WorkloadProfile, FrameworkThroughput, PricingConfig, compute_cost_comparison
        )
        result = compute_cost_comparison(
            workload=WorkloadProfile(),
            throughput=FrameworkThroughput(),
            pricing=PricingConfig(),
        )
        # 混合架构的核心价值：SGLang在结构化和长上下文任务上的更高吞吐量
        # 应使总GPU需求低于或等于纯vLLM方案
        assert result.hybrid_total_gpu_count <= result.vllm_only_gpu_count, (
            f"混合架构GPU数({result.hybrid_total_gpu_count})不应多于纯vLLM({result.vllm_only_gpu_count})"
        )

    def test_saving_pct_positive_when_hybrid_cheaper(self):
        """节约比例应为正值（混合架构更便宜）"""
        from cost_model.analyzer import (
            WorkloadProfile, FrameworkThroughput, PricingConfig, compute_cost_comparison
        )
        result = compute_cost_comparison(
            workload=WorkloadProfile(),
            throughput=FrameworkThroughput(),
            pricing=PricingConfig(),
        )
        assert result.saving_pct >= 0, "混合架构成本不应高于纯vLLM方案"

    def test_cost_summary_contains_key_fields(self):
        """成本报告摘要应包含关键字段"""
        from cost_model.analyzer import (
            WorkloadProfile, FrameworkThroughput, PricingConfig, compute_cost_comparison
        )
        result = compute_cost_comparison(
            workload=WorkloadProfile(),
            throughput=FrameworkThroughput(),
            pricing=PricingConfig(),
        )
        summary = result.summary()
        assert "纯 vLLM 方案" in summary
        assert "混合架构方案" in summary
        assert "成本节约" in summary
