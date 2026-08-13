# tests/test_memory_budget.py
# KV Cache 容量计算单元测试

import pytest

from inference.memory_budget import (
    MemoryReport,
    compute_cache_capacity,
    compute_kv_bytes_per_token,
)


class TestComputeKvBytesPerToken:
    """compute_kv_bytes_per_token() 的单元测试。"""

    def test_qwen2_72b_value(self):
        """
        测试：Qwen2-72B 的 KV Cache 单 token 大小。

        公式：80层 × 8头 × 128维 × 2（K+V）× 2字节（FP16）
            = 327,680 bytes
        """
        result = compute_kv_bytes_per_token()
        assert result == 327_680, (
            f"期望 327,680 bytes，实际 {result} bytes"
        )

    def test_approximately_320kb(self):
        """测试：单 token 约 320 KB。"""
        result = compute_kv_bytes_per_token()
        kb = result / 1024
        assert 318 <= kb <= 322, (
            f"期望约 320 KB，实际 {kb:.1f} KB"
        )


class TestComputeCacheCapacity:
    """compute_cache_capacity() 的单元测试。"""

    def test_returns_memory_report(self):
        """测试：返回值类型为 MemoryReport。"""
        result = compute_cache_capacity()
        assert isinstance(result, MemoryReport)

    def test_capacity_map_has_expected_keys(self):
        """测试：capacity_by_doc_length 包含预期的文书长度键。"""
        result = compute_cache_capacity()
        expected_keys = {14000, 18000, 20000, 25000, 32000}
        assert set(result.capacity_by_doc_length.keys()) == (
            expected_keys
        )

    def test_longer_docs_have_lower_capacity(self):
        """测试：文书越长，最大缓存文书数越少。"""
        result = compute_cache_capacity()
        caps = result.capacity_by_doc_length
        assert caps[14000] > caps[20000] > caps[32000]

    def test_recommended_limit_below_theoretical_max(self):
        """测试：建议上限低于 18K tokens 文书的理论上限。"""
        result = compute_cache_capacity()
        theoretical = result.capacity_by_doc_length[18000]
        assert result.recommended_cache_limit < theoretical

    def test_recommended_limit_is_positive(self):
        """测试：建议上限为正整数。"""
        result = compute_cache_capacity()
        assert result.recommended_cache_limit > 0

    def test_custom_params_reduce_capacity(self):
        """测试：降低显存参数后缓存容量相应减少。"""
        full = compute_cache_capacity()
        reduced = compute_cache_capacity(
            total_vram_per_card_gb=40.0,  # 降低至 40GB
        )
        assert (
            reduced.recommended_cache_limit
            < full.recommended_cache_limit
        )

    def test_kv_bytes_per_token_matches_standalone(self):
        """
        测试：MemoryReport 中的 kv_bytes_per_token 与
        compute_kv_bytes_per_token() 返回值一致。
        """
        report = compute_cache_capacity()
        assert report.kv_bytes_per_token == (
            compute_kv_bytes_per_token()
        )
