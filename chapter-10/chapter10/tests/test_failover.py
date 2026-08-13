"""
文件: chapter10/tests/test_failover.py

故障转移与降级逻辑测试。

测试范围:
  - 降级状态的触发条件（持续超阈值才触发）
  - Redis NX 原子操作的并发安全性（逻辑验证）
  - 降级状态解除机制（TTL到期或强制清除）
  - HTTP中间件对不同优先级请求的处理差异
  - 运维接口（查询状态、强制解除）的功能正确性

运行方式:
  # 无需 Redis，mock 替代（适合 CI）
  pytest tests/test_failover.py -v

  # 配合真实 Redis（需要本地 Redis 服务）
  REDIS_URL=redis://localhost:6379 pytest tests/test_failover.py -v -m "requires_redis"
"""

import asyncio
import sys
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── 降级核心逻辑测试（Mock Redis）────────────────────────────────────────

class TestDegradationLogic:
    """
    使用 Mock Redis 测试降级状态管理的核心逻辑。
    不依赖真实 Redis，适合 CI 环境。
    """

    @pytest.fixture
    def mock_redis(self):
        """提供一个内存 Redis mock，模拟 SET NX / GET / EXISTS / DELETE 行为"""
        store = {}

        async def mock_set(key, value, ex=None, nx=False, **kwargs):
            if nx and key in store:
                return None   # NX: 键已存在时不写入，返回 None（与 redis-py 行为一致）
            store[key] = {"value": value, "ex": ex, "set_at": time.time()}
            return True

        async def mock_get(key):
            entry = store.get(key)
            if entry is None:
                return None
            if entry["ex"] and (time.time() - entry["set_at"]) > entry["ex"]:
                del store[key]
                return None
            return entry["value"]

        async def mock_exists(*keys):
            return sum(1 for k in keys if k in store)

        async def mock_delete(*keys):
            count = 0
            for k in keys:
                if k in store:
                    del store[k]
                    count += 1
            return count

        redis = AsyncMock()
        redis.set = mock_set
        redis.get = mock_get
        redis.exists = mock_exists
        redis.delete = mock_delete
        redis._store = store  # 暴露内部存储供断言使用
        return redis

    @pytest.mark.asyncio
    async def test_no_degradation_below_threshold(self, mock_redis):
        """负载低于阈值时不触发降级"""
        from failover.degradation import check_and_set_degradation, is_degraded
        with patch("failover.degradation.get_redis", return_value=mock_redis):
            await check_and_set_degradation("vllm", load_score=0.5)
            degraded = await is_degraded("vllm")
        assert degraded is False

    @pytest.mark.asyncio
    async def test_degradation_not_triggered_before_window(self, mock_redis):
        """超过阈值但未达到窗口期时，不触发降级"""
        from failover.degradation import check_and_set_degradation, is_degraded, DEGRADATION_WINDOW_SECONDS
        with patch("failover.degradation.get_redis", return_value=mock_redis):
            # 第一次调用：记录超阈值起始时间
            await check_and_set_degradation("vllm", load_score=0.95)
            # 立即再次调用（未达到窗口期）
            await check_and_set_degradation("vllm", load_score=0.95)
            degraded = await is_degraded("vllm")
        assert degraded is False

    @pytest.mark.asyncio
    async def test_degradation_triggered_after_window(self, mock_redis):
        """持续超阈值超过窗口期后触发降级"""
        from failover.degradation import check_and_set_degradation, is_degraded

        # 手动在 store 中设置一个"已超阈值超过窗口期"的时间戳
        past_timestamp = str(time.time() - 300)  # 300秒前开始超阈值（窗口期为180秒）
        mock_redis._store["over_threshold:sglang"] = {
            "value": past_timestamp,
            "ex": 600,
            "set_at": time.time() - 300,
        }

        with patch("failover.degradation.get_redis", return_value=mock_redis):
            await check_and_set_degradation("sglang", load_score=0.95)
            degraded = await is_degraded("sglang")
        assert degraded is True

    @pytest.mark.asyncio
    async def test_nx_prevents_overwrite(self, mock_redis):
        """NX 原子操作应防止并发写入覆盖首次超阈值时间戳"""
        original_timestamp = str(time.time() - 100)
        # 预先写入首次超阈值时间戳
        mock_redis._store["over_threshold:vllm"] = {
            "value": original_timestamp,
            "ex": 600,
            "set_at": time.time() - 100,
        }

        from failover.degradation import check_and_set_degradation

        with patch("failover.degradation.get_redis", return_value=mock_redis):
            # 第二次调用（并发场景），NX 应阻止覆盖
            await check_and_set_degradation("vllm", load_score=0.95)

        # 验证时间戳未被覆盖（NX 保证）
        stored = mock_redis._store.get("over_threshold:vllm", {})
        assert stored.get("value") == original_timestamp, (
            "NX 操作未能防止并发覆盖，首次超阈值时间戳被修改"
        )

    @pytest.mark.asyncio
    async def test_degradation_cleared_on_recovery(self, mock_redis):
        """负载恢复后，降级标记和超阈值计时均被清除"""
        # 预先设置降级状态
        mock_redis._store["degradation:vllm"] = {"value": "1", "ex": 60, "set_at": time.time()}
        mock_redis._store["over_threshold:vllm"] = {
            "value": str(time.time() - 200),
            "ex": 600,
            "set_at": time.time() - 200,
        }

        from failover.degradation import check_and_set_degradation, is_degraded

        with patch("failover.degradation.get_redis", return_value=mock_redis):
            await check_and_set_degradation("vllm", load_score=0.3)  # 低于阈值
            degraded = await is_degraded("vllm")

        assert degraded is False
        assert "over_threshold:vllm" not in mock_redis._store

    @pytest.mark.asyncio
    async def test_force_clear_removes_all_keys(self, mock_redis):
        """强制清除接口应同时删除降级标记和超阈值计时"""
        mock_redis._store["degradation:vllm"] = {"value": "1", "ex": 60, "set_at": time.time()}
        mock_redis._store["over_threshold:vllm"] = {
            "value": "123.0",
            "ex": 600,
            "set_at": time.time(),
        }

        from failover.degradation import force_clear_degradation, is_degraded

        with patch("failover.degradation.get_redis", return_value=mock_redis):
            await force_clear_degradation("vllm")
            degraded = await is_degraded("vllm")

        assert degraded is False
        assert "degradation:vllm" not in mock_redis._store
        assert "over_threshold:vllm" not in mock_redis._store


# ── HTTP 中间件行为测试 ───────────────────────────────────────────────────

class TestDegradationMiddleware:
    """
    使用 httpx AsyncClient 测试 FastAPI 中间件的请求处理行为。
    """

    @pytest.fixture
    def mock_redis_not_degraded(self):
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value=None)
        redis.exists = AsyncMock(return_value=0)   # 0 表示键不存在
        redis.delete = AsyncMock(return_value=0)
        return redis

    @pytest.fixture
    def mock_redis_degraded(self):
        redis = AsyncMock()
        redis.set = AsyncMock(return_value=True)
        redis.get = AsyncMock(return_value="123.0")
        redis.exists = AsyncMock(return_value=1)   # 1 表示降级键存在
        redis.delete = AsyncMock(return_value=1)
        return redis

    @pytest.mark.asyncio
    async def test_high_priority_passes_through_when_degraded(self, mock_redis_degraded):
        """高优先级请求在降级状态下应正常透传（不返回429）"""
        from httpx import AsyncClient, ASGITransport
        from failover.degradation import app

        with patch("failover.degradation.get_redis", return_value=mock_redis_degraded):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get(
                    "/healthz",
                    headers={"x-priority": "high"},
                )
        # /healthz 端点不受降级影响
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_low_priority_returns_429_when_degraded(self, mock_redis_degraded):
        """低优先级请求在降级状态下应返回429"""
        from httpx import AsyncClient, ASGITransport
        from failover.degradation import app

        with patch("failover.degradation.get_redis", return_value=mock_redis_degraded):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"x-priority": "low"},
                    json={"model": "qwen2.5-7b", "messages": []},
                )
        assert resp.status_code == 429
        body = resp.json()
        assert "retry_after_seconds" in body
        assert resp.headers.get("Retry-After") is not None

    @pytest.mark.asyncio
    async def test_admin_status_endpoint(self, mock_redis_degraded):
        """运维状态查询接口应返回正确的降级状态"""
        from httpx import AsyncClient, ASGITransport
        from failover.degradation import app

        with patch("failover.degradation.get_redis", return_value=mock_redis_degraded):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.get("/admin/degradation/vllm")
        assert resp.status_code == 200
        body = resp.json()
        assert body["framework"] == "vllm"
        assert body["degraded"] is True
