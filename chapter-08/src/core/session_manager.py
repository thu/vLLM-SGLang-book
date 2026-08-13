"""
src/core/session_manager.py
Redis 会话状态管理器。

持久化策略:
- 键格式: session:{session_id}
- TTL: 1800 秒（30 分钟），每次成功处理后重置
- 序列化: Pydantic v2 model_dump_json()，性能优于标准 json.dumps
- 失败策略: save_state 失败只记录日志，不抛异常，不中断对话
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import List, Optional

import redis
from loguru import logger
from pydantic import ValidationError

from .state import DialogueState, TravelPreferences


class SessionManager:
    """
    Redis 会话管理器。
    设计要点:
    - save_state 永远不抛异常（软失败策略）
    - load_state 返回 None 表示会话不存在或已过期
    - TTL 在每次 save_state 时重置，支持无限期活跃会话
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_db: int = 0,
        redis_password: Optional[str] = None,
        default_ttl: int = 1800,
    ) -> None:
        self._redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
        )
        self.default_ttl = default_ttl

        # 启动时验证连接，尽早发现 Redis 不可用
        try:
            self._redis.ping()
            logger.info(
                f"Redis 连接成功 | {redis_host}:{redis_port}/db{redis_db}"
            )
        except redis.ConnectionError as exc:
            logger.critical(f"Redis 连接失败，会话管理不可用: {exc}")
            raise

    # ── 核心 CRUD ────────────────────────────────────────────────────

    def create_session(
        self,
        user_id: str,
        session_id: Optional[str] = None,
        ttl: Optional[int] = None,
    ) -> DialogueState:
        """创建新会话并立即写入 Redis。"""
        sid = session_id or str(uuid.uuid4())
        state = DialogueState(session_id=sid, user_id=user_id)
        self.save_state(state, ttl=ttl)
        logger.info(f"创建会话 | session_id={sid} user_id={user_id}")
        return state

    def save_state(
        self,
        state: DialogueState,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        保存会话状态到 Redis。
        返回 True 表示成功，False 表示失败。
        失败时只记录日志，不抛异常——状态丢失只影响下次重连的恢复能力，
        不影响当前轮次的响应质量。
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        state.updated_at = datetime.now().isoformat()
        key = f"session:{state.session_id}"
        try:
            self._redis.setex(key, effective_ttl, state.to_redis_value())
            return True
        except redis.RedisError as exc:
            logger.error(
                f"会话保存失败 | session_id={state.session_id} | {exc}"
            )
            return False

    def load_state(self, session_id: str) -> Optional[DialogueState]:
        """
        从 Redis 加载会话状态。
        返回 None 表示会话不存在或已过期，由调用方决定是创建新会话还是报错。
        """
        key = f"session:{session_id}"
        try:
            raw = self._redis.get(key)
            if raw is None:
                logger.debug(f"会话不存在或已过期 | session_id={session_id}")
                return None
            state = DialogueState.from_redis_value(raw)
            logger.debug(f"会话加载成功 | session_id={session_id}")
            return state
        except (redis.RedisError, ValidationError, Exception) as exc:
            logger.error(
                f"会话加载失败 | session_id={session_id} | {exc}"
            )
            return None

    def delete_state(self, session_id: str) -> bool:
        """显式删除会话（用户主动退出时调用）。"""
        try:
            deleted = self._redis.delete(f"session:{session_id}")
            if deleted:
                logger.info(f"会话已删除 | session_id={session_id}")
            return bool(deleted)
        except redis.RedisError as exc:
            logger.error(f"会话删除失败 | {exc}")
            return False

    def extend_ttl(
        self,
        session_id: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """
        延长会话 TTL（用于长时间活跃的会话）。
        正常情况下 save_state 会自动重置 TTL，此方法用于只读场景。
        """
        effective_ttl = ttl if ttl is not None else self.default_ttl
        try:
            result = self._redis.expire(
                f"session:{session_id}", effective_ttl
            )
            return bool(result)
        except redis.RedisError as exc:
            logger.error(f"TTL 延长失败 | {exc}")
            return False

    # ── 查询辅助 ─────────────────────────────────────────────────────

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在且未过期。"""
        try:
            return bool(self._redis.exists(f"session:{session_id}"))
        except redis.RedisError:
            return False

    def get_session_stats(self) -> dict:
        """
        扫描统计当前 Redis 中的会话数量和总轮次。
        注意: SCAN 操作在大规模数据下有一定开销，不建议在热路径上调用。
        """
        pattern = "session:*"
        cursor = 0
        total_sessions = 0
        total_turns = 0
        try:
            while True:
                cursor, keys = self._redis.scan(
                    cursor=cursor, match=pattern, count=100
                )
                total_sessions += len(keys)
                for key in keys:
                    raw = self._redis.get(key)
                    if raw:
                        try:
                            data = DialogueState.from_redis_value(raw)
                            total_turns += data.turn_count
                        except Exception:
                            pass
                if cursor == 0:
                    break
        except redis.RedisError as exc:
            logger.error(f"会话统计失败 | {exc}")
        return {
            "total_sessions": total_sessions,
            "total_turns": total_turns,
            "avg_turns_per_session": (
                total_turns / total_sessions if total_sessions else 0
            ),
        }
