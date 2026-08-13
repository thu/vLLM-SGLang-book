#!/usr/bin/env python3
# ch06/extensions/routing/canary.py
# 基于用户 ID 哈希的稳定灰度路由
#
# 稳定性：同一个用户始终被路由到同一个版本，
#         避免同一用户在 A/B 版本之间反复切换的体验问题。
#
# 用法：集成到 ch06/gateway/main.py 的路由逻辑中

import hashlib
from dataclasses import dataclass


@dataclass
class ModelConfig:
    name: str           # vLLM --served-model-name 对应的名称
    description: str


# 版本注册表：key 为逻辑版本名，value 为实际模型配置
MODEL_REGISTRY = {
    "stable": ModelConfig(
        name="Qwen2.5-7B-Instruct-AWQ",
        description="稳定版本（7B AWQ）",
    ),
    "canary": ModelConfig(
        name="Qwen2.5-14B-Instruct-AWQ",
        description="金丝雀版本（14B AWQ，待验证）",
    ),
}


def get_model_for_user(
    user_id: str,
    canary_ratio: float = 0.05,
) -> ModelConfig:
    """
    根据用户 ID 哈希决定路由目标版本。

    Args:
        user_id:      用户唯一标识（通常来自 JWT token 中的 sub 字段）
        canary_ratio: 路由到金丝雀版本的流量比例，范围 [0, 1]

    Returns:
        ModelConfig，包含应使用的模型名称

    Example:
        model = get_model_for_user("user_12345", canary_ratio=0.10)
        # 10% 的用户路由到金丝雀版本
    """
    if not 0.0 <= canary_ratio <= 1.0:
        raise ValueError(f"canary_ratio 必须在 [0, 1] 范围内，实际值: {canary_ratio}")

    # MD5 哈希取模：输出范围 [0, 99]
    # 使用 MD5 而非随机数，确保同一 user_id 每次得到相同结果（稳定路由）
    hash_val = int(hashlib.md5(user_id.encode()).hexdigest(), 16)
    bucket = hash_val % 100

    if bucket < int(canary_ratio * 100):
        return MODEL_REGISTRY["canary"]
    return MODEL_REGISTRY["stable"]


def get_canary_ratio_from_env() -> float:
    """
    从环境变量读取当前灰度比例，支持不重启服务动态调整。
    环境变量：CANARY_RATIO，默认 0（关闭灰度）

    灰度比例建议的推进节奏：0 → 0.05 → 0.20 → 0.50 → 1.0
    每个阶段等待足够的样本量（建议每阶段至少 10000 次请求），
    确认两个版本的 P99 延迟和错误率无显著差异后再推进。
    """
    import os
    try:
        return float(os.environ.get("CANARY_RATIO", "0"))
    except ValueError:
        return 0.0
