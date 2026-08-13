# config 包初始化
from config.model_config import MODEL_CONFIG, DEPLOY_CONFIG
from config.prompt_config import (
    UNIFIED_SYSTEM_PROMPT,
    STATIC_TASK_TEMPLATES,
    TASK_TYPES,
    build_stable_prefix,
)

__all__ = [
    "MODEL_CONFIG",
    "DEPLOY_CONFIG",
    "UNIFIED_SYSTEM_PROMPT",
    "STATIC_TASK_TEMPLATES",
    "TASK_TYPES",
    "build_stable_prefix",
]
