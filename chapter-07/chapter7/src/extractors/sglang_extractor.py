# src/extractors/sglang_extractor.py
# SGLang 约束解码提取管道
#
# 核心设计决策：
# 1. 通过 /v1/chat/completions + response_format 触发约束解码，
#    SGLang 内部自动处理，客户端对此透明
# 2. temperature=0.0：确定性提取，相同输入总产生相同输出
# 3. 约束解码保证格式正确，Pydantic 负责业务规则验证，两层互补
# 4. 重试逻辑不在此层，由调用方（batch_processor）决定策略

from __future__ import annotations

import json as json_module
from typing import Any, Optional

import aiohttp
from loguru import logger
from pydantic import ValidationError

from src.models.resume import Resume

# ──────────────────────────────────────────────
# 常量配置
# ──────────────────────────────────────────────

SGLANG_URL    = "http://127.0.0.1:30000/v1/chat/completions"
MODEL_NAME    = "Llama-3.1-8B-Instruct"
RESUME_SCHEMA = Resume.model_json_schema()

# 粗略字符数上限，约对应 3000 token，为输出预留足够空间
MAX_INPUT_CHARS = 12_000

SYSTEM_PROMPT = """\
你是一个专业的简历信息提取助手。从用户提供的简历文本中提取结构化信息。

提取规范：
1. 日期格式：统一转换为 YYYY 或 YYYY-MM 格式。
   "2019年7月" → "2019-07"，"2019年" → "2019"，当前在职 → "present"
2. 技能分类：
   - 编程语言：Python、Java、Go、C++ 等
   - 框架：Django、Spring Boot、React、Gin 等
   - 数据库：MySQL、Redis、MongoDB、PostgreSQL 等
   - 工具：Docker、Kubernetes、Git、Jenkins 等
   边界模糊时优先参考上述示例
3. 职责描述：每条职责单独列为一个字符串，保留原文中的数字指标（如"CTR提升15%"）
4. 缺失字段：简历中未提及的字段输出 null，不要凭空填写任何内容
5. 学历：从"专科"、"本科"、"硕士"、"博士"中选择最匹配的值\
"""


# ──────────────────────────────────────────────
# 核心提取函数
# ──────────────────────────────────────────────

async def extract_resume(
    text: str,
    session: aiohttp.ClientSession,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    system_prompt: str = SYSTEM_PROMPT,
) -> Resume:
    """
    从简历文本提取结构化信息，返回 Resume 对象。

    约束解码由 response_format 触发，SGLang 内部自动处理。
    返回的 raw_json 必然是符合 Resume Schema 的合法 JSON，
    model_validate_json 的主要作用是执行 Pydantic 业务规则验证
    （手机号格式、日期合法性等），而非再次进行 JSON 格式检查。

    Args:
        text:          简历纯文本
        session:       aiohttp 客户端会话（由调用方管理生命周期）
        temperature:   解码温度，0.0 为贪心解码，保证确定性
        max_tokens:    最大输出 token 数
        system_prompt: 系统提示，支持注入 Few-shot 示例

    Returns:
        验证通过的 Resume 对象

    Raises:
        aiohttp.ClientError:  网络层面的失败（连接超时、服务不可达等）
        pydantic.ValidationError: 业务规则验证失败（手机号格式不合法等）
        json.JSONDecodeError: JSON 被截断（罕见，通常由 max_tokens 过小导致）
    """
    payload = {
        "model":    MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"请提取以下简历的信息：\n\n{text}"},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name":   "resume_extraction",
                "schema": RESUME_SCHEMA,
                "strict": True,
            },
        },
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    async with session.post(SGLANG_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()

    raw_json: str = data["choices"][0]["message"]["content"]

    # 先验证 JSON 语法（检测截断），再执行 Pydantic 业务验证
    json_module.loads(raw_json)   # 若截断则抛出 JSONDecodeError
    return Resume.model_validate_json(raw_json)


async def extract_with_fallback(
    text: str,
    session: aiohttp.ClientSession,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, Any]:
    """
    带降级处理的提取函数。
    网络失败向上抛出（由 batch_processor 决定重试策略）。
    JSON 截断和业务验证失败在此捕获，保留原始内容供审核。

    Returns:
        成功：{"status": "ok", "resume": Resume}
        截断：{"status": "truncated", "raw_json": str}
        验证失败：{"status": "validation_error", "errors": list, "raw": dict}
    """
    # 超长文本截断
    truncated = False
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
        truncated = True
        logger.warning(f"文本超长，已截断至 {MAX_INPUT_CHARS} 字符")

    raw_json: Optional[str] = None

    try:
        # 截断文本时增大 max_tokens，尽量获得完整输出
        max_tokens = 4096 if truncated else 2048
        resume = await extract_resume(
            text, session,
            max_tokens=max_tokens,
            system_prompt=system_prompt
        )
        return {"status": "ok", "resume": resume}

    except json_module.JSONDecodeError:
        # JSON 被截断，记录并返回
        logger.warning("检测到 JSON 截断，请检查 max_tokens 设置")
        return {"status": "truncated", "raw_json": raw_json or ""}

    except ValidationError as e:
        # 业务规则验证失败（手机号格式不合法等）
        # 通常反映简历原始数据不规范，保留供人工审核
        logger.warning(
            f"字段验证失败（{e.error_count()} 处）："
            + "；".join(
                f"{'.'.join(str(x) for x in err['loc'])}：{err['msg']}"
                for err in e.errors()
            )
        )
        try:
            raw_dict = json_module.loads(raw_json) if raw_json else {}
        except Exception:
            raw_dict = {}
        return {
            "status": "validation_error",
            "errors": e.errors(),
            "raw":    raw_dict,
        }


# ──────────────────────────────────────────────
# 通用提取函数（支持任意 Pydantic 模型）
# 用于合同、医疗报告、金融舆情等场景延伸
# ──────────────────────────────────────────────

async def extract_structured(
    text: str,
    session: aiohttp.ClientSession,
    pydantic_model,
    system_prompt: str,
    schema_name: str = "extraction",
    temperature: float = 0.0,
    max_tokens: int = 2048,
):
    """
    通用结构化提取函数，支持任意 Pydantic v2 模型。

    场景延伸（合同、医疗报告、金融舆情）均使用此函数，
    只需传入对应的 Pydantic 模型和系统提示即可。

    Args:
        text:           输入文本
        session:        aiohttp 客户端会话
        pydantic_model: Pydantic v2 模型类
        system_prompt:  系统提示（包含领域判断标准）
        schema_name:    JSON Schema 的名称标识
        temperature:    解码温度
        max_tokens:     最大输出 token 数

    Returns:
        pydantic_model 实例

    Raises:
        同 extract_resume
    """
    schema = pydantic_model.model_json_schema()

    payload = {
        "model":    MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": text},
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name":   schema_name,
                "schema": schema,
                "strict": True,
            },
        },
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }

    async with session.post(SGLANG_URL, json=payload) as resp:
        resp.raise_for_status()
        data = await resp.json()

    raw_json: str = data["choices"][0]["message"]["content"]
    json_module.loads(raw_json)   # 截断检测
    return pydantic_model.model_validate_json(raw_json)
