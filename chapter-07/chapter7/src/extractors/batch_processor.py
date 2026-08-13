# src/extractors/batch_processor.py
# 异步并发批量处理器
#
# 设计要点：
# 1. asyncio + aiohttp：单线程内维持数十个并发 HTTP 连接，
#    GPU 计算时 CPU 同步准备下一批请求，避免 GPU 空闲
# 2. Semaphore 控制并发上限，防止请求积压导致 P95 延迟上升
# 3. 网络失败重试（指数退避），业务验证失败不重试（直接存档审核）
# 4. 超长文本在进入请求前截断，避免 max_tokens 截断导致 JSONDecodeError

from __future__ import annotations

import asyncio
import json as json_module
from pathlib import Path
from typing import Optional

import aiofiles
import aiohttp
from loguru import logger
from tqdm.asyncio import tqdm as atqdm

from src.extractors.sglang_extractor import (
    MAX_INPUT_CHARS,
    SYSTEM_PROMPT,
    extract_with_fallback,
)

# 网络失败最大重试次数
MAX_RETRIES = 3


async def process_batch(
    texts: list[str],
    output_dir: str | Path,
    max_concurrent: int = 20,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, int]:
    """
    异步并发处理一批简历文本，结果写入 output_dir。

    并发数建议（以实测为准，详见 scripts/benchmark_throughput.py）：
    - A100 80GB + Llama-3.1-8B：20~30
    - A100 40GB + Llama-3.1-8B：10~20

    输出文件命名：resume_XXXX.json（成功）或 resume_XXXX_error.json（失败/审核）

    Args:
        texts:          简历文本列表
        output_dir:     结果输出目录
        max_concurrent: 最大并发请求数
        system_prompt:  系统提示，支持传入注入了 Few-shot 的版本

    Returns:
        统计字典：{"success": N, "failed": N, "validation_error": N, "truncated": N}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    semaphore = asyncio.Semaphore(max_concurrent)
    stats = {"success": 0, "failed": 0, "validation_error": 0, "truncated": 0}

    async def process_one(idx: int, text: str):
        """处理单份简历，含超长截断、重试、结果写入"""
        # 超长文本截断
        if len(text) > MAX_INPUT_CHARS:
            text = text[:MAX_INPUT_CHARS]
            logger.warning(f"[{idx:04d}] 文本超长，已截断至 {MAX_INPUT_CHARS} 字符")

        last_error: Optional[Exception] = None

        for attempt in range(MAX_RETRIES):
            try:
                async with semaphore:
                    result = await extract_with_fallback(
                        text, session, system_prompt=system_prompt
                    )

                # 根据状态决定写入内容和文件名
                if result["status"] == "ok":
                    out_path = output_dir / f"resume_{idx:04d}.json"
                    content = result["resume"].model_dump_json(
                        indent=2, exclude_none=True
                    )
                    stats["success"] += 1

                elif result["status"] == "validation_error":
                    out_path = output_dir / f"resume_{idx:04d}_validation_error.json"
                    content = json_module.dumps(
                        {
                            "status": "validation_error",
                            "errors": result["errors"],
                            "raw":    result["raw"],
                        },
                        ensure_ascii=False,
                        indent=2,
                    )
                    stats["validation_error"] += 1
                    # 验证失败不重试
                    async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                        await f.write(content)
                    return

                else:  # truncated
                    out_path = output_dir / f"resume_{idx:04d}_truncated.json"
                    content = json_module.dumps(
                        {"status": "truncated", "raw_json": result.get("raw_json", "")},
                        ensure_ascii=False,
                        indent=2,
                    )
                    stats["truncated"] += 1

                async with aiofiles.open(out_path, "w", encoding="utf-8") as f:
                    await f.write(content)
                return

            except aiohttp.ClientError as e:
                # 网络失败：指数退避重试
                last_error = e
                wait_secs = 2 ** attempt  # 1s → 2s → 4s
                logger.warning(
                    f"[{idx:04d}] 第{attempt + 1}次请求失败，{wait_secs}s 后重试：{e}"
                )
                await asyncio.sleep(wait_secs)

            except Exception as e:
                # 其他非预期错误，直接跳过不重试
                logger.error(f"[{idx:04d}] 非预期错误：{type(e).__name__}: {e}")
                stats["failed"] += 1
                return

        # 超过最大重试次数
        logger.error(f"[{idx:04d}] 达到最大重试次数（{MAX_RETRIES}），最后错误：{last_error}")
        stats["failed"] += 1

    async with aiohttp.ClientSession() as session:
        tasks = [process_one(i, t) for i, t in enumerate(texts)]
        await atqdm.gather(*tasks, desc="处理简历", total=len(tasks))

    return stats


async def process_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    max_concurrent: int = 20,
    extensions: list[str] = None,
) -> dict[str, int]:
    """
    解析目录下所有文档，然后批量提取。
    集成文档解析层和提取层的完整流程。

    Args:
        input_dir:      包含 PDF/DOCX/TXT 文件的目录
        output_dir:     结果输出目录
        max_concurrent: 最大并发请求数
        extensions:     要处理的扩展名，默认 ['.pdf', '.docx', '.txt']

    Returns:
        同 process_batch 的统计字典，额外包含 parse_failed 计数
    """
    from src.parsers.base_parser import DocumentParser

    if extensions is None:
        extensions = [".pdf", ".docx", ".txt"]

    parser = DocumentParser()
    input_dir = Path(input_dir)

    logger.info(f"扫描目录：{input_dir}")
    parse_results = parser.parse_directory(input_dir, extensions)

    texts = []
    filenames = []
    parse_failed = 0

    for filename, text in parse_results.items():
        if text is None:
            parse_failed += 1
            logger.warning(f"解析失败，跳过：{filename}")
        else:
            texts.append(text)
            filenames.append(filename)

    logger.info(
        f"文档解析完成：{len(texts)} 成功，{parse_failed} 失败，开始提取..."
    )

    stats = await process_batch(texts, output_dir, max_concurrent)
    stats["parse_failed"] = parse_failed
    return stats
