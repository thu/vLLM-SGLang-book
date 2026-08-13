#!/usr/bin/env python3
# scripts/verify_constrained_decoding.py
# 验证 SGLang 约束解码功能是否正常工作
# 使用方法：python scripts/verify_constrained_decoding.py
#
# 执行两项测试：
# 1. 最简 Schema 测试：强制输出包含 name 字段的 JSON
# 2. 完整 Resume Schema 测试：用真实简历文本验证端到端流程

import asyncio
import json
import sys
from pathlib import Path

import aiohttp

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.sglang_extractor import SGLANG_URL, MODEL_NAME, extract_resume

# ── 测试用简历文本 ──
SAMPLE_RESUME = """\
张伟
电话：138-0000-1234
邮箱：zhangwei@example.com
地址：北京市朝阳区

教育背景
2015-2019  清华大学  计算机科学与技术  本科

工作经历
2019年7月 - 2021年3月  字节跳动  后端开发工程师
- 负责推荐系统后端开发，使用 Python 和 Go
- 优化推荐算法，CTR 提升 15%

2021年4月至今  阿里巴巴  高级开发工程师
- 负责电商平台核心交易系统
- 设计分布式事务方案

技能
编程语言：Python、Java、Go
框架：Django、Spring Boot、Gin
数据库：MySQL、Redis、MongoDB
工具：Docker、Kubernetes、Git
"""


async def test_minimal_schema() -> bool:
    """
    测试1：最简 Schema 约束解码。
    强制模型输出只包含 name 字段的 JSON。
    """
    print("\n[测试1] 最简 Schema 约束解码")

    minimal_schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "人名"}
        },
        "required": ["name"],
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": "请输出一个中文人名"}
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name":   "minimal_test",
                "schema": minimal_schema,
                "strict": True,
            },
        },
        "temperature": 0.0,
        "max_tokens":  50,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(SGLANG_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()

        content = data["choices"][0]["message"]["content"]
        parsed  = json.loads(content)   # 约束解码保证此行不抛出

        assert "name" in parsed, f"输出缺少 name 字段：{parsed}"
        assert isinstance(parsed["name"], str), f"name 不是字符串：{parsed}"

        print(f"  ✓ 输出：{parsed}")
        print(f"  ✓ JSON 格式合法，字段类型正确")
        return True

    except Exception as e:
        print(f"  ✗ 测试失败：{e}")
        return False


async def test_full_resume_schema() -> bool:
    """
    测试2：完整 Resume Schema 端到端测试。
    使用真实简历文本验证约束解码 + Pydantic 验证全流程。
    """
    print("\n[测试2] 完整 Resume Schema 端到端提取")

    try:
        async with aiohttp.ClientSession() as session:
            resume = await extract_resume(SAMPLE_RESUME, session)

        # 验证关键字段
        assert resume.basic_info.name == "张伟", \
            f"姓名提取错误：{resume.basic_info.name}"

        assert len(resume.work_experience) >= 1, \
            "工作经历为空"

        # 验证日期规范化（系统提示要求 YYYY-MM 格式）
        first_job = resume.work_experience[0]
        assert first_job.start_date is not None, "第一份工作缺少 start_date"
        assert "-" in first_job.start_date or len(first_job.start_date) == 4, \
            f"日期格式不符：{first_job.start_date}"

        # 验证学历枚举
        if resume.education:
            degree_val = resume.education[0].degree.value
            assert degree_val in ("专科", "本科", "硕士", "博士"), \
                f"学历枚举值不合法：{degree_val}"

        print(f"  ✓ 姓名：{resume.basic_info.name}")
        print(f"  ✓ 手机：{resume.basic_info.phone}")
        print(f"  ✓ 邮箱：{resume.basic_info.email}")
        print(f"  ✓ 教育经历：{len(resume.education)} 段")
        print(f"  ✓ 工作经历：{len(resume.work_experience)} 段")
        if resume.skills:
            pl = resume.skills.programming_languages
            print(f"  ✓ 编程语言：{pl}")

        print(f"\n  完整提取结果：")
        print(resume.model_dump_json(indent=2, exclude_none=True))
        return True

    except Exception as e:
        print(f"  ✗ 测试失败：{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    print("=" * 60)
    print("约束解码功能验证")
    print("=" * 60)
    print(f"SGLang 服务地址：{SGLANG_URL}")

    results = []
    results.append(await test_minimal_schema())
    results.append(await test_full_resume_schema())

    print("\n" + "=" * 60)
    passed = sum(results)
    total  = len(results)
    print(f"测试结果：{passed}/{total} 通过")
    if passed == total:
        print("约束解码功能正常，可以开始批量处理实验。")
    else:
        print("存在失败项，请检查 SGLang 服务状态和模型路径。")
    print("=" * 60)

    return passed == total


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
