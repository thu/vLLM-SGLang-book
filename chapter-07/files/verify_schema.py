#!/usr/bin/env python3
# scripts/verify_schema.py
# 验证 Pydantic 模型生成的 JSON Schema 是否符合预期
# 使用方法：python scripts/verify_schema.py
#
# 检查内容：
# 1. DegreeType 枚举被正确展开为四个值
# 2. basic_info 是必填字段
# 3. 日期字段的 pattern 约束存在
# 4. 打印完整 Schema 供人工确认

import json
import sys
from pathlib import Path

# 将项目根目录加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.resume import Resume, DegreeType


def verify_schema() -> bool:
    """验证 Resume JSON Schema 的关键属性"""
    print("=" * 60)
    print("Resume JSON Schema 验证")
    print("=" * 60)

    schema = Resume.model_json_schema()
    all_passed = True

    # ── 检查 1：basic_info 是必填字段 ──
    required = schema.get("required", [])
    if "basic_info" in required:
        print("  ✓ basic_info 是必填字段")
    else:
        print("  ✗ basic_info 不在 required 列表中")
        all_passed = False

    # ── 检查 2：DegreeType 枚举正确展开 ──
    defs = schema.get("$defs", {})
    if "DegreeType" in defs:
        degree_enum = defs["DegreeType"].get("enum", [])
        expected = {e.value for e in DegreeType}
        if set(degree_enum) == expected:
            print(f"  ✓ DegreeType 枚举正确：{degree_enum}")
        else:
            print(f"  ✗ DegreeType 枚举不符，期望 {expected}，实际 {degree_enum}")
            all_passed = False
    else:
        print("  ✗ $defs 中未找到 DegreeType")
        all_passed = False

    # ── 检查 3：Education 的 start_date 存在 pattern 约束 ──
    if "Education" in defs:
        edu_props = defs["Education"].get("properties", {})
        start_date_prop = edu_props.get("start_date", {})
        # Pydantic v2 可能将 pattern 包裹在 anyOf 中（因为字段是 Optional）
        has_pattern = (
            "pattern" in start_date_prop
            or any(
                "pattern" in item
                for item in start_date_prop.get("anyOf", [])
            )
        )
        if has_pattern:
            print("  ✓ Education.start_date 存在 pattern 约束")
        else:
            print("  ✗ Education.start_date 缺少 pattern 约束")
            all_passed = False
    else:
        print("  ✗ $defs 中未找到 Education")
        all_passed = False

    # ── 检查 4：WorkExperience 的 responsibilities 是数组 ──
    if "WorkExperience" in defs:
        work_props = defs["WorkExperience"].get("properties", {})
        resp_prop  = work_props.get("responsibilities", {})
        if resp_prop.get("type") == "array":
            print("  ✓ WorkExperience.responsibilities 是 array 类型")
        else:
            print("  ✗ WorkExperience.responsibilities 类型不是 array")
            all_passed = False
    else:
        print("  ✗ $defs 中未找到 WorkExperience")
        all_passed = False

    # ── 检查 5：Skills 字段是 Optional（不在 required 中）──
    top_required = schema.get("required", [])
    if "skills" not in top_required:
        print("  ✓ skills 是可选字段（不在顶层 required 中）")
    else:
        print("  ✗ skills 被设为必填，应为 Optional")
        all_passed = False

    # ── 打印完整 Schema ──
    print("\n── 完整 JSON Schema ──")
    print(json.dumps(schema, ensure_ascii=False, indent=2))

    print("\n" + "=" * 60)
    if all_passed:
        print("所有检查通过，Schema 可以传入 SGLang 约束解码。")
    else:
        print("存在检查失败项，请修正模型定义后重新验证。")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    ok = verify_schema()
    sys.exit(0 if ok else 1)
