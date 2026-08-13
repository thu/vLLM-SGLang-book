#!/usr/bin/env python3
# scripts/run_batch.py
# 批量简历处理完整入口
# 使用方法：
#   # 处理目录下所有文档（PDF/DOCX/TXT）
#   python scripts/run_batch.py --input data/raw/pdf/ --output results/extracted/
#
#   # 使用 Few-shot 示例（日期规范化 + 在职状态）
#   python scripts/run_batch.py \
#       --input data/raw/ \
#       --output results/extracted/ \
#       --few-shot date_normalization current_job \
#       --concurrency 20

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractors.batch_processor import process_directory
from src.prompts.few_shot_selector import get_prompt_with_examples, list_examples


def print_examples_help():
    """打印可用的 Few-shot 示例列表"""
    print("\n可用的 Few-shot 示例：")
    for ex in list_examples():
        print(f"  {ex['name']:<25} {ex['description']}")
        print(f"  {'':25} 标签：{ex['tags']}")
    print()


async def main():
    parser = argparse.ArgumentParser(
        description="批量简历结构化提取",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 基础用法（无 Few-shot）
  python scripts/run_batch.py --input data/raw/ --output results/

  # 使用日期规范化和在职状态示例
  python scripts/run_batch.py --input data/raw/ --output results/ \\
      --few-shot date_normalization current_job

  # 调整并发数
  python scripts/run_batch.py --input data/raw/ --output results/ \\
      --concurrency 30

  # 查看可用的 Few-shot 示例
  python scripts/run_batch.py --list-examples
        """
    )
    parser.add_argument("--input",       type=str, help="输入目录（PDF/DOCX/TXT）")
    parser.add_argument("--output",      type=str, help="输出目录")
    parser.add_argument(
        "--few-shot",    type=str, nargs="*", default=None,
        metavar="EXAMPLE_NAME",
        help="要使用的 Few-shot 示例名称（不指定则不使用示例）"
    )
    parser.add_argument(
        "--all-examples", action="store_true",
        help="使用全部内置 Few-shot 示例"
    )
    parser.add_argument(
        "--concurrency", type=int, default=20,
        help="最大并发请求数（默认 20）"
    )
    parser.add_argument(
        "--extensions",  type=str, nargs="+",
        default=[".pdf", ".docx", ".txt"],
        help="要处理的文件扩展名（默认 .pdf .docx .txt）"
    )
    parser.add_argument(
        "--list-examples", action="store_true",
        help="列出所有可用的 Few-shot 示例"
    )
    args = parser.parse_args()

    # 只查看示例列表
    if args.list_examples:
        print_examples_help()
        return

    if not args.input or not args.output:
        parser.print_help()
        sys.exit(1)

    # 构建系统提示
    if args.all_examples:
        system_prompt = get_prompt_with_examples(None)
        print("使用全部内置 Few-shot 示例")
    elif args.few_shot:
        system_prompt = get_prompt_with_examples(args.few_shot)
        print(f"使用 Few-shot 示例：{args.few_shot}")
    else:
        from src.extractors.sglang_extractor import SYSTEM_PROMPT
        system_prompt = SYSTEM_PROMPT
        print("不使用 Few-shot 示例")

    print("=" * 60)
    print("批量简历结构化提取")
    print("=" * 60)
    print(f"  输入目录：{args.input}")
    print(f"  输出目录：{args.output}")
    print(f"  最大并发：{args.concurrency}")
    print(f"  文件格式：{args.extensions}")

    t0    = time.perf_counter()
    stats = await process_directory(
        input_dir      = args.input,
        output_dir     = args.output,
        max_concurrent = args.concurrency,
        extensions     = args.extensions,
    )
    elapsed = time.perf_counter() - t0

    print("\n" + "=" * 60)
    print("处理完成")
    print("=" * 60)
    print(f"  总耗时：{elapsed:.1f}s")
    print(f"  成功：{stats.get('success', 0)}")
    print(f"  验证失败（需人工审核）：{stats.get('validation_error', 0)}")
    print(f"  截断：{stats.get('truncated', 0)}")
    print(f"  网络失败：{stats.get('failed', 0)}")
    print(f"  解析失败：{stats.get('parse_failed', 0)}")

    total_ok = stats.get("success", 0)
    total    = sum(stats.get(k, 0) for k in ("success", "validation_error", "truncated", "failed"))
    if total > 0:
        success_rate = total_ok / total * 100
        per_hour     = total_ok / elapsed * 3600 if elapsed > 0 else 0
        print(f"\n  成功率：{success_rate:.1f}%")
        print(f"  吞吐量：{per_hour:.0f} 份/小时（实测）")

    # 保存统计信息
    stats_path = Path(args.output) / "_batch_stats.json"
    stats["elapsed_s"]     = round(elapsed, 2)
    stats["input_dir"]     = args.input
    stats["few_shot_used"] = args.few_shot or []
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(f"\n统计信息已保存：{stats_path}")


if __name__ == "__main__":
    asyncio.run(main())
