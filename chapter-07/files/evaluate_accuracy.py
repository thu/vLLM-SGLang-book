#!/usr/bin/env python3
# scripts/evaluate_accuracy.py
# 准确率批量评估入口脚本
# 使用方法：
#   python scripts/evaluate_accuracy.py \
#       --predictions results/extracted/ \
#       --ground-truth data/ground_truth/ \
#       --output results/accuracy_report.json

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluators.accuracy_evaluator import (
    evaluate_batch,
    print_report,
    BatchEvalReport,
)


def save_report(report: BatchEvalReport, output_path: Path) -> None:
    """将评估报告保存为 JSON"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_dict = {
        "n_total":       report.n_total,
        "n_evaluated":   report.n_evaluated,
        "overall_avg":   report.overall_avg,
        "per_field_avg": report.per_field_avg,
        "worst_samples": report.worst_samples,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"\n评估报告已保存：{output_path}")


def main():
    parser = argparse.ArgumentParser(description="简历提取准确率评估")
    parser.add_argument(
        "--predictions",   required=True,
        help="模型提取结果目录（包含 resume_XXXX.json 文件）"
    )
    parser.add_argument(
        "--ground-truth",  required=True,
        help="人工标注目录（同名 JSON 文件）"
    )
    parser.add_argument(
        "--output",
        default="results/accuracy_report.json",
        help="评估报告保存路径（默认 results/accuracy_report.json）"
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions)
    gt_dir   = Path(args.ground_truth)

    if not pred_dir.exists():
        print(f"错误：预测结果目录不存在：{pred_dir}")
        sys.exit(1)
    if not gt_dir.exists():
        print(f"错误：标注目录不存在：{gt_dir}")
        sys.exit(1)

    print("=" * 60)
    print("简历提取准确率评估")
    print("=" * 60)
    print(f"  预测结果：{pred_dir}")
    print(f"  人工标注：{gt_dir}")

    # 执行批量评估
    report = evaluate_batch(pred_dir, gt_dir)

    # 控制台输出
    print_report(report)

    # 保存 JSON 报告
    save_report(report, Path(args.output))

    # 返回码：整体准确率低于 0.90 视为未达标
    if report.overall_avg < 0.90:
        print(f"\n⚠️  整体准确率 {report.overall_avg:.1%} 低于 0.90，建议检查提示词或数据质量。")
        sys.exit(1)
    else:
        print(f"\n✓ 整体准确率 {report.overall_avg:.1%}，达到基线要求。")
        sys.exit(0)


if __name__ == "__main__":
    main()
