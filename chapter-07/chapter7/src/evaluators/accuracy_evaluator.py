# src/evaluators/accuracy_evaluator.py
# 准确率评估框架
#
# 设计原则（对应正文 7.5.1）：
# 按字段语义特征分三类：
#   exact  - 精确匹配：姓名、邮箱、手机号、学历（规范化后完全一致）
#   fuzzy  - 模糊匹配：地址、公司名、学校名、职位名（编辑距离相似度）
#   set    - 集合F1：技能列表、职责条目（兼顾精确率和召回率）
#
# 按字段类别分别报告，不用单一平均数掩盖关键字段的局部失败。

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from loguru import logger


# ──────────────────────────────────────────────
# 字段得分计算
# ──────────────────────────────────────────────

def field_score(
    predicted: Any,
    ground_truth: Any,
    field_type: str,
) -> float:
    """
    计算单个字段的准确率得分，返回 0.0~1.0 之间的浮点数。

    Args:
        predicted:    模型提取的值
        ground_truth: 人工标注的真实值
        field_type:   "exact" | "fuzzy" | "set"

    Returns:
        得分：0.0（完全错误）~ 1.0（完全正确）
    """
    # 双方都为空：正确
    if predicted is None and ground_truth is None:
        return 1.0
    # 一方为空：错误
    if predicted is None or ground_truth is None:
        return 0.0

    if field_type == "exact":
        # 规范化后精确匹配：去除空白、统一小写
        norm = lambda s: re.sub(r"\s+", "", str(s)).lower()
        return 1.0 if norm(predicted) == norm(ground_truth) else 0.0

    elif field_type == "fuzzy":
        # 字符级编辑距离相似度
        ratio = SequenceMatcher(
            None,
            str(predicted).strip(),
            str(ground_truth).strip(),
        ).ratio()
        return ratio  # 连续得分，0.0~1.0

    elif field_type == "set":
        # 集合 F1：兼顾精确率和召回率
        pred_set = {s.strip().lower() for s in (predicted or [])}
        gt_set   = {s.strip().lower() for s in (ground_truth or [])}

        # ground truth 为空
        if not gt_set:
            return 1.0 if not pred_set else 0.0

        tp        = len(pred_set & gt_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall    = tp / len(gt_set)

        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    raise ValueError(f"未知的 field_type：{field_type!r}，应为 exact / fuzzy / set")


# ──────────────────────────────────────────────
# 简历级评估
# ──────────────────────────────────────────────

# 字段类型映射：字段路径 → 评估类型
FIELD_TYPE_MAP: dict[str, str] = {
    # 精确匹配字段
    "basic_info.name":   "exact",
    "basic_info.phone":  "exact",
    "basic_info.email":  "exact",
    # 模糊匹配字段
    "basic_info.address":             "fuzzy",
    "education.school":               "fuzzy",
    "education.major":                "fuzzy",
    "education.degree":               "exact",
    "work_experience.company":        "fuzzy",
    "work_experience.position":       "fuzzy",
    "work_experience.start_date":     "exact",
    "work_experience.end_date":       "exact",
    # 集合 F1 字段
    "work_experience.responsibilities": "set",
    "skills.programming_languages":     "set",
    "skills.frameworks":                "set",
    "skills.databases":                 "set",
    "skills.tools":                     "set",
}


@dataclass
class ResumeEvalResult:
    """单份简历的评估结果"""
    resume_id:    str
    field_scores: dict[str, float]  # 字段路径 → 得分
    overall:      float             # 所有字段的简单平均
    has_errors:   bool              # 是否存在得分为 0 的必填字段


def evaluate_resume(
    predicted: dict,
    ground_truth: dict,
    resume_id: str = "unknown",
) -> ResumeEvalResult:
    """
    评估单份简历的提取结果。

    Args:
        predicted:    模型提取结果（dict，由 Resume.model_dump() 获得）
        ground_truth: 人工标注结果（同格式 dict）
        resume_id:    简历标识，用于日志

    Returns:
        ResumeEvalResult
    """
    scores: dict[str, float] = {}

    # ── 基本信息 ──
    pred_basic = predicted.get("basic_info") or {}
    gt_basic   = ground_truth.get("basic_info") or {}

    for sub_field in ("name", "phone", "email", "address"):
        key = f"basic_info.{sub_field}"
        ftype = FIELD_TYPE_MAP.get(key, "fuzzy")
        scores[key] = field_score(
            pred_basic.get(sub_field),
            gt_basic.get(sub_field),
            ftype,
        )

    # ── 教育经历（取第一段做对比，多段取平均）──
    pred_edus = predicted.get("education") or []
    gt_edus   = ground_truth.get("education") or []

    for edu_field in ("school", "major", "degree", "start_date", "end_date"):
        key   = f"education.{edu_field}"
        ftype = FIELD_TYPE_MAP.get(key, "fuzzy")
        n     = max(len(pred_edus), len(gt_edus), 1)
        edu_scores = []
        for i in range(n):
            pred_val = pred_edus[i].get(edu_field) if i < len(pred_edus) else None
            gt_val   = gt_edus[i].get(edu_field)   if i < len(gt_edus)   else None
            edu_scores.append(field_score(pred_val, gt_val, ftype))
        scores[key] = sum(edu_scores) / len(edu_scores)

    # ── 工作经历 ──
    pred_works = predicted.get("work_experience") or []
    gt_works   = ground_truth.get("work_experience") or []

    for work_field in ("company", "position", "start_date", "end_date", "responsibilities"):
        key   = f"work_experience.{work_field}"
        ftype = FIELD_TYPE_MAP.get(key, "fuzzy")
        n     = max(len(pred_works), len(gt_works), 1)
        work_scores = []
        for i in range(n):
            pred_val = pred_works[i].get(work_field) if i < len(pred_works) else None
            gt_val   = gt_works[i].get(work_field)   if i < len(gt_works)   else None
            work_scores.append(field_score(pred_val, gt_val, ftype))
        scores[key] = sum(work_scores) / len(work_scores)

    # ── 技能 ──
    pred_skills = predicted.get("skills") or {}
    gt_skills   = ground_truth.get("skills") or {}

    for skill_field in ("programming_languages", "frameworks", "databases", "tools"):
        key = f"skills.{skill_field}"
        scores[key] = field_score(
            pred_skills.get(skill_field) if pred_skills else None,
            gt_skills.get(skill_field)   if gt_skills   else None,
            "set",
        )

    overall   = sum(scores.values()) / len(scores) if scores else 0.0
    has_errors = scores.get("basic_info.name", 1.0) < 0.99

    return ResumeEvalResult(
        resume_id=resume_id,
        field_scores=scores,
        overall=overall,
        has_errors=has_errors,
    )


# ──────────────────────────────────────────────
# 批量评估与报告
# ──────────────────────────────────────────────

@dataclass
class BatchEvalReport:
    """批量评估汇总报告"""
    n_total:       int
    n_evaluated:   int
    per_field_avg: dict[str, float]   # 字段 → 平均得分
    overall_avg:   float
    worst_samples: list[dict]         # 得分最低的 10 个样本


def evaluate_batch(
    predictions_dir: str | Path,
    ground_truth_dir: str | Path,
) -> BatchEvalReport:
    """
    批量评估提取结果。

    Args:
        predictions_dir:  模型提取结果目录（resume_XXXX.json）
        ground_truth_dir: 人工标注目录（resume_XXXX.json，同名文件）

    Returns:
        BatchEvalReport
    """
    predictions_dir  = Path(predictions_dir)
    ground_truth_dir = Path(ground_truth_dir)

    # 找到两个目录中文件名相同的文件对
    pred_files = {f.name: f for f in predictions_dir.glob("resume_*.json")
                  if "error" not in f.name and "truncated" not in f.name}
    gt_files   = {f.name: f for f in ground_truth_dir.glob("resume_*.json")}

    common = sorted(set(pred_files) & set(gt_files))
    logger.info(f"找到 {len(common)} 对可评估样本（预测 {len(pred_files)}，标注 {len(gt_files)}）")

    all_results: list[ResumeEvalResult] = []

    for fname in common:
        try:
            pred = json.loads(pred_files[fname].read_text(encoding="utf-8"))
            gt   = json.loads(gt_files[fname].read_text(encoding="utf-8"))
            result = evaluate_resume(pred, gt, resume_id=fname)
            all_results.append(result)
        except Exception as e:
            logger.warning(f"评估失败 {fname}：{e}")

    if not all_results:
        logger.error("没有可评估的样本")
        return BatchEvalReport(
            n_total=len(pred_files), n_evaluated=0,
            per_field_avg={}, overall_avg=0.0, worst_samples=[]
        )

    # 按字段汇总平均得分
    all_field_keys = list(all_results[0].field_scores.keys())
    per_field_avg: dict[str, float] = {}
    for key in all_field_keys:
        vals = [r.field_scores[key] for r in all_results if key in r.field_scores]
        per_field_avg[key] = sum(vals) / len(vals) if vals else 0.0

    overall_avg = sum(r.overall for r in all_results) / len(all_results)

    # 得分最低的 10 个样本（用于诊断）
    sorted_results = sorted(all_results, key=lambda r: r.overall)
    worst_samples = [
        {
            "resume_id": r.resume_id,
            "overall":   round(r.overall, 4),
            "low_fields": {
                k: round(v, 4)
                for k, v in r.field_scores.items()
                if v < 0.8
            },
        }
        for r in sorted_results[:10]
    ]

    return BatchEvalReport(
        n_total=len(pred_files),
        n_evaluated=len(all_results),
        per_field_avg={k: round(v, 4) for k, v in per_field_avg.items()},
        overall_avg=round(overall_avg, 4),
        worst_samples=worst_samples,
    )


def print_report(report: BatchEvalReport) -> None:
    """将评估报告打印到控制台"""
    print("\n" + "=" * 60)
    print("准确率评估报告")
    print("=" * 60)
    print(f"总样本数：{report.n_total}，已评估：{report.n_evaluated}")
    print(f"整体平均准确率：{report.overall_avg:.1%}")

    print("\n── 分字段得分 ──")
    for field_path, avg in sorted(report.per_field_avg.items()):
        bar_len = int(avg * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        flag = "  ⚠️" if avg < 0.80 else ""
        print(f"  {field_path:<42} [{bar}] {avg:.1%}{flag}")

    if report.worst_samples:
        print(f"\n── 得分最低的 {len(report.worst_samples)} 个样本 ──")
        for s in report.worst_samples:
            print(f"  {s['resume_id']:30} 整体：{s['overall']:.1%}")
            for f_path, score in s["low_fields"].items():
                print(f"    └─ {f_path}: {score:.1%}")

    print("=" * 60)
