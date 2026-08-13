# src/models/medical_report.py
# 医疗报告数据模型
# 核心挑战：异常值判断——每个检验项目需要同时记录测量值、
# 参考范围和异常状态，要求模型在提取数值的同时做出比较判断。
# 注意：危急值字段使用独立的召回率指标评估，不允许漏报。

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class AbnormalLevel(str, Enum):
    """
    异常程度枚举。
    区分轻度偏高/偏低与明显偏高/偏低，
    帮助临床医生快速判断处理优先级。
    """
    NORMAL        = "正常"
    SLIGHTLY_HIGH = "轻度偏高"
    SLIGHTLY_LOW  = "轻度偏低"
    HIGH          = "明显偏高"
    LOW           = "明显偏低"
    CRITICAL      = "危急值"


class ReportType(str, Enum):
    """报告类型枚举"""
    BLOOD_TEST  = "血液检查"
    URINE_TEST  = "尿液检查"
    CT_SCAN     = "CT扫描"
    MRI         = "核磁共振"
    XRAY        = "X光检查"
    ULTRASOUND  = "超声检查"
    ECG         = "心电图"
    PATHOLOGY   = "病理检查"
    OTHER       = "其他"


class LabResult(BaseModel):
    """单个检验项目结果"""
    model_config = ConfigDict(
        json_schema_extra={"description": "单个检验项目结果"}
    )

    item_name:       str           = Field(description="检验项目名称，如'空腹血糖'")
    value:           str           = Field(description="检测值，含单位，如 '5.2 mmol/L'")
    reference_range: Optional[str] = Field(
        default=None,
        description="参考范围，如 '3.9-6.1 mmol/L'"
    )
    abnormal:        bool          = Field(
        description="是否超出参考范围，true 或 false"
    )
    abnormal_level:  AbnormalLevel = Field(
        description="异常程度，正常时填'正常'"
    )


class MedicalReport(BaseModel):
    """医疗报告分析结果"""
    model_config = ConfigDict(
        json_schema_extra={"title": "医疗报告分析"}
    )

    report_type:   ReportType      = Field(description="报告类型")
    report_date:   Optional[str]   = Field(
        default=None,
        description="报告日期，格式 YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    patient_name:  Optional[str]   = Field(
        default=None,
        description="患者姓名（如报告中有记录）"
    )
    lab_results:   List[LabResult] = Field(
        default_factory=list,
        description="各检验项目结果列表"
    )
    conclusion:    Optional[str]   = Field(
        default=None,
        description="报告结论摘要，2-3句话"
    )
    suggestions:   List[str]       = Field(
        default_factory=list,
        description="医嘱建议列表，每条一个字符串"
    )
    urgent_flags:  List[str]       = Field(
        default_factory=list,
        description="需立即处理的危急值项目名称列表，无危急值时为空列表"
    )
