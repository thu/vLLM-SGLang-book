# src/models/contract.py
# 合同信息数据模型
# 与简历模型的核心差异：包含"推断类字段"（风险评估），
# 模型不只是从文本摘录信息，还需要根据内容做出判断。

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class RiskLevel(str, Enum):
    """
    风险等级枚举。
    约束解码保证模型只能输出这四个值，
    消除"高风险"/"高"/"HIGH"等写法不一致问题。
    """
    LOW      = "低风险"
    MEDIUM   = "中等风险"
    HIGH     = "高风险"
    CRITICAL = "严重风险"


class ContractType(str, Enum):
    """合同类型枚举"""
    SALES       = "销售合同"
    SERVICE     = "服务合同"
    EMPLOYMENT  = "劳动合同"
    LEASE       = "租赁合同"
    NDA         = "保密协议"
    PARTNERSHIP = "合作协议"
    OTHER       = "其他"


class RiskItem(BaseModel):
    """单个风险条款"""
    model_config = ConfigDict(
        json_schema_extra={"description": "识别出的风险条款"}
    )

    clause:      str       = Field(description="涉及的合同条款摘要，不超过100字")
    risk_level:  RiskLevel = Field(description="风险等级")
    description: str       = Field(description="风险说明，解释为何构成风险")
    suggestion:  str       = Field(description="修改建议")


class Contract(BaseModel):
    """合同信息提取结果"""
    model_config = ConfigDict(
        json_schema_extra={"title": "合同信息提取"}
    )

    parties:         List[str]      = Field(
        description="合同各方名称列表，按甲方、乙方顺序排列"
    )
    contract_type:   ContractType   = Field(
        description="合同类型"
    )
    effective_date:  Optional[str]  = Field(
        default=None,
        description="生效日期，格式 YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    expiry_date:     Optional[str]  = Field(
        default=None,
        description="到期日期，格式 YYYY-MM-DD",
        pattern=r"^\d{4}-\d{2}-\d{2}$"
    )
    total_amount:    Optional[str]  = Field(
        default=None,
        description="合同金额，含币种，如 '100,000 元人民币'"
    )
    key_obligations: List[str]      = Field(
        default_factory=list,
        description="各方主要义务，每条一个字符串"
    )
    risk_items:      List[RiskItem] = Field(
        default_factory=list,
        description="识别出的风险条款列表"
    )
    overall_risk:    RiskLevel      = Field(
        description="基于所有风险条款的整体风险评级"
    )
