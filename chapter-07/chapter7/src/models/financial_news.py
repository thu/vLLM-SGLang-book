# src/models/financial_news.py
# 金融舆情数据模型
# 核心挑战：一篇新闻可能涉及多家公司和多个事件类型，
# 信息密度高，要求模型保持全局视角不遗漏任何相关实体。

from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


class SentimentType(str, Enum):
    """情感倾向枚举"""
    POSITIVE = "积极"
    NEUTRAL  = "中性"
    NEGATIVE = "消极"


class EventType(str, Enum):
    """事件类型枚举"""
    EARNINGS    = "财报发布"
    MERGER      = "并购重组"
    LEADERSHIP  = "高管变动"
    INVESTMENT  = "投融资"
    PRODUCT     = "产品发布"
    LEGAL       = "法律诉讼"
    REGULATORY  = "监管处罚"
    PARTNERSHIP = "战略合作"
    MARKET      = "市场变化"
    OTHER       = "其他"


class CompanyMention(BaseModel):
    """新闻中提及的公司实体"""
    model_config = ConfigDict(
        json_schema_extra={"description": "新闻中提及的公司"}
    )

    name:      str                = Field(
        description="公司名称，使用全称，如'阿里巴巴集团控股有限公司'"
    )
    role:      Optional[str]      = Field(
        default=None,
        description="在事件中的角色，如'收购方'、'被收购方'、'监管机构'"
    )
    sentiment: SentimentType      = Field(
        description="该公司在此事件中的情感倾向"
    )


class FinancialNews(BaseModel):
    """金融舆情分析结果"""
    model_config = ConfigDict(
        json_schema_extra={"title": "金融舆情分析"}
    )

    event_type:        EventType             = Field(
        description="主要事件类型"
    )
    companies:         List[CompanyMention]  = Field(
        default_factory=list,
        description="新闻中涉及的所有公司，包括配角公司，不得遗漏"
    )
    key_figures:       List[str]             = Field(
        default_factory=list,
        description="关键数字指标，如'净利润同比增长23%'、'交易金额50亿元'"
    )
    impact_assessment: Optional[str]         = Field(
        default=None,
        description="对市场或相关公司的影响分析，2-3句话"
    )
    overall_sentiment: SentimentType         = Field(
        description="新闻整体情感倾向"
    )
