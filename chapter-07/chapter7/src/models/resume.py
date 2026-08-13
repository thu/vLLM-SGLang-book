# src/models/resume.py
# 简历数据模型定义
# 职责：Pydantic模型同时承担三个角色——
#   1. 通过 model_json_schema() 生成 JSON Schema 供 SGLang 约束解码使用
#   2. 通过 model_validate_json() 执行业务规则验证
#   3. 作为代码即文档的系统规格说明

from __future__ import annotations

import re
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


class DegreeType(str, Enum):
    """
    学历枚举。
    约束解码将模型的输出强制限定在这四个值内，
    消除"学士"、"bachelor"、"大学本科"等非标准写法。
    """
    ASSOCIATE = "专科"
    BACHELOR  = "本科"
    MASTER    = "硕士"
    DOCTORATE = "博士"


class BasicInfo(BaseModel):
    """候选人基本联系信息"""
    model_config = ConfigDict(
        json_schema_extra={"description": "候选人基本联系信息"}
    )

    name: str = Field(
        description="候选人姓名",
        min_length=1,
        max_length=50
    )
    phone: Optional[str] = Field(
        default=None,
        description="手机号，如 138-0000-1234 或 +86 13800001234"
    )
    email: Optional[EmailStr] = Field(
        default=None,
        description="电子邮箱"
    )
    address: Optional[str] = Field(
        default=None,
        description="居住地址",
        max_length=200
    )

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        """
        验证手机号合法性。
        规则：去除空格、连字符、+86 前缀后，剩余数字在 10 到 11 位之间。
        使用 validator 而非 pattern，避免正则约束与验证逻辑重复定义。
        """
        if v is None:
            return v
        digits = re.sub(r"[\s\-\+]", "", v)
        if digits.startswith("86") and len(digits) == 13:
            digits = digits[2:]
        if not (10 <= len(digits) <= 11):
            raise ValueError(f"手机号位数不合法：{v}")
        return v


class Education(BaseModel):
    """一段教育经历"""
    model_config = ConfigDict(
        json_schema_extra={"description": "一段教育经历"}
    )

    school: str = Field(description="学校名称")
    major:  str = Field(description="专业名称")
    degree: DegreeType = Field(description="学历层次")
    start_date: Optional[str] = Field(
        default=None,
        description="入学时间，格式 YYYY 或 YYYY-MM",
        pattern=r"^\d{4}(-\d{2})?$"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="毕业时间，格式 YYYY 或 YYYY-MM，在读填 present",
        pattern=r"^\d{4}(-\d{2})?$|^present$"
    )


class WorkExperience(BaseModel):
    """一段工作经历"""
    model_config = ConfigDict(
        json_schema_extra={"description": "一段工作经历"}
    )

    company:  str = Field(description="公司名称")
    position: str = Field(description="职位名称")
    start_date: Optional[str] = Field(
        default=None,
        description="入职时间，格式 YYYY 或 YYYY-MM",
        pattern=r"^\d{4}(-\d{2})?$"
    )
    end_date: Optional[str] = Field(
        default=None,
        description="离职时间，格式 YYYY 或 YYYY-MM，当前在职填 present",
        pattern=r"^\d{4}(-\d{2})?$|^present$"
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="主要职责和成就，每条一个字符串，保留关键数字指标"
    )


class Skills(BaseModel):
    """技能分类列表"""
    model_config = ConfigDict(
        json_schema_extra={"description": "技能分类列表"}
    )

    programming_languages: List[str] = Field(
        default_factory=list,
        description="编程语言，如 Python、Java、Go"
    )
    frameworks: List[str] = Field(
        default_factory=list,
        description="框架，如 Django、Spring Boot、React"
    )
    databases: List[str] = Field(
        default_factory=list,
        description="数据库，如 MySQL、Redis、MongoDB"
    )
    tools: List[str] = Field(
        default_factory=list,
        description="工具，如 Docker、Kubernetes、Git"
    )


class Resume(BaseModel):
    """简历信息提取结果"""
    model_config = ConfigDict(
        json_schema_extra={"title": "简历信息提取结果"}
    )

    basic_info:      BasicInfo
    education:       List[Education]      = Field(default_factory=list)
    work_experience: List[WorkExperience] = Field(default_factory=list)
    skills:          Optional[Skills]     = None
