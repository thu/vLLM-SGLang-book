# src/models/__init__.py
from .resume import Resume, BasicInfo, Education, WorkExperience, Skills, DegreeType
from .contract import Contract, RiskItem, RiskLevel, ContractType
from .medical_report import MedicalReport, LabResult, AbnormalLevel, ReportType
from .financial_news import FinancialNews, CompanyMention, SentimentType, EventType

__all__ = [
    # 简历模型
    "Resume", "BasicInfo", "Education", "WorkExperience", "Skills", "DegreeType",
    # 合同模型
    "Contract", "RiskItem", "RiskLevel", "ContractType",
    # 医疗报告模型
    "MedicalReport", "LabResult", "AbnormalLevel", "ReportType",
    # 金融舆情模型
    "FinancialNews", "CompanyMention", "SentimentType", "EventType",
]
