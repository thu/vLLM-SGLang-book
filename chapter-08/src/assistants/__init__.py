# src/assistants/__init__.py
from .coding_assistant import generate_code, debug_code, optimize_code, explain_code
from .education_bot import (
    explain_concept, evaluate_student_answer,
    generate_adaptive_exercise, create_study_plan,
)
from .knowledge_base_assistant import (
    answer_with_retrieved_context, disambiguate_query, answer_without_docs,
)

__all__ = [
    "generate_code", "debug_code", "optimize_code", "explain_code",
    "explain_concept", "evaluate_student_answer",
    "generate_adaptive_exercise", "create_study_plan",
    "answer_with_retrieved_context", "disambiguate_query", "answer_without_docs",
]
