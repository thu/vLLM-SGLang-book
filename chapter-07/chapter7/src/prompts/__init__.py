# src/prompts/__init__.py
from .few_shot_selector import (
    FewShotExample,
    BUILTIN_EXAMPLES,
    build_few_shot_prompt,
    get_prompt_with_examples,
    list_examples,
    DATE_NORMALIZATION_EXAMPLE,
    CURRENT_JOB_EXAMPLE,
    SKILL_CLASSIFICATION_EXAMPLE,
)

__all__ = [
    "FewShotExample",
    "BUILTIN_EXAMPLES",
    "build_few_shot_prompt",
    "get_prompt_with_examples",
    "list_examples",
    "DATE_NORMALIZATION_EXAMPLE",
    "CURRENT_JOB_EXAMPLE",
    "SKILL_CLASSIFICATION_EXAMPLE",
]
