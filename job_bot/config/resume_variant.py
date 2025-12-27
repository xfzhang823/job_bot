"""job_bot/configs/resume_variant.py"""

from enum import Enum


class ResumeVariant(str, Enum):
    """enum class for different versions of resume"""

    MI_STRATEGY = "mi_strategy"
    AI_ARCHITECT = "ai_architect"
