"""
job_bot/configs/io_configs.py

input output configuration tools.
"""

from pathlib import Path
from job_bot.config.resume_variant import ResumeVariant
from job_bot.config.project_config import (
    RESUME_AI_ARCHITECT_JSON_FILE,
    RESUME_MI_STRATEGY_JSON_FILE,
)

RESUME_FILES = {
    ResumeVariant.MI_STRATEGY: RESUME_MI_STRATEGY_JSON_FILE,
    ResumeVariant.AI_ARCHITECT: RESUME_AI_ARCHITECT_JSON_FILE,
}


def get_resume_json_path(variant: ResumeVariant) -> Path:
    """ "
    Get resume json path based on ResumeVariant (which version of resume
    templete)

    """
    try:
        return RESUME_FILES[variant]
    except KeyError:
        raise ValueError(f"Unsupported resume verseion.")
