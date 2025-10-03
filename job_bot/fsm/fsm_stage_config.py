"""pipelines_with_fsm/pipeline_stage_config.py"""

from typing import Optional
from job_bot.db_io.pipeline_enums import PipelineStage

# ✅ Canonical, ordered rail for the FSM/DB pipeline
PIPELINE_STAGE_SEQUENCE: list[PipelineStage] = [
    PipelineStage.JOB_URLS,
    PipelineStage.JOB_POSTINGS,
    PipelineStage.EXTRACTED_REQUIREMENTS,
    PipelineStage.FLATTENED_REQUIREMENTS,
    PipelineStage.FLATTENED_RESPONSIBILITIES,
    PipelineStage.SIM_METRICS_EVAL,  # eval BEFORE edited
    PipelineStage.EDITED_RESPONSIBILITIES,
    PipelineStage.SIM_METRICS_REVAL,
    PipelineStage.ALIGNMENT_REVIEW,
    PipelineStage.FINAL_RESPONSIBILITIES,
]

# Strings when needed (views, SQL, logging)
PIPELINE_STAGE_SEQUENCE_VALUES = [s.value for s in PIPELINE_STAGE_SEQUENCE]

# Derived maps
_NEXT = {
    cur: (
        PIPELINE_STAGE_SEQUENCE[i + 1] if i + 1 < len(PIPELINE_STAGE_SEQUENCE) else None
    )
    for i, cur in enumerate(PIPELINE_STAGE_SEQUENCE)
}
_PREV = {
    cur: (PIPELINE_STAGE_SEQUENCE[i - 1] if i - 1 >= 0 else None)
    for i, cur in enumerate(PIPELINE_STAGE_SEQUENCE)
}


def next_stage(cur: PipelineStage) -> Optional[PipelineStage]:
    return _NEXT.get(cur)


def prev_stage(cur: PipelineStage) -> Optional[PipelineStage]:
    return _PREV.get(cur)


def validate_stage_set() -> None:
    # Catch drift: rail must include every enum exactly once
    assert set(PIPELINE_STAGE_SEQUENCE) == set(
        PipelineStage
    ), "FSM rail mismatch: PIPELINE_STAGE_SEQUENCE != PipelineStage"
