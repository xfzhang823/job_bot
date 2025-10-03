"""
fsm/check_pipeline_fsm.py

FSM integrity validation for the pipeline control table.

This module validates that the control-plane snapshot (`pipeline_control`)
is consistent with the defined FSM:

1. **Enum validity**: All rows have valid `stage` and `status` values.
2. **Prerequisite presence**: For each stage, the required upstream tables
   contain rows for that URL (ensuring the control stage is backed by data).
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence
import logging

# Project imports
from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.db_utils import (
    get_urls_ready_for_transition,
    get_pipeline_state,
)
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    PipelineProcessStatus,
    TableName,
)

logger = logging.getLogger(__name__)


def _default_stage_order() -> Sequence[PipelineStage]:
    """
    Return the canonical order of pipeline stages.

    Override this function if the FSM stage sequence changes.
    """
    return [
        PipelineStage.JOB_URLS,
        PipelineStage.JOB_POSTINGS,
        PipelineStage.EXTRACTED_REQUIREMENTS,
        PipelineStage.FLATTENED_REQUIREMENTS,
        PipelineStage.FLATTENED_RESPONSIBILITIES,
        PipelineStage.SIM_METRICS_EVAL,
        PipelineStage.EDITED_RESPONSIBILITIES,
        PipelineStage.SIM_METRICS_REVAL,
        PipelineStage.ALIGNMENT_REVIEW,
        # PipelineStage.FINAL_RESPONSIBILITIES,  # TODO: enable later
    ]


def _default_stage_prereqs() -> Mapping[PipelineStage, Iterable[TableName]]:
    """
    Define prerequisite tables for each stage.

    A stage is only valid if rows exist for the URL in all
    of its prerequisite tables.
    """
    return {
        PipelineStage.JOB_URLS: [TableName.JOB_URLS],
        PipelineStage.JOB_POSTINGS: [
            TableName.JOB_URLS,
            TableName.JOB_POSTINGS,
        ],
        PipelineStage.EXTRACTED_REQUIREMENTS: [
            TableName.JOB_URLS,
            TableName.JOB_POSTINGS,
            TableName.EXTRACTED_REQUIREMENTS,
        ],
        PipelineStage.FLATTENED_REQUIREMENTS: [
            TableName.EXTRACTED_REQUIREMENTS,
            TableName.FLATTENED_REQUIREMENTS,
        ],
        PipelineStage.FLATTENED_RESPONSIBILITIES: [
            TableName.FLATTENED_RESPONSIBILITIES,
        ],
        # Eval sim metrics (before editing)
        PipelineStage.SIM_METRICS_EVAL: [
            TableName.FLATTENED_REQUIREMENTS,
            TableName.FLATTENED_RESPONSIBILITIES,
            TableName.SIMILARITY_METRICS,  # optional: enforce presence of outputs
        ],
        PipelineStage.EDITED_RESPONSIBILITIES: [
            TableName.FLATTENED_RESPONSIBILITIES,
            TableName.EDITED_RESPONSIBILITIES,
        ],
        # Reval sim metrics (after editing)
        PipelineStage.SIM_METRICS_REVAL: [
            TableName.FLATTENED_REQUIREMENTS,
            TableName.EDITED_RESPONSIBILITIES,
            TableName.SIMILARITY_METRICS,  # optional
        ],
        # todo: comment out for now; add in couple days
        # # Human-in-the-loop alignment review
        # PipelineStage.ALIGNMENT_REVIEW: [
        #     TableName.SIMILARITY_METRICS,
        #     TableName.ALIGNMENT_CROSSTAB,
        # ],
        # TODO: implement later
        # PipelineStage.FINAL_RESPONSIBILITIES: [
        #     TableName.EDITED_RESPONSIBILITIES,
        #     TableName.FINAL_RESPONSIBILITIES,
        # ],
    }


def check_fsm_integrity(
    *,
    statuses: Sequence[PipelineStatus] = (
        PipelineStatus.NEW,
        PipelineStatus.IN_PROGRESS,
        PipelineStatus.COMPLETED,
        PipelineStatus.ERROR,
    ),
    stage_order: Sequence[PipelineStage] | None = None,
    stage_prereqs: Mapping[PipelineStage, Iterable[TableName]] | None = None,
    verbose: bool = True,
) -> dict:
    """
    Validate consistency of the `pipeline_control` snapshot.

    Checks performed:
    -----------------
    1. **Enum validity**:
       Ensure all rows have valid `stage` and `status` values.
    2. **Prerequisite presence**:
       For each stage, verify that required upstream tables
       contain rows for the given URL.

    Args:
        statuses: Pipeline statuses to include when checking each stage.
        stage_order: Optional override for canonical stage order.
        stage_prereqs: Optional override for prerequisite rules.
        verbose: If True, prints human-readable results.

    Returns:
        dict containing:
          - "invalid_enum": list of (url, stage, status) tuples with
            invalid values
          - "missing_prereqs": list of (url, stage, missing_table) tuples
    """
    stage_order = stage_order or _default_stage_order()
    prereqs = stage_prereqs or _default_stage_prereqs()

    con = get_db_connection()

    issues = {
        "invalid_enum": [],  # rows with invalid stage/status
        "missing_prereqs": [],  # rows missing required upstream data
    }

    # --- 1) Enum validity ---
    enum_df = con.execute("SELECT url, stage, status FROM pipeline_control").df()
    valid_stages = {s.value for s in stage_order}
    valid_statuses = {s.value for s in PipelineStatus}

    for _, row in enum_df.iterrows():
        if row["stage"] not in valid_stages or row["status"] not in valid_statuses:
            issues["invalid_enum"].append((row["url"], row["stage"], row["status"]))

    # --- 2) Prerequisite presence ---
    def _urls_for_stage(s: PipelineStage) -> set[str]:
        urls: set[str] = set()
        for st in statuses:
            urls.update(get_urls_ready_for_transition(stage=s))
        return urls

    def _has_rows(table: TableName, url: str) -> bool:
        sql = f"SELECT 1 FROM {table.value} WHERE url = ? LIMIT 1"
        return not con.execute(sql, (url,)).df().empty

    for stage in stage_order:
        urls = _urls_for_stage(stage)
        required_tables = prereqs.get(stage, [])
        for url in urls:
            _ = get_pipeline_state(
                url
            )  # snapshot (currently unused, but kept for parity)
            for t in required_tables:
                if not _has_rows(t, url):
                    issues["missing_prereqs"].append((url, stage.value, t.value))

    # --- Reporting ---
    if verbose:
        if not any(issues.values()):
            print("No FSM integrity issues detected.")
        else:
            if issues["invalid_enum"]:
                print("Invalid enum rows:")
                for url, stg, sts in issues["invalid_enum"]:
                    print(f"  - {url}  stage='{stg}'  status='{sts}'")
            if issues["missing_prereqs"]:
                print("Missing prerequisite data:")
                for url, stg, tbl in issues["missing_prereqs"]:
                    print(f"  - {url}  control_stage={stg}  missing_table={tbl}")

    return issues
