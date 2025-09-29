"""
pipelines_with_fsm/update_flattened_responsibilities_pipeline_fsm.py

DB-native mini-pipeline: ensure `flattened_responsibilities` is up to date from the
current resume JSON for a worklist of job posting URLs.

Why use this
------------
Use this to (re)materialize flattened responsibilities from your single
nested resume JSON for one or more job URLs before downstream FSM stages.
It extracts and validates the resume data, then writes rows to DuckDB’s
`flattened_responsibilities` table.


Worklist behavior
-----------------
- If `url` is provided, process only that URL.
- Otherwise, fetch a worklist from `pipeline_control` where:
  - stage = FLATTENED_RESPONSIBILITIES
  - status = NEW
This mirrors how your other FSM pipelines discover work.

Insert semantics
----------------
- mode="append" (default):
  - Uses table PKs for dedup inside `insert_df_with_config` (no full wipe).
  - Only rows whose full PKs match incoming rows are replaced;
    all other older rows remain.
- mode="replace":
  - Broadens the delete scope by passing `key_cols=["url"]` to
    `insert_df_with_config`,
    effectively **wiping all rows for that URL** before inserting the fresh set.
  - If you want snapshot-only replacement, change to `["url","iteration","version"]`.

What this pipeline does *not* do
--------------------------------
- No explicit DuckDB connections here; lower-level helpers manage that.
- No filesystem writes for the flattened data (this is DB-native).
*- No FSM status updates on `pipeline_control`:
    this moduleRuns a control-plane sync (when do_control_sync=True)
    to upsert pipeline_control entries for processed URLs;
    it does not perform FSM transitions or change status beyond that).

Inputs & outputs
----------------
Input:
  - Resume JSON file (nested, single-resume).
  - URL(s) from pipeline_control (or a single URL you pass in).
Output:
  - Rows in `flattened_responsibilities` (validated & aligned to schema).

Typical usage
-------------
1) Process one URL explicitly:
   >>> run_update_flattened_responsibilities_pipeline_fsm(url="https://example.com/job/123")

2) Process all NEW URLs at the FLATTENED_RESPONSIBILITIES stage:
   >>> run_update_flattened_responsibilities_pipeline_fsm()

3) Replace (full reset) for a URL:
   >>> run_update_flattened_responsibilities_pipeline_fsm(
   ...     url="https://example.com/job/123",
   ...     mode="replace",
   ... )

Errors & logging
----------------
- Validation or alignment errors are logged and the offending URL is skipped.
- Empty worklists short-circuit with an informational log message.

Dependencies
------------
- process_responsibilities_from_resume(...)  → returns validated `Responsibilities`
- flatten_model_to_df(...)                   → aligns to `flattened_responsibilities` schema
- insert_df_with_config(...)                       → handles connection, delete-then-insert, dedup
- get_urls_by_stage_and_status(...)          → pulls NEW URLs for this stage
- sync_flattened_responsibilities_to_pipeline_control() → control-plane sync (optional)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional
import pandas as pd

# ✅ Pure processing (no file I/O): returns a validated Responsibilities model
from job_bot.evaluation_optimization.evaluation_optimization_utils import (
    process_responsibilities_from_resume,
)

# ✅ DB utilities
from job_bot.db_io.flatten_and_rehydrate import flatten_model_to_df
from job_bot.db_io.db_inserters import insert_df_with_config
from job_bot.db_io.pipeline_enums import TableName, PipelineStage, PipelineStatus
from job_bot.db_io.db_utils import get_urls_by_stage_and_status

# ✅ Pydantic models (typing only)
from job_bot.models.resume_job_description_io_models import Responsibilities

# ✅ Control-plane sync helper
from job_bot.fsm.pipeline_control_sync import (
    sync_flattened_responsibilities_to_pipeline_control,
)

# ✅ Default: nested resume JSON (NOT a flat responsibilities file)
from job_bot.config.project_config import RESUME_JSON_FILE as DEFAULT_RESUME_JSON_FILE

logger = logging.getLogger(__name__)


def run_flattened_responsibilities_pipeline_fsm(
    *,
    url: Optional[str] = None,  # If None, pull NEW URLs at FLATTENED_RESPONSIBILITIES
    resume_json_file: str | Path = DEFAULT_RESUME_JSON_FILE,
    mode: Literal["append", "replace"] = "append",
    do_control_sync: bool = True,
) -> None:
    """
    Extract + flatten resume responsibilities and ingest into DuckDB.

    Worklist
    --------
    - If `url` is provided: process only that URL.
    - Else: fetch URLs where stage=FLATTENED_RESPONSIBILITIES AND status=NEW.

    Behavior
    --------
    1) Parse/flatten responsibilities from resume JSON (per URL), validate via Pydantic.
    2) Insert into `flattened_responsibilities`:
       - mode="append"  (default): PK-based dedup insert
       - mode="replace": wipe all rows for that URL first (via broader key set)
    3) Optionally sync `pipeline_control`.
    """
    resume_json_path = Path(resume_json_file)

    # Build worklist
    if url:
        urls = [url]
        logger.info("🔎 Using single URL: %s", url)
    else:
        urls = get_urls_by_stage_and_status(
            stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
            status=PipelineStatus.NEW,
        )
        logger.info(
            "🔎 Fetched %d NEW URL(s) at stage=%s",
            len(urls),
            PipelineStage.FLATTENED_RESPONSIBILITIES.value,
        )

    if not urls:
        logger.info("📭 No URLs to process for flattened responsibilities.")
        return

    for u in urls:
        logger.info(
            "📥 Processing resume JSON (%s) into flattened_responsibilities (mode=%s, url=%s)",
            resume_json_path,
            mode,
            u,
        )

        # 1) Process responsibilities → validated model
        model: Responsibilities | None = process_responsibilities_from_resume(
            resume_json_file=resume_json_path,
            url=u,
        )
        if model is None:
            logger.error("❌ Failed to process responsibilities for url=%s", u)
            continue

        # 2) Flatten to DataFrame aligned to table schema
        df: pd.DataFrame = flatten_model_to_df(
            model=model,
            table_name=TableName.FLATTENED_RESPONSIBILITIES,
            source_file=resume_json_path,
        )

        # 3) Insert (no explicit DB connection here)
        insert_df_with_config(
            df,
            TableName.FLATTENED_RESPONSIBILITIES.value,
            mode=mode,
        )

    # 4) Optional control-plane sync
    if do_control_sync:
        try:
            sync_flattened_responsibilities_to_pipeline_control()
            logger.info(
                "🔄 Control-plane sync complete for flattened_responsibilities."
            )
        except Exception:
            logger.warning("⚠️ Control-plane sync failed (non-fatal).")

    logger.info("✅ Done. Processed %d URL(s).", len(urls))
