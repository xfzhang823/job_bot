"""job_bot/pipelines_with_fsm/update_job_urls_pipeline_fsm.py"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

from job_bot.db_io.create_db_tables import create_all_db_tables
from job_bot.db_io.pipeline_enums import TableName
from job_bot.db_io.file_ingestion import ingest_single_file  # ← delegate here

from job_bot.config.project_config import JOB_POSTING_URLS_FILE
from job_bot.utils.pydantic_model_loaders_for_files import (
    load_job_posting_urls_file_model,
)

logger = logging.getLogger(__name__)


def run_ingest_job_urls_pipeline_fsm(
    file_path: Path | str = JOB_POSTING_URLS_FILE,
    mode: Literal["append", "replace"] = "append",
) -> None:
    """
    Ingest the job URLs JSON file into DuckDB (`job_urls` table).

    - Default `mode="append"` inserts only new rows (dedup aware).
    - Use `mode="replace"` to refresh matching rows based on the table’s PKs.

    Args:
        file_path: Path to the job URLs JSON (default: JOB_POSTING_URLS_FILE).
        mode: "append" | "replace" (default: "append").

    Returns:
        None
    """
    logger.info("✅ Running ingest job urls pipeline...")
    logger.info("🏗️ Ensuring DuckDB schema exists …")
    create_all_db_tables()

    file_path = Path(file_path)
    logger.info(f"📥 Ingesting job URLs from: {file_path} (mode={mode})")

    # job_urls is a seed table → no iteration/version/provider/model_id needed
    ingest_single_file(
        TableName.JOB_URLS,
        file_path,
        load_job_posting_urls_file_model,
        mode=mode,
    )

    logger.info("✅ job_urls ingestion complete.")
