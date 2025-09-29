# control_plane_maintenance.py
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.fsm.pipeline_control_sync import sync_job_urls_to_pipeline_control

logger = logging.getLogger(__name__)


def apply_worklist_csv(csv_path: str | Path) -> None:
    """
    Apply a human-editable CSV “switchboard” to the `pipeline_control` table.

    Purpose
    -------
    Allows operators to requeue specific URLs back to a target stage
    (e.g., `job_postings`) in one shot.

    CSV schema (headers are required)
    ---------------------------------
    - url:   string (required)
    - action:string (optional) — currently supports 'requeue'
    - stage: string (required when action == 'requeue')

    Example rows
    ------------
    url,action,stage
    https://jobs.example.com/123,requeue,job_postings
    https://jobs.example.com/456,,
    https://jobs.example.com/789,requeue,extracted_requirements

    Behavior
    --------
    For rows where `action='requeue'` and `stage` is provided:
      UPDATE pipeline_control
         SET stage = <stage>, status = 'NEW', updated_at = now()
       WHERE url = <url>;

    Notes
    -----
    • Only affects rows present in the CSV.
    • Non-'requeue' or missing stage rows are ignored.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        logger.info("📭 No worklist CSV found at %s; skipping.", csv_path)
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        logger.info("📭 Worklist CSV is empty; skipping.")
        return

    if "url" not in df.columns:
        raise ValueError("worklist CSV must include a 'url' column")

    # Normalize expected columns
    for col in ("action", "stage"):
        if col not in df.columns:
            df[col] = None

    con = get_db_connection()
    con.register("worklist", df)

    # Requeue selected rows to a target stage
    con.execute(
        """
        UPDATE pipeline_control AS p
        SET stage = w.stage,
            status = 'NEW',
            updated_at = now()
        FROM worklist AS w
        WHERE p.url = w.url
          AND lower(COALESCE(w.action, '')) = 'requeue'
          AND w.stage IS NOT NULL
        """
    )

    # Optional: report how many rows were affected
    # (DuckDB doesn't directly expose rowcount for UPDATE ... FROM with a registered table)
    logger.info("✅ Applied worklist requeue actions from %s.", csv_path)


def run_control_plane_maintenance_pipeline(
    *,
    worklist_csv: Optional[str] = None,
) -> None:
    """
    One-button maintenance for the control plane (`pipeline_control`).

    Steps
    -----
    1) Incrementally sync `job_urls` → `pipeline_control`
       - Inserts new URLs
       - Refreshes light metadata (if your sync does so)
    2) (Optional) Apply a worklist CSV to requeue specific URLs to a stage

    Idempotency
    -----------
    Safe to run repeatedly. Each run will pull in any newly-added URLs and
    apply the latest worklist instructions if provided.
    """
    logger.info("🧭 Running control-plane maintenance …")

    # Step 1: keep `pipeline_control` in sync with `job_urls`
    sync_job_urls_to_pipeline_control()

    # Step 2: apply switchboard CSV (optional)
    if worklist_csv:
        apply_worklist_csv(worklist_csv)

    logger.info("✅ Control-plane maintenance complete.")


def main() -> None:
    """
    Minimal CLI entrypoint for control-plane maintenance.

    Usage
    -----
    python -m job_bot.fsm.control_plane_maintenance --worklist-csv /path/to/worklist.csv
    """
    parser = argparse.ArgumentParser(
        description="Control-plane maintenance for `pipeline_control`."
    )
    parser.add_argument(
        "--worklist-csv",
        type=str,
        default=None,
        help="Optional CSV with columns (url,action,stage) to apply (supports 'requeue').",
    )
    args = parser.parse_args()

    run_control_plane_maintenance_pipeline(worklist_csv=args.worklist_csv)


if __name__ == "__main__":
    main()
