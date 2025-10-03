"""
fsm/fsm_state_control.py

Manages creation, updating, bulk operations, and FSM steps for pipeline states.
"""

import logging
from datetime import datetime
from job_bot.db_io.state_sync import (
    load_pipeline_state,
)
from job_bot.db_io.persist_pipeline_state import update_and_persist_pipeline_state
from job_bot.db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    # Version,
)
from job_bot.fsm.pipeline_fsm_manager import PipelineFSMManager
from job_bot.models.db_table_models import PipelineState


logger = logging.getLogger(__name__)


class PipelineControl:
    """
    PipelineControl is the controller and utility interface for managing
    pipeline states in the pipeline_control DuckDB table.

    Functions include:
    - Creating new FSM states for unseen job URLs.
    - Bulk-updating pipeline status (e.g., set multiple URLs to in_progress).
    - Stepping the FSM forward for a specific URL.
    """

    def __init__(self):
        self.fsm_manager = PipelineFSMManager()

    def initialize_new_urls(
        self,
        urls: list[str],
        iteration: int = 0,
    ) -> None:
        """
        Initializes pipeline state for a list of job URLs if they are not already present.

        For each URL:
        - Checks if a PipelineState exists in DuckDB.
        - If not, creates a new record with default metadata (created_at auto-filled).
        - Skips already-initialized URLs.

        Args:
            urls (list[str]): Job posting URLs to initialize.
            iteration (int): Optional iteration count (default: 0).
        """
        for url in urls:
            try:
                state = load_pipeline_state(url)
                if not state:
                    new_state = PipelineState(
                        url=url,
                        iteration=iteration,
                        source_file=None,
                        # version is optional on control-plane; start as None.
                        version=None,
                        stage=PipelineStage.JOB_URLS,
                        status=PipelineStatus.NEW,
                        notes=None,
                        created_at=datetime.now(),
                        updated_at=datetime.now(),
                    )
                    update_and_persist_pipeline_state(new_state)
                    logger.info(f"✅ Initialized pipeline state for URL: {url}")
                else:
                    logger.info(f"🔁 URL already initialized: {url}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize URL {url}: {e}", exc_info=True)

    def skip(self, urls: list[str], reason: str = ""):
        self._bulk_update(
            urls,
            fields={"status": PipelineStatus.SKIPPED},
            reason=reason or "Manually skipped",
        )

    def mark_failed(self, urls: list[str], reason: str = ""):
        self._bulk_update(
            urls,
            fields={"status": PipelineStatus.ERROR},
            reason=reason or "Marked failed",
        )

    def retry(
        self,
        urls: list[str],
        restart_stage: PipelineStage,
        reset_status: PipelineStatus = PipelineStatus.NEW,
        reason: str = "",
    ):
        """
        Roll back the stage for each URL and reset its status.
        """
        self._bulk_update(
            urls,
            fields={
                "stage": restart_stage,
                "status": reset_status,
            },
            reason=reason or f"Manual retry from {restart_stage}",
        )

    def _bulk_update(self, urls: list[str], fields: dict, reason: str = ""):
        for url in urls:
            try:
                state = load_pipeline_state(url)
                if not state:
                    logger.warning(f"⚠️ URL not found: {url}")
                    continue

                # Update fields
                for k, v in fields.items():
                    setattr(state, k, v)

                # Append to notes
                if reason:
                    note_line = f"[{datetime.now().isoformat()}] {reason}"
                    state.notes = f"{note_line}\n{state.notes or ''}".strip()

                # Touch updated_at (TimestampedMixin field on PipelineState)
                state.updated_at = datetime.now()

                update_and_persist_pipeline_state(state)
                logger.info(f"✅ Updated state for {url}: {fields}")
            except Exception as e:
                logger.error(f"❌ Failed to update {url}: {e}", exc_info=True)

    def step_fsm(self, url: str):
        """
        Explicitly advance FSM for a URL.
        """
        fsm = self.fsm_manager.get_fsm(url)
        fsm.step()
