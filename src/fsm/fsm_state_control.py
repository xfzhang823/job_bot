"""
fsm/fsm_state_control.py

Manages creation, updating, bulk operations, and FSM steps for pipeline states.
"""

import logging
from datetime import datetime
from db_io.state_sync import persist_pipeline_state_to_duckdb, load_pipeline_state
from db_io.pipeline_enums import PipelineStage, PipelineStatus, LLMProvider, Version
from fsm.pipeline_fsm_manager import PipelineFSMManager
from models.duckdb_table_models import PipelineState


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
        llm_provider: str = "openai",
    ):
        """
        Initializes pipeline state for a list of job URLs if they are not already present.

        For each URL:
        - Checks if a PipelineState exists in DuckDB.
        - If not, creates a new record with default metadata.
        - Skips already-initialized URLs.

        Args:
            urls (list[str]): Job posting URLs to initialize.
            iteration (int): Optional iteration count (default: 0).
            llm_provider (str): LLM provider to associate with the state (default: "openai").
        """
        for url in urls:
            try:
                state = load_pipeline_state(url)
                if not state:
                    new_state = PipelineState(
                        url=url,
                        iteration=iteration,
                        source_file=None,
                        timestamp=datetime.now(),
                        llm_provider=LLMProvider(llm_provider),
                        version=Version.ORIGINAL,
                        last_stage=PipelineStage.JOB_URLS,
                        status=PipelineStatus.NEW,
                        is_active=True,
                        notes=None,
                    )
                    persist_pipeline_state_to_duckdb(new_state)
                    logger.info(f"✅ Initialized pipeline state for URL: {url}")

                else:
                    logger.info(f"🔁 URL already initialized: {url}")

            except Exception as e:
                logger.error(f"❌ Failed to initialize URL {url}: {e}", exc_info=True)

    def skip(self, urls: list[str], reason: str = ""):
        self._bulk_update(
            urls,
            fields={"status": PipelineStatus.SKIPPED, "is_active": False},
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
        self._bulk_update(
            urls,
            fields={
                "last_stage": restart_stage,
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

                persist_pipeline_state_to_duckdb(state)
                logger.info(f"✅ Updated state for {url}: {fields}")
            except Exception as e:
                logger.error(f"❌ Failed to update {url}: {e}", exc_info=True)

    def step_fsm(self, url: str):
        """
        Explicitly advance FSM for a URL.
        """
        fsm = self.fsm_manager.get_fsm(url)
        fsm.step()
