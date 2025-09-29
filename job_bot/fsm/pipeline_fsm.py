"""
fsm/pipeline_fsm.py

Finite State Machine (FSM) for managing pipeline stage progression.

This module implements a state machine to control sequential advancement through
a predefined pipeline workflow.

Key features:

1. Linear Stage Progression:
   - Defined stages: JOB_URLS → JOB_POSTINGS → ... → FINAL_RESPONSIBILITIES
   - Strictly sequential transitions via 'advance' trigger
   - Automatic persistence to DuckDB after each transition

2. Core Components:
   - PipelineFSM: Main controller class wrapping transitions.Machine
   - State tracking via PipelineState Pydantic model
   - Automatic timestamping of state changes

3. Key Operations:
   - step(): Advance to next stage and persist
   - get_current_stage(): Check current position
   - get_next_stage(): Preview upcoming stage

Usage:
    state = PipelineState(url="...")
    fsm = PipelineFSM(state)
    fsm.step()  # Advances pipeline and saves state


    * Example Walk-Through Flowchart:
    +-------------------+
    |   pipeline_control|
    |   (DuckDB table)  |
    +-------------------+
            |
            | Initial insert (status='new', stage='job_urls')
            v
    +-----------------------------+
    | url: https://job.com/abc123|
    | stage: job_urls            |
    | status: new                |
    +-----------------------------+

                    |
                    |   job_postings pipeline runs
                    |   └─ queries where stage = 'job_urls'
                    |   └─ scrapes content
                    |   └─ FSM: fsm.step() ➝ 'job_postings'
                    v

    +-----------------------------+
    | url: https://job.com/abc123|
    | stage: job_postings        |
    | status: new                |
    +-----------------------------+

                    |
                    |   extracted_requirements pipeline runs
                    |   └─ queries where stage = 'job_postings'
                    |   └─ extracts requirements
                    |   └─ FSM: fsm.step() ➝ 'extracted_requirements'
                    v

    +-------------------------------+
    | url: https://job.com/abc123   |
    | stage: extracted_requirements |
    | status: new                   |
    +-------------------------------+

Note: Designed for forward-only progression through irreversible stages.
"""

# Imports
import logging
from datetime import datetime
from typing import Optional

# User defined
from job_bot.db_io.state_sync import update_and_persist_pipeline_state
from job_bot.db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from job_bot.models.db_table_models import PipelineState

logger = logging.getLogger(__name__)


def _get_stage_names():
    return [
        PipelineStage.JOB_URLS.value,
        PipelineStage.JOB_POSTINGS.value,
        PipelineStage.EXTRACTED_REQUIREMENTS.value,
        PipelineStage.FLATTENED_REQUIREMENTS.value,
        PipelineStage.FLATTENED_RESPONSIBILITIES.value,
        PipelineStage.EDITED_RESPONSIBILITIES.value,
        PipelineStage.SIM_METRICS_EVAL.value,
        PipelineStage.SIM_METRICS_REVAL.value,
        PipelineStage.ALIGNMENT_REVIEW.value,
        PipelineStage.FINAL_RESPONSIBILITIES.value,
    ]


def get_transitions(stages: list[str]) -> list[dict]:
    """
    Generate linear transition mappings for an ordered list of pipeline stages.

    Each transition moves from one stage to the next in sequence using the trigger 'advance',
    suitable for use with the `transitions` state machine library.

    * Note: {"trigger": "advance", "source": "editing", "dest": "revaluation"} are
    * standard parameters recognized by the transitions library.

    Example:
        Given stages = ["job_urls", "job_postings", "extracted_requirements"],
        the returned transitions will be:
            [
                {"trigger": "advance", "source": "job_urls", "dest": "job_postings"},
                {"trigger": "advance", "source": "job_postings", "dest": "extracted_requirements"},
            ]

    Args:
        stages (list[str]): Ordered stage names representing pipeline progression.

    Returns:
        list[dict]: Transition rules in the format expected by the `transitions`
        Machine.
    """
    return [
        {"trigger": "advance", "source": s, "dest": d}
        for s, d in zip(stages, stages[1:])
    ]


class PipelineFSM:
    """
    A finite state machine (FSM) controller for managing job progress through
    atomic pipeline stages.

    This controller wraps a `PipelineState` model and tracks a job's current
    `stage` and `status`, providing stepwise advancement through a linear sequence
    of well-defined stages
    (e.g., job_urls → job_postings → extracted_requirements → … → final_export).

    After each successful transition, the FSM updates the associated
    `PipelineState` record and persists it to the DuckDB `pipeline_control` table.

    ---
    Workflow:
        1. Initialize with a `PipelineState` object (defaults to the first stage
           if unset).
        2. Call `.step()` to advance from the current stage to the next defined stage.
        3. After advancing, the `PipelineState` is updated with the new stage,
           status, and `updated_at` timestamp.
        4. FSM persists the state via the upsert helper
           (`update_and_persist_pipeline_state`).
        5. If the job is already at the final stage, the FSM does not advance further.

    ---
    Class-level Constants:
        STAGES (list[PipelineStage]): Ordered list of atomic pipeline stages.
        TRANSITIONS (list[dict]): Advance transitions between each stage
        (used by the FSM engine).

    ---
    Instance Fields:
        state_model (PipelineState): Pydantic model containing job metadata and
            pipeline tracking info.
        _state (PipelineStage): Internal tracker of the FSM's current stage
            (mirrors `state_model.stage`).
        _status (PipelineStatus): Internal tracker of the FSM's current status
            (mirrors `state_model.status`).

    ---
    Key Methods:
        step(): Advance to the next stage and persist updated state.
        get_current_stage(): Returns the current stage (enum).
        get_next_stage(): Returns the next stage based on the transition map.
        mark_status(): Set a new status and persist the change.

    ---
    Example:
        >>> fsm = PipelineFSM(state)
        >>> fsm.step()  # Moves from current stage to the next one and updates DuckDB
        >>> fsm.mark_status(PipelineStatus.IN_PROGRESS,
        ...                 notes="Requirements parsed and stored.")
    """

    STAGES = _get_stage_names()
    TRANSITIONS = get_transitions(STAGES)

    def __init__(self, state: PipelineState):
        """
        Initialize the FSM with a given PipelineState.

        The initializer stores the control-plane record (`stage`, `status`)
        as strongly typed enums (`PipelineStage`, `PipelineStatus`) and keeps
        a reference to the full Pydantic model for persistence.

        Args:
            state (PipelineState):
                The current pipeline state record loaded from the
                `pipeline_control` table.
        """
        # New control-plane fields: stage/status (enums)
        self.state_model = state
        # Normalize to PipelineStage enum internally
        self._state: PipelineStage = (
            state.stage
            if isinstance(state.stage, PipelineStage)
            else PipelineStage(str(state.stage))
        )
        self._status: PipelineStatus = (
            state.status
            if isinstance(state.status, PipelineStatus)
            else PipelineStatus(str(state.status))
        )

    @property
    def state(self) -> str:
        """Current state of the FSM."""
        return self._state

    def _advance(self) -> None:
        """Advance to the next state."""
        pass  # Transitions will inject this method

    def get_current_stage(self) -> str:
        """
        Get the current stage from the pipeline state.

        Returns:
            str: The current pipeline stage.
        """
        return self._state

    def get_next_stage(self) -> str | None:
        """
        Get the next stage in the pipeline based on the current stage.

        Returns:
            str | None: The next pipeline stage or None if at final stage.
        """
        try:
            for t in self.TRANSITIONS:
                if t["source"] == self._state:
                    return t["dest"]
            return None
        except Exception as e:
            logger.warning(
                f"⚠️ Could not determine next stage from '{self._state}': {e}"
            )
            return None

    def mark_status(
        self,
        status: PipelineStatus,
        notes: Optional[str] = None,
        table_name: TableName = TableName.PIPELINE_CONTROL,
    ) -> None:
        """
        Update the pipeline status (without advancing the stage) and persist.

        Args:
            status: New PipelineStatus (e.g., IN_PROGRESS, ERROR, SKIPPED).
            notes: Optional free-text note to append (timestamped).
            table_name: Control-plane table name (default: pipeline_control).
        """
        try:
            # Update internal and model status as enums
            self._status = status
            self.state_model.status = status

            # Append timestamped note (preserve prior notes)
            if notes:
                line = f"[{datetime.now().isoformat(timespec='seconds')}] {notes}"
                self.state_model.notes = (
                    f"{line}\n{self.state_model.notes}".strip()
                    if self.state_model.notes
                    else line
                )

            # Touch updated_at only (created_at is immutable after insert)
            self.state_model.updated_at = datetime.now()

            update_and_persist_pipeline_state(self.state_model, table_name)

            logger.info(
                "📝 Status updated for %s: %s — %s",
                self.state_model.url,
                status.name,
                notes or "",
            )
        except Exception as e:
            logger.error(
                "❌ Failed to update status for %s: %s",
                self.state_model.url,
                e,
                exc_info=True,
            )

    def step(self, table_name: TableName = TableName.PIPELINE_CONTROL) -> None:
        """
        Advance to the next stage and persist the update.
        Does not modify status (only `stage` and `updated_at`).
        """
        try:
            final_stage = PipelineStage.list()[-1]  # relies on your enum utility
            if self._state == final_stage:
                logger.info(
                    "✅ Already at final stage for %s: %s",
                    self.state_model.url,
                    self._state.name,
                )
                return

            logger.info(
                "⏩ Advancing %s from '%s'", self.state_model.url, self._state.name
            )

            # Advance internal FSM (your implementation should set the next stage)
            self._advance()  # assumes this updates self._state internally

            # Mirror internal stage to the persisted model
            self.state_model.stage = self._state
            self.state_model.updated_at = datetime.now()

            update_and_persist_pipeline_state(self.state_model, table_name)
            logger.info("✅ Advanced to: %s", self.state_model.stage.name)
        except Exception as e:
            logger.error(
                "❌ Failed to advance stage for %s: %s",
                self.state_model.url,
                e,
                exc_info=True,
            )
