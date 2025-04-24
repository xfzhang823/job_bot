"""
fsm/pipeline_fsm.py

Finite State Machine (FSM) for managing pipeline stage progression.

This module implements a state machine to control sequential advancement through
a predefined pipeline workflow. Key features:

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
from transitions import Machine
from db_io.state_sync import persist_pipeline_state_to_duckdb
from db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from src.models.duckdb_table_models import PipelineState

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
        PipelineStage.CROSSTAB_REVIEW.value,
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

    This controller wraps a `PipelineState` model and tracks a job's current stage,
    providing stepwise advancement through a linear sequence of well-defined stages
    (e.g., job_urls → job_postings → extracted_requirements → ... → final_export).

    After each successful transition, the FSM updates the associated `PipelineState` record
    and persists it to the DuckDB `pipeline_control` table.

    ---
    * Workflow:
        1. Initialize with a `PipelineState` object (defaults to the first stage if unset).
        2. Call `.step()` to advance from the current stage to the next defined stage.
        3. After advancing, the `PipelineState` is updated with the new stage and timestamp.
        4. FSM persists the state to DuckDB using `save_pipeline_state_to_duckdb`.
        5. If the job is already at the final stage, the FSM does not advance further.

    ---
    Class-level Constants:
        STAGES (list[str]): Ordered list of atomic pipeline stages.
        TRANSITIONS (list[dict]): Advance transitions between each stage
        (used by `transitions` Machine).

    ---
    Instance Fields:
        state_model (PipelineState): Pydantic model containing job metadata and
        pipeline tracking info.
        _state (str): Internal tracker of the FSM's current stage
        (mirrors state_model.last_stage).

    ---
    Key Methods:
        step(): Advance to the next stage and persist updated state.
        get_current_stage(): Returns the current stage from state_model or internal state.
        get_next_stage(): Returns the next stage based on the defined transition map.

    ---
    Example:
    >>>    fsm = PipelineFSM(state)
    >>>    fsm.step()  # Moves from current stage to the next one and updates DuckDB
    >>>    fsm.mark_status(PipelineStatus.IN_PROGRESS, notes="Requirements parsed and stored.")

    """

    STAGES = _get_stage_names()
    TRANSITIONS = get_transitions(STAGES)

    def __init__(self, state: PipelineState):
        """
        Initialize the FSM with the provided PipelineState.

        Args:
            state (PipelineState): The Pydantic model representing the current
            pipeline state.
        """
        self.state_model = state
        self._state = state.last_stage or PipelineStage.JOB_URLS.value
        self.machine = Machine(
            model=self,
            states=self.STAGES,
            transitions=self.TRANSITIONS,
            initial=self._state,
            auto_transitions=False,
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
        return self.state_model.last_stage or self._state

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
        table_name: str = TableName.PIPELINE_CONTROL.value,
    ):
        """
        Update the pipeline status and optional notes for the current pipeline stage,
        without advancing the stage.

        Args:
            - status (PipelineStatus): Enum representing job status (e.g. IN_PROGRESS, ERROR).
            - notes (str): Optional message to attach to the current pipeline state.
            table_name (str): DuckDB table to persist update (default: pipeline_control).
        """
        try:
            self.state_model.status = status.value
            self.state_model.notes = notes
            self.state_model.timestamp = datetime.now()

            persist_pipeline_state_to_duckdb(self.state_model, table_name)
            logger.info(
                f"📝 Status updated for {self.state_model.url}: {status} — {notes}"
            )
        except Exception as e:
            logger.error(
                f"❌ Failed to update status for {self.state_model.url}: {e}",
                exc_info=True,
            )

    def step(self, table_name: str = TableName.PIPELINE_CONTROL.value):
        """
        Advance to the next stage and persist the update.
        Does not modify status or notes.
        """
        try:
            if self._state != PipelineStage.list()[-1]:
                logger.info(
                    f"⏩ Advancing {self.state_model.url} from stage '{self._state}'"
                )

                self._advance()
                self._state = self.state
                self.state_model.last_stage = self._state
                self.state_model.timestamp = datetime.now()

                persist_pipeline_state_to_duckdb(self.state_model, table_name)
                logger.info(f"✅ Advanced to: {self.state_model.last_stage}")
            else:
                logger.info(f"✅ Already at final stage: {self.state_model.url}")
        except Exception as e:
            logger.error(
                f"❌ Failed to advance stage for {self.state_model.url}: {e}",
                exc_info=True,
            )
