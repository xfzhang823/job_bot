"""
fsm/pipeline_fsm_manager.py

Provides a high-level interface for loading, inspecting, and advancing
pipeline FSM states stored in the DuckDB `pipeline_control` table.

This manager encapsulates:
- Fetching the latest active `PipelineState` row for a given URL.
- Validating and converting DuckDB rows into Pydantic `PipelineState` models.
- Constructing `PipelineFSM` instances for state-transition logic.
- Querying the current stage without manual SQL.
"""

# Standard
from pathlib import Path
from typing import Optional, Union

# From project modules
from job_bot.db_io.get_db_connection import get_db_connection
from job_bot.db_io.pipeline_enums import TableName, PipelineStage
from job_bot.fsm.pipeline_fsm import PipelineFSM
from job_bot.models.db_table_models import PipelineState


class PipelineFSMManager:
    """
    PipelineFSMManager

    A utility wrapper for reading the `pipeline_control` table in DuckDB.
    Responsibilities:
       • Retrieve the most recent `PipelineState` for a URL (by updated_at/created_at).
       • Convert raw rows to validated Pydantic models.
       • Construct and return `PipelineFSM` instances.

    FSM transition rules are implemented by `PipelineFSM`; this manager only loads state.
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        table_name: str = TableName.PIPELINE_CONTROL.value,
    ):
        """
        Initialize the FSM manager.

        Args:
            db_path (str | Path | None):
                Optional filesystem path to the DuckDB database file.
                If None, uses the default configured path.
            table_name (str):
                Name of the DuckDB table that holds pipeline states.
                Defaults to the value of TableName.PIPELINE_CONTROL.
        """
        self.conn = get_db_connection(db_path)
        self.table_name = table_name

    def get_state_model(self, url: Union[str, Path]) -> PipelineState:
        """
        Load and return the latest PipelineState for a given URL.

        Ordering uses the most recent non-null timestamp:
          ORDER BY COALESCE(updated_at, created_at) DESC

        Args:
            url (str | Path):
                The job posting URL whose state we want to fetch.

        Returns:
            PipelineState:
                A fully-validated Pydantic model representing the current state.

        Raises:
            ValueError:
                If no active state row is found for the given URL.

        Example:
            >>> manager = PipelineFSMManager()
            >>> state = manager.get_state_model("https://example.com/job/123")
            >>> print(state.stage, state.status)
        """
        url_str = str(url)
        query = f"""
            SELECT *
            FROM {self.table_name}
            WHERE url = ?
            ORDER BY COALESCE(updated_at, created_at) DESC
            LIMIT 1
        """
        df = self.conn.execute(query, [url_str]).fetchdf()
        if df.empty:
            raise ValueError(f"No pipeline state found for URL: {url_str}")

        record = df.iloc[0].to_dict()
        return PipelineState.model_validate(record)

    def get_fsm(self, url: Union[str, Path]) -> PipelineFSM:
        """
        Construct and return a PipelineFSM for the given URL.
        Loads current state via `get_state_model` and wraps it.

        Args:
            url (str | Path):
                The job posting URL for which to build the FSM.

        Returns:
            PipelineFSM:
                An FSM object initialized at the current state.

        Example:
            >>> manager = PipelineFSMManager()
            >>> fsm = manager.get_fsm("https://example.com/job/123")
            >>> fsm.get_current_stage()
            'job_postings'
        """
        state_model = self.get_state_model(url)
        return PipelineFSM(state_model)

    def get_stage(self, url: Union[str, Path]) -> PipelineStage:
        """
        Return the current pipeline stage (PipelineStage enum).

        Args:
            url (str | Path):
                The job posting URL whose current stage we want.

        Returns:
            PipelineStage:
                The current pipeline stage, as defined in `pipeline_enums.PipelineStage`.
        """
        fsm = self.get_fsm(url)
        stage = fsm.get_current_stage()
        # If get_current_stage() already returns a PipelineStage, just return it.
        return stage if isinstance(stage, PipelineStage) else PipelineStage(stage)
