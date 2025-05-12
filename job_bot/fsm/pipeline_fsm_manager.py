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

from pathlib import Path
from typing import Optional, Union

from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import TableName
from db_io.pipeline_enums import PipelineStage
from fsm.pipeline_fsm import PipelineFSM
from models.duckdb_table_models import PipelineState


class PipelineFSMManager:
    """
    PipelineFSMManager

    A utility wrapper for managing and interacting with the `pipeline_control` table
    in DuckDB. Responsibilities include:

      - Retrieving the most recent active `PipelineState` for a job URL.
      - Converting raw DuckDB rows into validated Pydantic models.
      - Constructing and returning `PipelineFSM` instances.
      - Exposing helper methods for querying current stage or stepping the FSM.

    This class does NOT itself implement the FSM transitions — it delegates
    that to `PipelineFSM`, which enforces valid stage steps and persistence.
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
        self.conn = get_duckdb_connection(db_path)
        self.table_name = table_name

    def get_state_model(self, url: Union[str, Path]) -> PipelineState:
        """
        Load and return the latest active PipelineState for a given URL.

        This method performs a SQL query against the `pipeline_control`
        table, filters by URL and `is_active = TRUE`, orders by
        `timestamp DESC`, and returns the first result as a Pydantic model.

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
        SELECT * FROM {self.table_name}
        WHERE url = ? AND is_active = TRUE
        ORDER BY timestamp DESC
        LIMIT 1
        """
        df = self.conn.execute(query, [url_str]).fetchdf()
        if df.empty:
            raise ValueError(f"No pipeline state found for URL: {url_str}")

        record = df.iloc[0].to_dict()
        return PipelineState.model_validate(record)

    def get_fsm(self, url: Union[str, Path]) -> PipelineFSM:
        """
        Construct and return a PipelineFSM for the given job URL.

        This method loads the current PipelineState via `get_state_model`,
        then wraps it in a `PipelineFSM` instance which provides `step()`,
        `get_current_stage()`, and other transition methods.

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
        Return the current pipeline stage as a PipelineStage enum.

        This method delegates to the PipelineFSM instance to fetch the
        raw stage string, then casts it to the PipelineStage enum.

        Args:
            url (str | Path):
                The job posting URL whose current stage we want.

        Returns:
            PipelineStage:
                The current pipeline stage, as defined in `pipeline_enums.PipelineStage`.
        """
        raw_stage = self.get_fsm(url).get_current_stage()
        return PipelineStage(raw_stage)
