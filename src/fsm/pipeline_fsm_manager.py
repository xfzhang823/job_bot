"""pipeline_fsm_manager.py"""

from pathlib import Path
from typing import Optional, Union
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.db_schema_registry import TableName
from fsm.pipeline_fsm import PipelineFSM
from models.duckdb_table_models import PipelineState


class PipelineFSMManager:
    """
    PipelineFSMManager

    A utility wrapper for managing and interacting with the pipeline control table
    stored in DuckDB. This class provides a convenient interface for:

    - Retrieving the latest pipeline state for a given URL.
    - Constructing a `PipelineFSM` instance from a validated `PipelineState` row.
    - Querying the current stage of a job (e.g., 'preprocessing', 'staging', etc.).
    - Delegating FSM control without manually handling DB queries.

    This manager does NOT define FSM logic itself — it loads state and delegates
    transitions to the `PipelineFSM` class, which defines valid states and transitions.

    Methods:
        - get_state_model(url): Returns the latest `PipelineState` for a given job URL.
        - get_fsm(url): Constructs and returns a `PipelineFSM` instance.
        - get_stage(url): Returns the current FSM stage for the URL.

    Example:
        >>> manager = PipelineFSMManager("pipeline_state.duckdb")
        >>> fsm = manager.get_fsm(url)
        >>> current_stage = fsm.get_current_stage()

        if current_stage == PipelineStage.PREPROCESSING:
            ...
            fsm.step()  # Advance to next stage (e.g., 'staging')
    """

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        table_name: str = TableName.PIPELINE_CONTROL.value,
    ):
        self.conn = get_duckdb_connection(db_path)
        self.table_name = table_name

    def get_state_model(self, url: Union[str, Path]) -> PipelineState:
        """
        Load the most recent PipelineState for a given URL.

        Args:
            url (str | Path): Job posting URL.

        Returns:
            PipelineState: Most recent pipeline state model.
        """
        url_str = str(url)
        query = f"""
        SELECT * FROM {self.table_name}
        WHERE url = ? AND is_active = TRUE
        ORDER BY last_updated DESC
        LIMIT 1
        """
        result = self.conn.execute(query, [url_str]).fetchdf()
        if result.empty:
            raise ValueError(f"No pipeline state found for URL: {url_str}")

        return PipelineState.model_validate(result.iloc[0].to_dict())

    def get_fsm(self, url: Union[str, Path]) -> PipelineFSM:
        """
        Construct and return a PipelineFSM instance for the given job URL.

        Args:
            url (str | Path): Job posting URL.

        Returns:
            PipelineFSM: Initialized FSM instance.
        """
        state_model = self.get_state_model(url)
        return PipelineFSM(state_model)

    def get_stage(self, url: Union[str, Path]) -> str:
        """
        Return the current stage for the given job URL.

        Args:
            url (str | Path): Job posting URL.

        Returns:
            str: Current pipeline stage (e.g., "preprocessing").
        """
        return self.get_fsm(url).get_current_stage()
