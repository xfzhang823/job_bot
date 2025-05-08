# fsm_utils.py

# TODO: more advanced; do it later

from typing import Literal
from fsm.pipeline_fsm import PipelineFSM
from src.models.duckdb_table_models import PipelineState
from db_io.state_sync import load_pipeline_state, upsert_pipeline_state_to_duckdb


# ✅ Aliases to use throughout your code
VersionType = Literal["original", "edited", "final"]
StatusType = Literal["new", "in_progress", "complete", "skipped", "error"]


def advance_fsm_for_url(
    url: str,
    llm_provider: str = "openai",
    iteration: int = 0,
    version: VersionType = "original",
    status: StatusType = "new",  # Optional
):
    """
    Advances the pipeline FSM for a given URL and persists the updated state
    into the `pipeline_control` DuckDB table.

    This function performs the following steps:
    1. Attempts to load the current `PipelineState` for the given URL from DuckDB.
    2. If no existing state is found, it initializes a new `PipelineState` with default
       metadata, including the URL, LLM provider, iteration number, version, and status.
    3. Creates a `PipelineFSM` instance using the current state.
    4. Calls `fsm.step()` to advance the pipeline to the next defined stage in the
       workflow (e.g., from "preprocessing" to "staging").
    5. Persists the updated state back into DuckDB, either inserting or updating
       the corresponding row in the `pipeline_control` table.

    This function is designed to be invoked at the end of a successful processing step
    (e.g., after data insertion, metric computation, or LLM editing), so that the pipeline
    controller remains synchronized with progress.

    Args:
        url (str): The job posting URL that uniquely identifies the job being processed.
        llm_provider (str): The name of the LLM provider used for this job (e.g., "openai").
        iteration (int): The iteration number of the current pipeline run. Used to track
            reruns or multi-pass refinement.
        version (VersionType): The type of content version being processed, such as
            "original", "edited", or "final".
        status (StatusType): The initial status for new pipeline states. Defaults to "new".

    Example:
        >>> advance_fsm_for_url(
        ...     url="https://jobs.example.com/posting123",
        ...     llm_provider="anthropic",
        ...     iteration=1,
        ...     version="edited"
        ... )

    Raises:
        Any exception raised during state persistence or FSM step advancement will propagate.
    """

    state = load_pipeline_state(url) or PipelineState(
        url=url,
        llm_provider=llm_provider,
        iteration=iteration,
        version=version,
        status=status,
    )
    fsm = PipelineFSM(state)
    fsm.step(table_name="pipeline_control")
