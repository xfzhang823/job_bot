"""
state_orchestration_pipeline.py

This module serves as the central orchestrator for managing job postings pipeline state
progression using a Finite State Machine (FSM). It coordinates initialization,
state transitions, FSM integrity checks, and pipeline state control, ensuring that each
job posting URL is accurately tracked and progresses smoothly through each defined stage.

---

🛠️ Core Responsibilities:

1. **Pipeline State Initialization:**
   - Initializes FSM states for new URLs, setting default stages and metadata.
   - Ensures each job URL is prepared for FSM-driven progression.

2. **FSM Integrity Checks:**
   - Regularly validates FSM state transitions and detects potential inconsistencies.
   - Ensures that the state of each job URL strictly adheres to allowed FSM transitions.

3. **FSM State Control and Management:**
   - Provides centralized control for state updates, bulk operations, and explicit
     FSM stepping.
   - Enables batch updates of statuses and FSM stages, simplifying administration.

4. **High-Level FSM Orchestration:**
   - Coordinates overall pipeline stage execution based on FSM state.
   - Interfaces with existing pipeline execution logic (`run_pipeline.py`,
   `pipeline_fsm_manager.py`).

---

🔄 High-Level Process Flow:

```
1. Initialize URLs:
   - Gather URLs to process.
   - Initialize FSM states via `fsm_state_control.py`.

2. Run Integrity Checks:
   - Validate FSM transitions and consistency via `fsm_integrity_checker.py`.
   - Address any flagged issues before proceeding.

3. Trigger FSM-based Pipeline Execution:
   - Query current FSM states from DuckDB (`pipeline_control` table).
   - Identify URLs eligible for each pipeline stage (e.g., job_postings,
   extracted_requirements).
   - Execute respective pipeline modules based on FSM stage/status.

4. FSM State Advancement:
   - Upon successful completion of each stage for each URL, explicitly advance
   FSM states.
   - Persist FSM updates into DuckDB to ensure accurate tracking.

5. Repeat Process:
   - Regularly run FSM integrity checks between stages.
   - Continuously update, manage, and progress URLs through pipeline stages.
```

---

📦 Module Usage:

```python
# Typical usage from main orchestrator
from fsm.fsm_integrity_checker import validate_fsm_integrity
from fsm.fsm_state_control import FSMStateControl

class StateOrchestrator:

    def __init__(self):
        self.control = FSMStateControl()

    def initialize_pipeline(self, urls: list[str]):
        self.control.initialize_urls(urls)
        validate_fsm_integrity()

    def execute_pipeline_stages(self):
        # Call appropriate pipeline execution functions based on FSM stage
        pass  # logic to query FSM and execute pipeline stages

    def advance_states(self, urls: list[str]):
        for url in urls:
            self.control.step_fsm(url)

# Run orchestration
orchestrator = StateOrchestrator()
orchestrator.initialize_pipeline(["https://job1.com", "https://job2.com"])
orchestrator.execute_pipeline_stages()
```

---

🚩 Scalability and Future Extensions:

- **LLM-Based FSM Routing (`llm_fsm.py`):**
  - Integrate an optional LLM-based decision engine to dynamically decide FSM transitions
    based on content, state, or analytics.

- **Automated Integrity Checking:**
  - Set up periodic integrity checks as cron jobs or background processes.

- **Advanced Analytics:**
  - Extend analytics (`fsm_analytics.py`) to visualize FSM progression, detect bottlenecks,
    and enhance operational insights.

---

⚙️ Integration Points with Existing Modules:

- `pipeline_fsm_manager.py`: Detailed FSM transition logic per URL.
- `run_pipeline.py`: General-purpose pipeline execution logic.
- `db_io`: FSM state persistence (DuckDB).
- `models`: FSM state validation (Pydantic).

---

📌 Module Dependencies:

- **Internal:**
  - `fsm.fsm_integrity_checker`
  - `fsm.fsm_state_control`
  - `fsm.pipeline_fsm_manager`
  - `db_io.state_sync`
  - `db_io.schema_definitions`

- **External:**
  - `DuckDB` (state persistence)
  - `Pydantic` (state validation)

---

This structured and FSM-centric orchestration ensures robust, transparent,
and scalable pipeline execution, laying a clear foundation for future
enhancements like LLM-driven state management.

"""

# from fsm.llm_fsm import LLMBasedFSMRouter (Future)

from datetime import datetime
import pandas as pd
from typing import Optional, Dict, Any
import logging

# Project specific imports
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.create_db_tables import create_single_db_table
from db_io.state_sync import upsert_pipeline_state_to_duckdb
from db_io.db_utils import get_urls_by_stage_and_status, get_pipeline_state
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from db_io.pipeline_enums import PipelineStage, PipelineStatus, TableName
from db_io.state_sync import upsert_pipeline_state_to_duckdb

from fsm.fsm_integrity_checker import validate_fsm_integrity
from fsm.fsm_state_control import PipelineFSMManager
from models.duckdb_table_models import PipelineState

logger = logging.getLogger(__name__)


def sync_table_to_pipeline_control(
    source_table: TableName,
    defaults: Optional[Dict[str, Any]] = None,
    status: PipelineStatus = PipelineStatus.NEW,
    is_active: bool = True,
    notes: Optional[str] = None,
):
    """
    Generic function to sync any source table to pipeline_control table.
    This function reads records from a given source table and ensures that
    corresponding entries exist in the `pipeline_control` table.

    Args:
        source_table (TableName): Enum value for the source table.
        defaults (Optional[Dict[str, Any]]): Default values to fill missing fields.
        status (PipelineStatus): Default pipeline status.
        is_active (bool): Whether the row is active.
        notes (Optional[str]): Optional note field.
    """
    con = get_duckdb_connection()

    source_schema = DUCKDB_SCHEMA_REGISTRY.get(source_table)
    pipeline_schema = DUCKDB_SCHEMA_REGISTRY.get(TableName.PIPELINE_CONTROL)

    if not source_schema or not pipeline_schema:
        logger.error(f"❌ Schema not found for {source_table} or pipeline_control.")
        return

    source_fields = set(source_schema.model.model_fields.keys())
    pipeline_fields = set(pipeline_schema.model.model_fields.keys())

    missing_fields = pipeline_fields - source_fields
    select_columns = ", ".join(source_fields)

    query = f"SELECT DISTINCT {select_columns} FROM {source_table.value}"

    try:
        records = con.execute(query).fetchdf().to_dict(orient="records")
    except Exception as e:
        logger.error(f"❌ Error querying {source_table.value}: {e}")
        return

    base_defaults = {
        "status": status.value,
        "is_active": is_active,
        "notes": notes or "",
        "timestamp": datetime.now(),
    }

    if defaults:
        base_defaults.update(defaults)

    synced_count = 0
    for record in records:
        state_data = {field: record.get(field) for field in source_fields}
        for field in missing_fields:
            state_data[field] = base_defaults.get(field)

        try:
            pipeline_state = PipelineState(**state_data)
            upsert_pipeline_state_to_duckdb(pipeline_state)
            synced_count += 1
        except Exception as e:
            logger.warning(
                f"⚠️ Failed to upsert state for URL '{record.get('url')}': {e}"
            )

    logger.info(
        f"✅ Synced {synced_count} records from '{source_table.value}' to pipeline_control."
    )


# Helper functions for each table sync with enum-aligned stage/version values
def sync_job_urls_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_URLS,
        defaults={
            "stage": PipelineStage.JOB_URLS.value,
            "version": "original",
            "llm_provider": None,
        },
    )


def sync_job_postings_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_POSTINGS,
        defaults={
            "stage": PipelineStage.JOB_POSTINGS.value,
            "version": "original",
            "llm_provider": None,
        },
    )


def sync_extracted_requirements_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.EXTRACTED_REQUIREMENTS,
        defaults={
            "stage": PipelineStage.EXTRACTED_REQUIREMENTS.value,
            "version": "original",
            "llm_provider": None,
        },
    )


def sync_flattend_requirements_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_REQUIREMENTS,
        defaults={
            "stage": PipelineStage.FLATTENED_REQUIREMENTS.value,
            "version": "original",
            "llm_provider": None,
        },
    )


def sync_flattened_responsibilities_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_RESPONSIBILITIES,
        defaults={
            "stage": PipelineStage.FLATTENED_RESPONSIBILITIES.value,
            "version": "original",
            "llm_provider": None,
        },
    )


def sync_edited_responsibilities_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.EDITED_RESPONSIBILITIES,
        defaults={
            "stage": PipelineStage.EDITED_RESPONSIBILITIES.value,
            "version": "edited",
            "llm_provider": "openai",
        },
    )


def sync_similarity_metrics_to_pipeline_control():
    sync_table_to_pipeline_control(
        source_table=TableName.SIMILARITY_METRICS,
        defaults={
            "stage": PipelineStage.SIM_METRICS_EVAL.value,
            "version": "original",
            "llm_provider": "openai",
        },
    )


def run_state_orchestration_pipeline():
    """
    Orchestrates syncing all pipeline stages to the `pipeline_control` table.
    This is the entry point for setting up or recovering FSM state.
    """
    logger.info("🧭 Starting pipeline state orchestration...")
    create_single_db_table(TableName.PIPELINE_CONTROL)

    sync_job_urls_to_pipeline_control()
    # sync_job_postings_to_pipeline_control()
    # sync_extracted_requirements_to_pipeline_control()
    # sync_flattend_requirements_to_pipeline_control()
    # sync_flattened_responsibilities_to_pipeline_control()
    # sync_edited_responsibilities_to_pipeline_control()
    # sync_similarity_metrics_to_pipeline_control()

    logger.info("✅ Pipeline control table sync complete.")


# Commented out (older version)
# def update_pipeline_control(
#     stage: PipelineStage,
#     source_file: str,
#     iteration: int,
#     version: str,
#     llm_provider: str,
#     status: PipelineStatus = PipelineStatus.NEW,
#     notes: Optional[str] = None,
# ):
#     """
#     Updates the control table for all URLs in a specific stage and iteration.

#     Args:
#         stage (PipelineStage): The pipeline stage being updated.
#         source_file (str): The source file associated with the update.
#         iteration (int): The iteration of the pipeline.
#         version (str): The version identifier.
#         llm_provider (str): The LLM provider.
#         status (PipelineStatus, optional): The status to set. Defaults to 'new'.
#         notes (Optional[str], optional): Additional notes to include.
#           Defaults to None.
#     """

#     # Fetch URLs for the specified stage and iteration
#     urls = get_urls_by_stage_and_status(
#         stage=stage, iteration=iteration, version=version
#     )

#     # Initialize empty list for pipeline state objects
#     pipeline_states = []

#     # Create new pipeline states based on the provided parameters
#     for url in urls:
#         # Fetch the existing state
#         state_df = get_pipeline_state(url)

#         # If the state exists, update it; otherwise, create a new one
#         if not state_df.empty:
#             # Update existing state
#             state_data = state_df.iloc[0].to_dict()
#             state_data.update(
#                 {
#                     "stage": stage.value,
#                     "source_file": source_file,
#                     "timestamp": datetime.now(),
#                     "version": version,
#                     "llm_provider": llm_provider,
#                     "iteration": iteration,
#                     "status": status.value,
#                     "notes": notes,
#                 }
#             )
#         else:
#             # Create new state
#             state_data = {
#                 "url": url,
#                 "stage": stage.value,
#                 "source_file": source_file,
#                 "timestamp": datetime.now(),
#                 "version": version,
#                 "llm_provider": llm_provider,
#                 "iteration": iteration,
#                 "status": status.value,
#                 "notes": notes,
#             }

#         # Convert to PipelineState and add to list
#         pipeline_state = PipelineState(**state_data)
#         pipeline_states.append(pipeline_state)

#     # Save all states to DuckDB
#     for state in pipeline_states:
#         upsert_pipeline_state_to_duckdb(state)

#     print(
#         f"✅ Updated {len(pipeline_states)} records in pipeline_control for stage '{stage.value}'."
#     )
