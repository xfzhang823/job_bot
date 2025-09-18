"""
pipelines_with_fsm/state_orchestration_pipeline.py

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
What this module does NOT do:
    - It does not execute the heavy stage work (scraping, LLM parsing, evaluation, editing).
    Those are run by the stage pipelines; this module keeps the control table accurate
    and consistent so those pipelines know what to work on next.

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
from typing import Optional, Dict, Any, Literal
from pydantic import ValidationError
import logging

# Project specific imports
from db_io.duckdb_adapter import get_duckdb_connection
from db_io.create_db_tables import create_single_db_table
from db_io.state_sync import upsert_pipeline_state_to_duckdb
from db_io.db_utils import get_urls_by_stage_and_status, get_pipeline_state
from db_io.db_schema_registry import DUCKDB_SCHEMA_REGISTRY
from db_io.pipeline_enums import (
    PipelineStage,
    PipelineStatus,
    TableName,
    LLMProvider,
    Version,
)
from db_io.state_sync import upsert_pipeline_state_to_duckdb

from fsm.fsm_integrity_checker import validate_fsm_integrity
from fsm.fsm_state_control import PipelineFSMManager
from models.duckdb_table_models import PipelineState

logger = logging.getLogger(__name__)


VALID_LLM_PROVIDERS = {"openai", "anthropic", "llama"}
DEFAULT_LLM_PROVIDER = "openai"


def sync_table_to_pipeline_control(
    source_table: TableName,
    stage: PipelineStage,
    defaults: Optional[Dict[str, Any]] = None,
    mode: Literal["append", "replace"] = "append",
    is_active: bool = True,
    notes: Optional[str] = None,
):
    """
    Sync rows from a **source stage table** into `pipeline_control`.

    Behavior
    --------
    - Reads distinct rows from `source_table` based on that table’s schema.
    - Builds a `PipelineState` per record and upserts into `pipeline_control`
      using the composite PK (url, iteration, version, llm_provider).
    - If `mode="replace"`, deletes any existing matching control rows before upserting.
    - Applies `defaults` and standard metadata (timestamp, is_active, etc.).
    - **Important:** the `status` field in `defaults` (or the module default) is
      always enforced on the upserted control rows.

    Parameters
    ----------
    source_table : TableName
        Stage/source table to sync from (e.g., JOB_POSTINGS).
    stage : PipelineStage
        Pipeline stage to assign in `pipeline_control` for these rows.
    defaults : dict, optional
        Default values to fill/override (e.g., llm_provider, version, status).
    mode : {"append","replace"}
        - "append": upsert without pre-delete
        - "replace": delete matching control rows first, then upsert
    is_active : bool
        Whether to mark control rows active.
    notes : str, optional
        Free-text notes stored on the control rows.
    """
    con = get_duckdb_connection()

    source_schema = DUCKDB_SCHEMA_REGISTRY.get(source_table)
    pipeline_schema = DUCKDB_SCHEMA_REGISTRY.get(TableName.PIPELINE_CONTROL)

    if not source_schema or not pipeline_schema:
        logger.error(f"Schema not found for {source_table} or pipeline_control.")
        return

    source_fields = set(source_schema.model.model_fields.keys())
    select_columns = ", ".join(source_fields)

    # Fetch source data
    try:
        records = (
            con.execute(f"SELECT DISTINCT {select_columns} FROM {source_table.value}")
            .fetchdf()
            .to_dict(orient="records")
        )
    except Exception as e:
        logger.error(f"Error querying {source_table.value}: {e}")
        return

    # Base defaults
    base_defaults = {
        "stage": stage,  # now use the Enum directly
        "status": PipelineStatus.IN_PROGRESS,
        "is_active": is_active,
        "notes": notes,
        "timestamp": datetime.now(),
        "llm_provider": DEFAULT_LLM_PROVIDER,
        "version": "original",
    }

    # Merge in any overrides
    if defaults:
        base_defaults.update(defaults)

    synced_count = 0

    for record in records:
        # Start with whatever came from the source
        state_data: Dict[str, Any] = {f: record.get(f) for f in source_fields}

        # Apply defaults, but always overwrite 'stage' so it matches our argument
        for field, value in base_defaults.items():
            if field == "stage" or state_data.get(field) is None:
                state_data[field] = value

        # Always enforce 'status' default
        state_data["status"] = base_defaults["status"]

        logger.debug(f"Final state data before model construction: {state_data}")

        # Build the Pydantic model (now expecting 'stage', not 'last_stage')
        try:
            pipeline_state = PipelineState(**state_data)
        except ValidationError as e:
            logger.warning(f"Data validation error for URL '{record.get('url')}': {e}")
            continue

        # Identify existing rows by the composite key
        where_clause = (
            f"url = '{pipeline_state.url}' AND "
            f"iteration = {pipeline_state.iteration} AND "
            f"version = '{pipeline_state.version}' AND "
            f"llm_provider = '{pipeline_state.llm_provider}'"
        )

        if mode == "replace":
            con.execute(f"DELETE FROM pipeline_control WHERE {where_clause};")

        # Upsert via your existing helper
        try:
            upsert_pipeline_state_to_duckdb(pipeline_state)
            synced_count += 1
        except Exception as e:
            logger.warning(
                f"Failed to upsert state for URL '{pipeline_state.url}': {e}"
            )

    logger.info(
        f"Synced {synced_count} records from '{source_table.value}' "
        f"to pipeline_control using '{mode}' mode."
    )


# Helper functions for each table sync with enum-aligned stage/version values
# def sync_job_urls_to_pipeline_control():
#     sync_table_to_pipeline_control(
#         source_table=TableName.JOB_URLS, stage=PipelineStage.JOB_URLS, mode="append"
#     )


def sync_job_urls_to_pipeline_control():
    """Sync `job_urls` into `pipeline_control` (seed NEW rows for initial FSM)."""
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_URLS,
        stage=PipelineStage.JOB_URLS,
        defaults={
            "stage": PipelineStage.JOB_URLS,
            "llm_provider": LLMProvider.OPENAI,  # enum, not string
            "status": PipelineStatus.NEW,  # enum, not NEW.value
        },
        mode="replace",
    )


def sync_job_postings_to_pipeline_control():
    """Sync `job_postings` into `pipeline_control` (mark IN_PROGRESS for postings stage)."""
    sync_table_to_pipeline_control(
        source_table=TableName.JOB_POSTINGS,
        stage=PipelineStage.JOB_POSTINGS,
        defaults={
            "stage": PipelineStage.JOB_POSTINGS,
            "version": Version.ORIGINAL,  # enum
            "llm_provider": LLMProvider.OPENAI,  # enum
            "status": PipelineStatus.IN_PROGRESS,  # enum
        },
        mode="replace",
    )


def sync_extracted_requirements_to_pipeline_control():
    """Sync `extracted_requirements` into `pipeline_control`."""
    sync_table_to_pipeline_control(
        source_table=TableName.EXTRACTED_REQUIREMENTS,
        stage=PipelineStage.EXTRACTED_REQUIREMENTS,
        defaults={
            "version": Version.ORIGINAL,
            "llm_provider": LLMProvider.OPENAI,
        },
        mode="replace",
    )


def sync_flattened_requirements_to_pipeline_control():
    """Sync `flattened_requirements` into `pipeline_control`."""
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_REQUIREMENTS,
        stage=PipelineStage.FLATTENED_REQUIREMENTS,
        defaults={
            "version": Version.ORIGINAL,
            "llm_provider": LLMProvider.OPENAI,
        },
        mode="replace",
    )


def sync_flattened_responsibilities_to_pipeline_control():
    """Sync `flattened_responsibilities` into `pipeline_control`."""
    sync_table_to_pipeline_control(
        source_table=TableName.FLATTENED_RESPONSIBILITIES,
        stage=PipelineStage.FLATTENED_RESPONSIBILITIES,
        defaults={
            "version": Version.ORIGINAL,
            "llm_provider": LLMProvider.OPENAI,
        },
        mode="replace",
    )


def sync_edited_responsibilities_to_pipeline_control():
    """Sync `edited_responsibilities` into `pipeline_control`
    (inherits version/provider where present)."""
    sync_table_to_pipeline_control(
        source_table=TableName.EDITED_RESPONSIBILITIES,
        stage=PipelineStage.EDITED_RESPONSIBILITIES,
        defaults={
            "version": None,  # Inherit from source data
            "llm_provider": None,  # Inherit from source data
        },
        mode="replace",
    )


def sync_similarity_metrics_to_pipeline_control(batch_size: int = 1000):
    """
    Sync **similarity_metrics** keys into `pipeline_control` using the registry PKs.

    Behavior
    --------
    - Loads batches of distinct primary keys from `similarity_metrics`.
    - Creates minimal `PipelineState` rows (using PKs + standard metadata) and
        upserts them.
    - Sets `stage = SIM_METRICS_EVAL` and `status = IN_PROGRESS` for these rows;
      status is not inherited from the source table.

    Parameters
    ----------
    batch_size : int
        Number of rows per SELECT batch.
    """
    logger.info("Syncing similarity metrics to pipeline_control using primary keys...")

    con = get_duckdb_connection()

    # Get primary keys dynamically from schema registry
    schema = DUCKDB_SCHEMA_REGISTRY.get(TableName.SIMILARITY_METRICS)
    if not schema:
        logger.error(f"Schema not found for {TableName.SIMILARITY_METRICS}")
        return

    primary_keys = schema.primary_keys

    # Ensure primary keys are defined
    if not primary_keys:
        logger.error(f"No primary keys defined for {TableName.SIMILARITY_METRICS}")
        return

    logger.info(f"Primary Keys: {primary_keys}")

    # Fetch the source data in batches
    offset = 0
    processed_count = 0

    while True:
        # Construct the SELECT query based on primary keys
        select_columns = ", ".join(primary_keys)
        query = f"""
            SELECT DISTINCT {select_columns}
            FROM similarity_metrics
            LIMIT {batch_size} OFFSET {offset}
        """

        try:
            df = con.execute(query).fetchdf()
        except Exception as e:
            logger.error(f"Error fetching similarity metrics: {e}")
            break

        if df.empty:
            break

        # Convert to records
        records = df.to_dict(orient="records")

        # Base defaults
        base_defaults = {
            "stage": PipelineStage.SIM_METRICS_EVAL.value,
            "status": PipelineStatus.IN_PROGRESS.value,  # ✅ Explicitly set
            "is_active": True,
            "notes": "",
            "timestamp": datetime.now(),
        }

        # Process each record
        for record in records:
            # Construct primary key clause dynamically
            primary_key_clause = " AND ".join(
                [f"{pk} = '{record[pk]}'" for pk in primary_keys]
            )

            # Prepare state data
            state_data = {pk: record.get(pk) for pk in primary_keys}

            # Apply base defaults
            for field, value in base_defaults.items():
                state_data[field] = value

            logger.debug(f"State data before model construction: {state_data}")

            # Construct the PipelineState object
            try:
                pipeline_state = PipelineState(**state_data)
                upsert_pipeline_state_to_duckdb(pipeline_state)
                processed_count += 1

            except Exception as e:
                logger.warning(f"Data validation error for record {state_data}: {e}")
                continue

        # Increment offset for next batch
        offset += batch_size

    logger.info(
        f"✅ Synced {processed_count} similarity metrics to pipeline_control using primary keys."
    )


def run_state_orchestration_pipeline(
    *,
    full: bool = True,
    create_table: bool = True,
    integrity_check: bool = True,
) -> None:
    """
    Orchestrate syncing stage/source tables into `pipeline_control` (FSM control plane).

    When to run
    -----------
    - **Bootstrap**: first-time setup to seed control rows from existing stage tables.
    - **Recovery**: after failures or manual edits, to realign control rows.
    - **Periodic refresh**: keep `pipeline_control` consistent with stage tables.

    What it does
    ------------
    - Optionally ensures the `pipeline_control` table exists.
    - If `full=True`, runs all stage syncs in order
        (job_urls → edited_responsibilities).
    - Always refreshes similarity metrics (often your gating signal).
    - Optionally runs a quick FSM integrity check.

    Notes
    -----
    - Each sync is isolated with its own try/except so one failure does not
        halt the rest.
    - This function does **not** execute the stage pipelines;
        it only manages control state.
    """
    logger.info("🧭 Starting pipeline state orchestration...")

    if create_table:
        try:
            create_single_db_table(table_name=TableName.PIPELINE_CONTROL)
        except Exception as e:
            logger.warning(
                f"⚠️ Could not ensure pipeline_control table exists: {e}", exc_info=True
            )

    if full:
        sync_steps = [
            ("job_urls", sync_job_urls_to_pipeline_control),
            ("job_postings", sync_job_postings_to_pipeline_control),
            ("extracted_requirements", sync_extracted_requirements_to_pipeline_control),
            ("flattened_requirements", sync_flattened_requirements_to_pipeline_control),
            (
                "flattened_responsibilities",
                sync_flattened_responsibilities_to_pipeline_control,
            ),
            (
                "edited_responsibilities",
                sync_edited_responsibilities_to_pipeline_control,
            ),
        ]
        for name, func in sync_steps:
            try:
                func()
            except Exception as e:
                logger.error(f"❌ Failed syncing {name}: {e}", exc_info=True)

    # Similarity metrics are frequently the gating/control signal—always refresh them.
    try:
        sync_similarity_metrics_to_pipeline_control()
    except Exception as e:
        logger.error(f"❌ Failed syncing similarity_metrics: {e}", exc_info=True)

    if integrity_check:
        try:
            validate_fsm_integrity()
        except Exception as e:
            logger.warning(
                f"⚠️ FSM integrity check encountered an issue: {e}", exc_info=True
            )

    logger.info("✅ Pipeline control table sync complete.")
