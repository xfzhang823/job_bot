"""main_duckdb_fsm.py"""

import asyncio
import logging

import logging_config
from db_io.create_db_tables import create_all_db_tables
from pipelines_with_fsm.duckdb_ingestion_pipeline import run_duckdb_ingestion_pipeline
from pipelines_with_fsm.state_orchestration_pipeline import (
    run_state_orchestration_pipeline,
)


# FSM-aware runners that READ/WRITE DuckDB and advance stages
from pipelines_with_fsm.preprocessing_pipeline_async_fsm import (
    run_preprocessing_pipeline_async_fsm,
)
from pipelines_with_fsm.resume_eval_pipeline_async_with_fsm import (
    run_metrics_processing_pipeline_async,
)
from pipelines_with_fsm.resume_editing_pipeline_async_fsm import (
    run_resume_editing_pipeline_async,
)
from pipelines_with_fsm.resps_reqs_crosstab_pipeline_async_fsm import (
    run_resps_reqs_crosstab_pipeline_async,
)


async def run_all():
    # 0) Ensure DB schema exists (idempotent)
    create_all_db_tables()

    # 1) Seed control-plane so FSM runners have a worklist (idempotent)
    #    Full sync the first time; later runs can pass full=False for speed.
    run_state_orchestration_pipeline(
        full=True, create_table=False, integrity_check=False
    )

    # 2) Do work (each runner queries pipeline_control, writes back to DuckDB, advances FSM)
    await run_preprocessing_pipeline_async_fsm()
    await run_metrics_processing_pipeline_async()
    await run_resume_editing_pipeline_async()

    # 3) Refresh control-plane (targeted; keeps pipeline_control perfectly aligned)
    run_state_orchestration_pipeline(
        full=False, create_table=False, integrity_check=True
    )

    # 4) Optional: reports (Excel) as the very last step
    await run_resps_reqs_crosstab_pipeline_async()


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    asyncio.run(run_all())
