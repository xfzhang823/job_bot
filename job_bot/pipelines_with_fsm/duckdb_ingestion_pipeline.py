"""
duckdb_ingestion_pipeline.py

This module orchestrates the ingestion of structured resume-job alignment data
into a DuckDB database.

It is organized around modular 'mini-pipelines' that handle different phases of
the processing pipeline.

Each mini-pipeline:
- Loads validated Pydantic models or DataFrames
- Applies schema alignment and metadata enrichment
- Deduplicates based on key fields
- Inserts aligned data into DuckDB tables

Supported phases:
- Preprocessing: Raw job postings, URLs, and extracted requirements
- Staging: Flattened resume responsibilities and job requirements
- Evaluation: Original and edited semantic similarity metrics
- Editing: LLM-optimized responsibilities
"""

import logging

# User defined
from job_bot.db_io.create_db_tables import create_all_db_tables
from job_bot.db_io.pipeline_enums import PipelineStage, LLMProvider, Version


from job_bot.db_io.file_ingestion import (
    ingest_job_urls_file,
    ingest_job_postings_file,
    ingest_extracted_requirements_file,
    ingest_flattened_requirements_file,
    ingest_flattened_responsibilities_file,
    ingest_similarity_metrics_file,
    ingest_edited_responsibilities_file,
)
from job_bot.config.project_config import (
    RESPS_FILES_ITERATE_0_OPENAI_DIR,
    REQS_FILES_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_ANTHROPIC_DIR,
    SIMILARITY_METRICS_ITERATE_1_ANTHROPIC_DIR,
)

logger = logging.getLogger(__name__)


def run_duckdb_ingestion_pipeline():
    """
    Main orchestrator for DuckDB ingestion.

    Ingests all structured outputs from the resume-job alignment pipeline, including:
    - Preprocessing outputs: job URLs, job postings, extracted requirements
    - Staging outputs: flattened responsibilities and requirements (iteration 0)
    - Evaluation outputs: similarity metrics (original and edited)
    - Editing outputs: LLM-optimized responsibilities (iteration 1)

    This function assumes all source files exist in their expected paths
    as defined in `project_config.py`.
    """
    logger.info("🏗️ Creating DuckDB tables...")
    create_all_db_tables()
    logger.info("✅ DuckDB schema setup complete.")

    # 🔹 Preprocessing (single-file tables)
    ingest_job_urls_file()
    ingest_job_postings_file()
    ingest_extracted_requirements_file()

    # 🔹 Flattened requirements & responsibilities (iteration 0)
    for file_path in REQS_FILES_ITERATE_0_OPENAI_DIR.glob("*.json"):
        ingest_flattened_requirements_file(file_path)

    for file_path in RESPS_FILES_ITERATE_0_OPENAI_DIR.glob("*.json"):
        ingest_flattened_responsibilities_file(file_path)

    # 🔹 Original similarity metrics (iteration 0)
    for file_path in SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR.glob("*.csv"):
        ingest_similarity_metrics_file(
            source_file=file_path,
            version=Version.ORIGINAL,
            stage=PipelineStage.SIM_METRICS_EVAL,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    # 🔹 Edited responsibilities (iteration 1)
    for file_path in RESPS_FILES_ITERATE_1_OPENAI_DIR.glob("*.json"):
        ingest_edited_responsibilities_file(
            file_path=file_path,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    # 🔹 Edited similarity metrics (iteration 1)
    for file_path in SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR.glob("*.csv"):
        ingest_similarity_metrics_file(
            source_file=file_path,
            version=Version.EDITED,
            stage=PipelineStage.SIM_METRICS_REVAL,
            llm_provider=LLMProvider.OPENAI,
            iteration=0,
        )

    # 🔹 Edited responsibilities (iteration 1 - Anthropic)
    for file_path in RESPS_FILES_ITERATE_1_ANTHROPIC_DIR.glob("*.json"):
        ingest_edited_responsibilities_file(
            file_path=file_path,
            llm_provider=LLMProvider.ANTHROPIC,
            iteration=0,
        )

    # 🔹 Edited similarity metrics (iteration 1 - Anthropic)
    for file_path in SIMILARITY_METRICS_ITERATE_1_ANTHROPIC_DIR.glob("*.csv"):
        ingest_similarity_metrics_file(
            source_file=file_path,
            version=Version.EDITED,
            stage=PipelineStage.SIM_METRICS_REVAL,
            llm_provider=LLMProvider.ANTHROPIC,
            iteration=0,
        )

    logger.info("🏁 DuckDB ingestion pipeline complete.")
