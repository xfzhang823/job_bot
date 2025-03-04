"""
Module: main.py

This module serves as the entry point for running the pipeline to generate cross-tab reports.

Steps:
1. Loads the mapping file for iteration 1 (Anthropic).
2. Calls `run_resps_reqs_crosstab_pipeline_async()` to process similarity metrics.
3. Logs the progress and handles errors gracefully.

Usage:
- This script should be executed directly.
- It runs asynchronously to process multiple files concurrently.

Last Updated: 2025 Feb
"""

from pathlib import Path
import logging
import asyncio
from pipelines.resps_reqs_crosstab_pipeline_async import (
    run_resps_reqs_crosstab_pipeline_async,
)
from project_config import (
    ITERATE_0_ANTHROPIC_DIR,
    ITERATE_1_ANTHROPIC_DIR,
    mapping_file_name,
)
import logging_config


# Set up logger
logger = logging.getLogger(__name__)


def main():
    """Run pipeline to create cross-tab reporting files."""
    logger.info("🚀 Start running pipeline to generate cross-tab files...")

    try:
        mapping_file = ITERATE_1_ANTHROPIC_DIR / mapping_file_name

        if not mapping_file.exists():
            logger.error(f"❌ Mapping file not found: {mapping_file}")
            return

        asyncio.run(run_resps_reqs_crosstab_pipeline_async(mapping_file=mapping_file))

        logger.info("✅ Finished running pipeline to generate cross-tab files.")

    except Exception as e:
        logger.exception(f"❌ An error occurred while running the pipeline: {e}")


def main_iter0():
    """Run pipeline to create cross-tab reporting files (before editing)"""
    logger.info("🚀 Start running pipeline to generate cross-tab files...")

    try:
        mapping_file = ITERATE_0_ANTHROPIC_DIR / mapping_file_name

        if not mapping_file.exists():
            logger.error(f"❌ Mapping file not found: {mapping_file}")
            return

        cross_tab_output_dir = Path(
            r"C:\github\job_bot\input_output\human_review\resps_reqs_matching_pre_edit"
        )

        asyncio.run(
            run_resps_reqs_crosstab_pipeline_async(
                mapping_file=mapping_file, cross_tab_output_dir=cross_tab_output_dir
            )
        )

        logger.info("✅ Finished running pipeline to generate cross-tab files.")

    except Exception as e:
        logger.exception(f"❌ An error occurred while running the pipeline: {e}")


if __name__ == "__main__":
    main_iter0()
