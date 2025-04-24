import logging
from pipelines.duckdb_ingestion_pipeline import run_duckdb_ingestion_pipeline
import logging_config

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting DuckDB ingestion pipeline...")
    run_duckdb_ingestion_pipeline()
    logger.info("Pipeline finished.")


if __name__ == "__main__":
    main()
