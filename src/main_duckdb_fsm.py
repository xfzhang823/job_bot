from db_io.db_schema_registry import DUCKDB_TABLE_SCHEMAS, DUCKDB_TABLE_COLUMN_ORDER
import logging
import logging_config

logger = logging.getLogger(__name__)

logger.info(DUCKDB_TABLE_SCHEMAS)
print()

logger.info(DUCKDB_TABLE_COLUMN_ORDER)
