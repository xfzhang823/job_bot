from pathlib import Path
import logging
import job_bot.config.logging_config
from job_bot.fsm.pipeline_control_sync import (
    sync_all_tables_to_pipeline_control,
)

logger = logging.getLogger(__name__)


def main():
    sync_all_tables_to_pipeline_control()


if __name__ == "__main__":
    main()
