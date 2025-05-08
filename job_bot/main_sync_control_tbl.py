from pathlib import Path
import logging
import logging_config
from pipelines_with_fsm.state_orchestration_pipeline import (
    run_state_orchestration_pipeline,
)

logger = logging.getLogger(__name__)


def main():
    run_state_orchestration_pipeline()


if __name__ == "__main__":
    main()
