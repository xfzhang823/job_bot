""" Temp for testing """

import logging
import os
import sys
from pathlib import Path

# Set and append root directory (required to access config.py)
# root_dir = Path(__file__).resolve().parent.parent.parent
# sys.path.append(str(root_dir))

import src.logging_config as logging_config

from pipelines.resume_eval_pipeline import run_pipeline
from utils.generic_utils import load_or_create_json, pretty_print_json


logger = logging.getLogger(__name__)


def example_function():
    logger.info("This is an info message from main_testing_2.")
    logger.error("This is an error message from main_testing_2.")


if __name__ == "__main__":
    example_function()
