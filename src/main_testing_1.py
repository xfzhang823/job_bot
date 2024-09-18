""" Temp for testing """

import os
import logging
import json

from pipelines.preprocessing_pipeline import resume_and_requirements_json_to_csv
from config import resume_json_file, job_requirements_json_file
from utils.generic_utils import pretty_print_json

# Set up logging
logger = logging.getLogger(__name__)


def main():

    print("\n\n")
    pretty_print_json(requirements)
    print(type(requirements))


if __name__ == "__main__":
    main()
