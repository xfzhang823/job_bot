""" Temp for testing """

import os
import logging
import json


from config import resume_json_file, job_requirements_json_file
from utils.generic_utils import pretty_print_json
from utils.dict_utils import flatten_dict

# Set up logging
logger = logging.getLogger(__name__)


def main():

    f_path = r"C:\github\job_bot\input_output\evaluation_optimization\modified_responsibilities_flat_iteration_1.json"
    with open(f_path, "r") as file:
        dict = json.load(file)

    print(json.dumps(dict, indent=4))

    flat_dict = flatten_dict(dict)
    pretty_print_json(flat_dict)


if __name__ == "__main__":
    main()
