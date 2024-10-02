""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
...
)

"""

# config.py

from pathlib import Path
import os


# Base/Root Directory
BASE_DIR = Path(r"C:\github\job_bot")  # base/root directory

# Subdirectories under base directory that are part of the package inclue:
# - input_output
# - src
# rest of the sub directories, such as "data" are for ananlysis and other purposes only
# (not to be accessed programmatically)


# Input/Output directory, sub-directories, and file paths
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder

# Sub directories
# Input
INPUT_DIR = INPUT_OUTPUT_DIR / "input"

# Resume file in JSON format
resume_json_file = INPUT_DIR / "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
job_posting_urls_file = INPUT_DIR / "job_posting_urls.json"


# Preprocessing Input/Output
PREPROCESSING_INPUT_OUTPUT_DIR = (
    INPUT_OUTPUT_DIR / "preprocessing"
)  # preprocessed data (each row converted to arragy friendly format)
job_descriptions_json_file = PREPROCESSING_INPUT_OUTPUT_DIR / "jobpostings.json"
job_requirements_json_file = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "extracted_job_requirements.json"
)
description_text_holder = PREPROCESSING_INPUT_OUTPUT_DIR / "jobposting_text_holder.txt"
responsibilities_flat_json_file = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "responsibilities_flat.json"
)  # extracted & flattened responsibilities from resume in JSON format (dict)
requirements_flat_json_file = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "requirements_flat.json"
)  # extracted & flattened responsibilities from resume in JSON format (dict)


# Evaluation and Optimization
EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "evaluation_optimization"

# CSV file containing the original resp (from resume) and reqs (from job description)
# along with different similarity related scores/score categories
resp_req_sim_metrics_0_csv_file = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "output_seg_by_seg_sim_metrics_0.csv"
)
resp_req_sim_metrics_1_csv_file = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "output_seg_by_seg_sim_metrics_1.csv"
)
resp_req_sim_metrics_2_csv_file = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "output_seg_by_seg_sim_metrics_2.csv"
)

modified_resps_flat_iter_1_json_file = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR
    / "modified_responsibilities_flat_iteration_1.json"
)
modified_resps_flat_iter_2_json_file = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR
    / "modified_responsibilities_flat_iteration_2.json"
)


METRICS_OUTPUTS_EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR = "C:\github\job_bot\input_output\evaluation_optimization\output_seg_by_seg_sim_metrics"
# CSV file containing df with resp. excluded from modification b/c they are just
# factual statements, i.e., "promoted to .... in ..."
# They need to be added back to the final edited resume

# excluded_from_modification_file = (
#     RESUME_EDITING_INPUT_OUTPUT_DIR / "dataframe_to_add_back.csv"
# )
