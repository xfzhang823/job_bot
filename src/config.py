""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
...
)

"""

from pathlib import Path
import os
import logging

# logging.info("Starting config.py")

# Subdirectories under base directory that are part of the package inclue:
# - input_output
# - src
# rest of the sub directories, such as "data" are for ananlysis and other purposes only
# (not to be accessed programmatically)

# Base/Root Directory
# logging.info("Setting up base directories")
BASE_DIR = Path(r"C:\github\job_bot")  # base/root directory
# logging.info(f"BASE_DIR set to {BASE_DIR}")

# Input/Output directory, sub-directories, and file paths
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder
INPUT_DIR = INPUT_OUTPUT_DIR / "input"

# Resume file in JSON format
resume_json_file = INPUT_DIR / "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
job_posting_urls_file = INPUT_DIR / "job_posting_urls.json"
resume_docx_file = INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.docx"
resume_json_file_temp = (
    INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.json"
)  # for testing for now

# Preprocessing Input/Output
PREPROCESSING_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "preprocessing"
job_descriptions_json_file = PREPROCESSING_INPUT_OUTPUT_DIR / "jobpostings.json"
job_requirements_json_file = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "extracted_job_requirements.json"
)
description_text_holder = PREPROCESSING_INPUT_OUTPUT_DIR / "jobposting_text_holder.txt"
responsibilities_flat_json_file = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "responsibilities_flat.json"
)
requirements_flat_json_file = PREPROCESSING_INPUT_OUTPUT_DIR / "requirements_flat.json"

# Evaluation and Optimization Input/Output
EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "evaluation_optimization"
ITERATE_0_DIR = EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "iteration_0"
ITERATE_1_DIR = EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "iteration_1"
ITERATE_2_DIR = EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "iteration_2"

requirements_dir_name = "requirements"
responsibilities_dir_name = "responsibilities"
pruned_responsibilities_dir_name = "pruned_responsibilities"
similarity_metrics_dir_name = "similarity_metrics"
mapping_file_name = "url_to_file_mapping.json"
elbow_curve_plot_file_name = "elbow_curve_plot.png"

# Iteration 0
# logging.info("Setting up iteration 0 directories")
REQS_FILES_ITERATE_0_DIR = ITERATE_0_DIR / requirements_dir_name
RESPS_FILES_ITERATE_0_DIR = ITERATE_0_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_0_DIR = ITERATE_0_DIR / pruned_responsibilities_dir_name
SIMILARITY_METRICS_ITERATE_0_DIR = ITERATE_0_DIR / similarity_metrics_dir_name
url_to_file_mapping_file_iterate_0 = ITERATE_0_DIR / mapping_file_name

list_of_dirs = [
    REQS_FILES_ITERATE_0_DIR,
    RESPS_FILES_ITERATE_0_DIR,
    PRUNED_RESPS_FILES_ITERATE_0_DIR,
    SIMILARITY_METRICS_ITERATE_0_DIR,
]
# logging.info(
#     f"Iteration 0 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Create directories if they don't exist
for dir in list_of_dirs:
    dir.mkdir(parents=True, exist_ok=True)

# Iteration 1
# logging.info("Setting up iteration 1 directories")
REQS_FILES_ITERATE_1_DIR = ITERATE_1_DIR / requirements_dir_name
RESPS_FILES_ITERATE_1_DIR = ITERATE_1_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_1_DIR = ITERATE_1_DIR / pruned_responsibilities_dir_name
SIMILARITY_METRICS_ITERATE_1_DIR = ITERATE_1_DIR / similarity_metrics_dir_name
url_to_file_mapping_file_iterate_1 = ITERATE_1_DIR / mapping_file_name

list_of_dirs = [
    REQS_FILES_ITERATE_1_DIR,
    RESPS_FILES_ITERATE_1_DIR,
    PRUNED_RESPS_FILES_ITERATE_1_DIR,
    SIMILARITY_METRICS_ITERATE_1_DIR,
]
# logging.info(
#     f"Iteration 1 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Create directories for iteration 1
for dir in list_of_dirs:
    dir.mkdir(parents=True, exist_ok=True)

# Iteration 2
REQS_FILES_ITERATE_2_DIR = ITERATE_2_DIR / requirements_dir_name
RESPS_FILES_ITERATE_2_DIR = ITERATE_2_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_2_DIR = ITERATE_2_DIR / pruned_responsibilities_dir_name
SIMILARITY_METRICS_ITERATE_2_DIR = ITERATE_2_DIR / similarity_metrics_dir_name
url_to_file_mapping_file_iterate_2 = ITERATE_2_DIR / mapping_file_name

# Add directory creation for iteration 2 as well
list_of_dirs = [
    REQS_FILES_ITERATE_2_DIR,
    RESPS_FILES_ITERATE_2_DIR,
    PRUNED_RESPS_FILES_ITERATE_2_DIR,
    SIMILARITY_METRICS_ITERATE_2_DIR,
]
for dir in list_of_dirs:
    dir.mkdir(parents=True, exist_ok=True)
