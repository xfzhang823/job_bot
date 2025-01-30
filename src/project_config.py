""" Data Input/Output dir/file configuration 

# example_usagage (from modules)

from config import (
...
)

"""

from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def find_project_root(starting_path=None, marker=".git"):
    """
    Recursively find the root directory of the project by looking for a specific marker.

    Args:
        - starting_path (str or Path): The starting path to begin the search. Defaults to
        the current script's directory.
        - marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project,
        or None if not found.
    """
    # Start from the directory of the current file if not specified
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    # Convert starting_path to a Path object if it's not already
    starting_path = Path(starting_path)

    # Traverse up the directory tree
    for parent in starting_path.parents:
        # Check if the marker exists in the current directory
        if (parent / marker).exists():
            return parent

    return None  # Return None if the marker is not found


# Subdirectories under base directory that are part of the package inclue:
# - input_output
# - src
# rest of the sub directories, such as "data" are for ananlysis and other purposes only
# (not to be accessed programmatically)

# Base/Root Directory
# logging.info("Setting up base directories")
BASE_DIR = find_project_root()  # base/root directory
if not BASE_DIR or not BASE_DIR.exists():
    raise ValueError("Invalid project root directory")


# Input/Output directory, sub-directories, and file paths
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder
INPUT_DIR = INPUT_OUTPUT_DIR / "input"

# * Resume file in JSON format
# resume_json_file = INPUT_DIR / "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
resume_json_file = INPUT_DIR / "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
job_posting_urls_file = INPUT_DIR / "job_posting_urls.json"
resume_docx_file = INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.docx"
# resume_json_file_temp = (
#     INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.json"
# )  # for testing for now
resume_json_file_temp = (
    INPUT_DIR / "Resume Xiao-Fei Zhang 2025_Mkt_Intel.json"
)  # for testing for now

# * Preprocessing Input/Output
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

# * Evaluation and Optimization Input/Output
EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "evaluation_optimization"

# Evaluation and Optimization I/O by OpenAI
EVALUATION_OPTIMIZATION_BY_OPENAI_DIR = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "evaluation_optimization_by_openai"
)
ITERATE_0_OPENAI_DIR = EVALUATION_OPTIMIZATION_BY_OPENAI_DIR / "iteration_0"
ITERATE_1_OPENAI_DIR = EVALUATION_OPTIMIZATION_BY_OPENAI_DIR / "iteration_1"
ITERATE_2_OPENAI_DIR = EVALUATION_OPTIMIZATION_BY_OPENAI_DIR / "iteration_2"

requirements_dir_name = "requirements"
responsibilities_dir_name = "responsibilities"
pruned_responsibilities_dir_name = "pruned_responsibilities"
similarity_metrics_dir_name = "similarity_metrics"
mapping_file_name = "url_to_file_mapping.json"
elbow_curve_plot_file_name = "elbow_curve_plot.png"

# Iteration 0
# logging.info("Setting up iteration 0 directories")
REQS_FILES_ITERATE_0_OPENAI_DIR = ITERATE_0_OPENAI_DIR / requirements_dir_name
RESPS_FILES_ITERATE_0_OPENAI_DIR = ITERATE_0_OPENAI_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_0_OPENAI_DIR = (
    ITERATE_0_OPENAI_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR = (
    ITERATE_0_OPENAI_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_0_openai = ITERATE_0_OPENAI_DIR / mapping_file_name


# logging.info(
#     f"Iteration 0 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Iteration 1
# logging.info("Setting up iteration 1 directories")
REQS_FILES_ITERATE_1_OPENAI_DIR = ITERATE_1_OPENAI_DIR / requirements_dir_name
RESPS_FILES_ITERATE_1_OPENAI_DIR = ITERATE_1_OPENAI_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_1_DIR_OPENAI = (
    ITERATE_1_OPENAI_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR = (
    ITERATE_1_OPENAI_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_1_openai = ITERATE_1_OPENAI_DIR / mapping_file_name

# logging.info(
#     f"Iteration 1 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Iteration 2
REQS_FILES_ITERATE_2_OPENAI_DIR = ITERATE_2_OPENAI_DIR / requirements_dir_name
RESPS_FILES_ITERATE_2_OPENAI_DIR = ITERATE_2_OPENAI_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_2_OPENAI_DIR = (
    ITERATE_2_OPENAI_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_2_OPENAI_DIR = (
    ITERATE_2_OPENAI_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_2_openai = ITERATE_2_OPENAI_DIR / mapping_file_name

# *Eval and Optimization - by Claude
# Evaluation and Optimization Input/Output
EVALUATION_OPTIMIZATION_BY_CLAUDE_DIR = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "evaluation_optimization_by_claude"
)

ITERATE_0_CLAUDE_DIR = EVALUATION_OPTIMIZATION_BY_CLAUDE_DIR / "iteration_0"
ITERATE_1_CLAUDE_DIR = EVALUATION_OPTIMIZATION_BY_CLAUDE_DIR / "iteration_1"
ITERATE_2_CLAUDE_DIR = EVALUATION_OPTIMIZATION_BY_CLAUDE_DIR / "iteration_2"

# Iteration 0
# logging.info("Setting up iteration 0 directories")
REQS_FILES_ITERATE_0_CLAUDE_DIR = ITERATE_0_CLAUDE_DIR / requirements_dir_name
RESPS_FILES_ITERATE_0_CLAUDE_CLAUDE = ITERATE_0_CLAUDE_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_0_DIR_CLAUDE = (
    ITERATE_0_CLAUDE_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_0_CLAUDE_DIR = (
    ITERATE_0_CLAUDE_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_0_claude = ITERATE_0_CLAUDE_DIR / mapping_file_name

# logging.info(
#     f"Iteration 0 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Iteration 1
# logging.info("Setting up iteration 1 directories")
REQS_FILES_ITERATE_1_CLAUDE_DIR = ITERATE_1_CLAUDE_DIR / requirements_dir_name
RESPS_FILES_ITERATE_1_CLAUDE_DIR = ITERATE_1_OPENAI_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_1_CLAUDE_DIR = (
    ITERATE_1_CLAUDE_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_1_CLAUDE_DIR = (
    ITERATE_1_CLAUDE_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_1_claude = ITERATE_1_CLAUDE_DIR / mapping_file_name

# Iteration 2
REQS_FILES_ITERATE_2_CLAUDE_DIR = ITERATE_2_CLAUDE_DIR / requirements_dir_name
RESPS_FILES_ITERATE_2_CLAUDE_DIR = ITERATE_2_CLAUDE_DIR / responsibilities_dir_name
PRUNED_RESPS_FILES_ITERATE_2_CLAUDE_DIR = (
    ITERATE_2_CLAUDE_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_2_CLAUDE_DIR = (
    ITERATE_2_CLAUDE_DIR / similarity_metrics_dir_name
)
url_to_file_mapping_file_iterate_2_claude = ITERATE_2_CLAUDE_DIR / mapping_file_name


# *LLM Models
# Anthropic (Claude) models
CLAUDE_OPUS = "claude-3-opus-20240229"
CLAUDE_SONNET = "claude-3-5-sonnet-20241022"
CLAUDE_HAIKU = "claude-3-haiku-20240307"

# OpenAI models
GPT_35_TURBO = "gpt-3.5-turbo"
GPT_35_TURBO_16K = "gpt-3.5-turbo-16k"
GPT_4 = "gpt-4"
GPT_4_TURBO = "gpt-4-turbo"
GPT_4_TURBO_32K = "gpt-4-turbo-32k"
GPT_4O = "gpt-4o"


# *LLM Provider configuration
# config.py (additions)
LLM_CONFIG = {
    "openai": {
        "model_id": "gpt-4-turbo",
        "iteration_dirs": {
            "0": ITERATE_0_OPENAI_DIR,
            "1": ITERATE_1_OPENAI_DIR,
            "2": ITERATE_2_OPENAI_DIR,
            # Add more as needed
        },
    },
    "claude": {
        "model_id": "claude-3-5-sonnet-20241022",
        "iteration_dirs": {
            "0": ITERATE_0_CLAUDE_DIR,
            "1": ITERATE_1_CLAUDE_DIR,
            "2": ITERATE_2_CLAUDE_DIR,
            # Add more as needed
        },
    },
    # Add configurations for other LLM providers here
}
