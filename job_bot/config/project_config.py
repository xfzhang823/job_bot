"""
config/project_config.py

Data Input/Output dir/file configuration

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


# * Input/Output directory, sub-directories, and file paths
INPUT_OUTPUT_DIR = BASE_DIR / "input_output"  # input/output data folder
INPUT_DIR = INPUT_OUTPUT_DIR / "input"


# resume_json_file = INPUT_DIR / "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
RESUME_JSON_FILE = (
    INPUT_DIR / "Resume_Xiaofei_Zhang_2025_template_for_LLM.json"
)  # * Resume file in JSON format

JOB_POSTING_URLS_FILE = (
    INPUT_DIR / "job_posting_urls.json"
)  # * Job posting urls in JSON format
JOB_POSTING_URLS_TO_EXCLUDE_FILE = (
    INPUT_DIR / "job_posting_urls_to_exclude.json"
)  # * Job posting urls to exclude in JSON format
JOB_POSTING_URLS_FILTERED_FILE = (
    INPUT_DIR / "job_posting_urls_filtered.json"
)  # * Job posting urls to exclude in JSON format

RESUME_DOCX_FILE = INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.docx"
# resume_json_file_temp = (
#     INPUT_DIR / "Resume Xiao-Fei Zhang 2024_Mkt_Intel.json"
# )  # for testing for now
resume_json_file_temp = (
    INPUT_DIR / "Resume Xiao-Fei Zhang 2025_Mkt_Intel.json"
)  # for testing for now

# * Preprocessing Input/Output
PREPROCESSING_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "preprocessing"
JOB_DESCRIPTIONS_JSON_FILE = PREPROCESSING_INPUT_OUTPUT_DIR / "jobpostings.json"
JOB_REQUIREMENTS_JSON_FILE = (
    PREPROCESSING_INPUT_OUTPUT_DIR / "extracted_job_requirements.json"
)
description_text_holder = PREPROCESSING_INPUT_OUTPUT_DIR / "jobposting_text_holder.txt"
RESPONSIBILITIES_FLAT_JSON_FILE = (
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
URL_TO_FILE_MAPPING_FILE_ITERATE_0_OPENAI = ITERATE_0_OPENAI_DIR / mapping_file_name

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

# *Eval and Optimization - by Anthropic/Claude
# Evaluation and Optimization Input/Output
EVALUATION_OPTIMIZATION_BY_ANTHROPIC_DIR = (
    EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR / "evaluation_optimization_by_anthropic"
)

ITERATE_0_ANTHROPIC_DIR = EVALUATION_OPTIMIZATION_BY_ANTHROPIC_DIR / "iteration_0"
ITERATE_1_ANTHROPIC_DIR = EVALUATION_OPTIMIZATION_BY_ANTHROPIC_DIR / "iteration_1"
ITERATE_2_ANTHROPIC_DIR = EVALUATION_OPTIMIZATION_BY_ANTHROPIC_DIR / "iteration_2"

# Iteration 0
# logging.info("Setting up iteration 0 directories")
REQS_FILES_ITERATE_0_ANTHROPIC_DIR = ITERATE_0_ANTHROPIC_DIR / requirements_dir_name
RESPS_FILES_ITERATE_0_ANTHROPIC_DIR = (
    ITERATE_0_ANTHROPIC_DIR / responsibilities_dir_name
)
PRUNED_RESPS_FILES_ITERATE_0_ANTHROPIC_DIR = (
    ITERATE_0_ANTHROPIC_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_0_ANTHROPIC_DIR = (
    ITERATE_0_ANTHROPIC_DIR / similarity_metrics_dir_name
)
URL_TO_FILE_MAPPING_FILE_ITERATE_0_ANTHROPIC = (
    ITERATE_0_ANTHROPIC_DIR / mapping_file_name
)

# logging.info(
#     f"Iteration 0 directories set to: {', '.join(str(dir) for dir in list_of_dirs)}"
# )

# Iteration 1
# logging.info("Setting up iteration 1 directories")
REQS_FILES_ITERATE_1_ANTHROPIC_DIR = ITERATE_1_ANTHROPIC_DIR / requirements_dir_name
RESPS_FILES_ITERATE_1_ANTHROPIC_DIR = (
    ITERATE_1_ANTHROPIC_DIR / responsibilities_dir_name
)
PRUNED_RESPS_FILES_ITERATE_1_ANTHROPIC_DIR = (
    ITERATE_1_ANTHROPIC_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_1_ANTHROPIC_DIR = (
    ITERATE_1_ANTHROPIC_DIR / similarity_metrics_dir_name
)
URL_TO_FILE_MAPPING_FILE_ITERATE_1_ANTHROPIC = (
    ITERATE_1_ANTHROPIC_DIR / mapping_file_name
)

# Iteration 2
REQS_FILES_ITERATE_2_ANTHROPIC_DIR = ITERATE_2_ANTHROPIC_DIR / requirements_dir_name
RESPS_FILES_ITERATE_2_ANTHROPIC_DIR = (
    ITERATE_2_ANTHROPIC_DIR / responsibilities_dir_name
)
PRUNED_RESPS_FILES_ITERATE_2_ANTHROPIC_DIR = (
    ITERATE_2_ANTHROPIC_DIR / pruned_responsibilities_dir_name
)
SIMILARITY_METRICS_ITERATE_2_ANTHROPIC_DIR = (
    ITERATE_2_ANTHROPIC_DIR / similarity_metrics_dir_name
)
URL_TO_FILE_MAPPING_FILE_ITERATE_2_ANTHROPIC = (
    ITERATE_2_ANTHROPIC_DIR / mapping_file_name
)

# Human Review Directory
HUMAN_REVIEW_INPUT_OUTPUT_DIR = INPUT_OUTPUT_DIR / "human_review"

RESPS_REQS_MATCHINGS_DIR = (
    HUMAN_REVIEW_INPUT_OUTPUT_DIR / "resps_reqs_matchings"
)  # Contains raw matching crosstab excel files
REVIEWED_MATCHINGS_DIR = (
    HUMAN_REVIEW_INPUT_OUTPUT_DIR / "resps_reqs_matchings_reviewed"
)  # Contains reviewed matching crosstab excel files
TRIMMED_MATCHINGS_DIR = (
    HUMAN_REVIEW_INPUT_OUTPUT_DIR / "resps_reqs_matchings_trimmed"
)  # Contains trimmed final json files (trimmed by LLMs)
# "C:\github\job_bot\input_output\human_review\resps_reqs_matching\reviewed_matchings"

# * Data File Configuration
DATA_FILES_CONFIG = {
    "job_posting_urls_file": {
        "path": JOB_POSTING_URLS_FILE,
        "description": "This file contains URLs for job postings along with \
company names and positions.",
        "format": "JSON",
        "fields": ["url", "company", "position"],
        "example": {
            "url": "https://example.com/job-posting",
            "company": "Tech Corp",
            "position": "Software Engineer",
        },
    },
    "job_descriptions_json_file": {
        "path": JOB_DESCRIPTIONS_JSON_FILE,
        "description": "Contains raw content of job postings scraped from URLs. \
Each entry contains the job title, company, and a full description.",
        "format": "JSON",
        "fields": ["job_title", "company", "location", "description"],
        "example": {
            "job_title": "Software Engineer",
            "company": "Tech Corp",
            "location": "San Francisco, CA",
            "description": "We are looking for a Software Engineer...",
        },
    },
    "job_requirements_json_file": {
        "path": JOB_REQUIREMENTS_JSON_FILE,
        "description": "Contains the extracted job requirements from the job descriptions. \
This file lists the key qualifications and responsibilities.",
        "format": "JSON",
        "fields": ["job_title", "company", "requirements"],
        "example": {
            "job_title": "Software Engineer",
            "company": "Tech Corp",
            "requirements": [
                "10+ years of experience in software development",
                "Proficiency in Python, JavaScript",
            ],
        },
    },
}


# * Pipeline Data directory, sub-directories, and file paths (contains duckdb database)
PIPELINE_DATA_DIR = BASE_DIR / "pipeline_data"  # input/output data folder
DB_DIR = PIPELINE_DATA_DIR / "db"
DUCKDB_FILE = DB_DIR / "pipeline_data.duckdb"

"db_io/db_loaders_config.yaml"
# *LLM Models

# llm_providers
OPENAI = "openai"
ANTHROPIC = "anthropic"

# Model IDs
# Anthropic (Claude) models
CLAUDE_OPUS = "claude-3-opus-20240229"
CLAUDE_SONNET_3_5 = "claude-3-5-sonnet-20241022"
CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"
CLAUDE_HAIKU = "claude-3-haiku-20240307"

# ── OpenAI models
# Keep legacy constant so existing imports don't break (use only if you must)
GPT_35_TURBO = "gpt-3.5-turbo"  # legacy; keep for backward compatibility

# Preferred current options for cheap/fast alignment:
GPT_4O_MINI = "gpt-4o-mini"  # ✅ recommended default for your pipelines
GPT_4O = "gpt-4o"

# Newer 4.1 family (mini for larger contexts with good price/perf)
GPT_4_1 = "gpt-4.1"
GPT_4_1_MINI = "gpt-4.1-mini"
GPT_4_1_NANO = "gpt-4.1-nano"

# Reasoning family (use sparingly for alignment tasks)
GPT_O3 = "o3"
GPT_O3_MINI = "o3-mini"
GPT_O3_PRO = "o3-pro"

# Back-compat alias: map older code expecting GPT_4_TURBO onto 4o-mini
# (This avoids touching pipeline_config/main.py right now.)
GPT_4_TURBO = GPT_4O_MINI

# EMBEDDINGS (RAG etc.)
EMBEDDING_3_SMALL = "text-embedding-3-small"
EMBEDDING_3_LARGE = "text-embedding-3-large"


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
    "anthropic": {
        "model_id": "claude-3-5-sonnet-20241022",
        "iteration_dirs": {
            "0": ITERATE_0_ANTHROPIC_DIR,
            "1": ITERATE_1_ANTHROPIC_DIR,
            "2": ITERATE_2_ANTHROPIC_DIR,
            # Add more as needed
        },
    },
    # Add configurations for other LLM providers here
}

# Config Files
SRC_ROOT_DIR = BASE_DIR / "job_bot"
DB_IO_DIR = SRC_ROOT_DIR / "db_io"
CONFIG_DIR = SRC_ROOT_DIR / "config"
DB_LOADERS_YAML = CONFIG_DIR / "db_loaders.yaml"
DB_INSERTERS_YAML = CONFIG_DIR / "db_inserters.yaml"
