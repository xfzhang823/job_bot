"""
pipeline_config.py

This module defines the configuration for various pipeline stages, including preprocessing,
evaluation, and editing. It specifies the functions, file paths, and LLM providers (OpenAI or
Anthropic) for each stage of the pipeline. The configurations are stored in the `PIPELINE_CONFIG`
dictionary and are used to determine how data is processed at each stage.

The `PipelineStage` enum defines the stages of the pipeline (PREPROCESSING, EVALUATION, EDITING).
Each pipeline configuration includes:
- `description`: A brief explanation of the pipeline's purpose.
- `function`: The function to execute for the pipeline (e.g., `run_preprocessing_pipeline`).
- `io`: Input and output file mappings for different LLM providers (OpenAI or Anthropic).

This configuration is used by the `run_pipeline` function to dynamically select
and execute the appropriate pipeline based on the `pipeline_id`. The `run_pipeline` function
accesses these configurations to determine the function to call, the required files,
and the LLM provider, ensuring the correct processing
steps are followed for each pipeline.
"""

from enum import Enum
import logging

# * User defined: import pipeline functions
# Proprocessing
from pipelines.preprocessing_pipeline import run_preprocessing_pipeline
from pipelines.preprocessing_pipeline_async import run_preprocessing_pipeline_async

# Create/upsert mapping file for iteration 0
from pipelines.upserting_mapping_file_iter0_mini_pipeline import (
    run_upserting_mapping_file_iter0_mini_pipeline,
)

# Create flatttened files
from pipelines.flattened_resps_reqs_processing_mini_pipeline import (
    run_flatten_resps_reqs_processing_mini_pipeline,
)

# Compute & save similarity/entailment metrics & indices
from pipelines.resume_eval_pipeline import (
    run_metrics_processing_pipeline,
    generate_metrics_from_flat_json,
    generate_metrics_from_nested_json,
    run_multivariate_indices_processing_mini_pipeline,
)
from pipelines.resume_eval_pipeline_async import (
    generate_metrics_from_flat_json_async,
    generate_metrics_from_nested_json_async,
    run_metrics_processing_pipeline_async,
    run_multivariate_indices_processing_mini_pipeline_async,
    run_metrics_re_processing_pipeline_async,
)

from evaluation_optimization.evaluation_optimization_utils import (
    add_multivariate_indices,
)

# Clean metric files (couldn't get it cleaned in previous pipelines)
from pipelines.cleaning_metrics_files_pipeline import (
    run_cleaning_similarity_metrics_files_pipeline,
)

# Copy responsibilities over to pruned resps dir (for consistency)
from pipelines.copying_resps_to_pruned_resps_dir_mini_pipeline import (
    run_copying_resps_to_pruned_resps_dir_mini_pipeline,
)

# Exclude some resume items
from pipelines.excluding_resps_mini_pipeline import run_excluding_resps_mini_pipeline

# Create/upsert mapping file for iteration 1
from pipelines.upserting_mapping_file_iter1_mini_pipeline import (
    run_upserting_mapping_file_iter1_mini_pipeline,
)

# Edidting responsitibilites by matching requirements: iter 0 -> iter 1
from pipelines.resume_editing_pipeline import run_resume_editing_pipeline
from pipelines.resume_editing_pipeline_async import run_resume_editing_pipeline_async

from pipelines.copying_reqs_to_next_iter_mini_pipeline import (
    run_copying_reqs_to_next_iter_mini_pipeline,
)


# File & dir paths
from project_config import (
    RESUME_JSON_FILE,
    JOB_POSTING_URLS_FILE,
    JOB_POSTING_URLS_FILTERED_FILE,
    JOB_DESCRIPTIONS_JSON_FILE,
    JOB_REQUIREMENTS_JSON_FILE,
    mapping_file_name,
    # OpenAI directories for each iteration
    ITERATE_0_OPENAI_DIR,
    REQS_FILES_ITERATE_0_OPENAI_DIR,
    RESPS_FILES_ITERATE_0_OPENAI_DIR,
    PRUNED_RESPS_FILES_ITERATE_0_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
    ITERATE_1_OPENAI_DIR,
    REQS_FILES_ITERATE_1_OPENAI_DIR,
    RESPS_FILES_ITERATE_1_OPENAI_DIR,
    SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
    PRUNED_RESPS_FILES_ITERATE_1_DIR_OPENAI,
    # Claude directories for each iteration
    ITERATE_0_ANTHROPIC_DIR,
    REQS_FILES_ITERATE_0_ANTHROPIC_DIR,
    RESPS_FILES_ITERATE_0_ANTHROPIC_DIR,
    PRUNED_RESPS_FILES_ITERATE_0_ANTHROPIC_DIR,
    SIMILARITY_METRICS_ITERATE_0_ANTHROPIC_DIR,
    ITERATE_1_ANTHROPIC_DIR,
    REQS_FILES_ITERATE_1_ANTHROPIC_DIR,
    RESPS_FILES_ITERATE_1_ANTHROPIC_DIR,
    SIMILARITY_METRICS_ITERATE_1_ANTHROPIC_DIR,
    PRUNED_RESPS_FILES_ITERATE_1_ANTHROPIC_DIR,
)

# LLM providers & model ids
from project_config import (
    ANTHROPIC,
    OPENAI,
    GPT_35_TURBO,
    GPT_35_TURBO_16K,
    GPT_4,
    GPT_4_TURBO,
    GPT_4_TURBO_32K,
    GPT_4O,
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    CLAUDE_OPUS,
)
from db_io.pipeline_enums import PipelineStage

logger = logging.getLogger()


# Default model IDs for OpenAI and Anthropic (Claude)
DEFAULT_MODEL_IDS = {
    "openai": GPT_35_TURBO,  # Default OpenAI model
    "anthropic": CLAUDE_HAIKU,  # Default Anthropic (Claude) model
}


# Dictionary of configurations for each pipeline stage
PIPELINE_CONFIG = {
    "1": {
        "stage": PipelineStage.PREPROCESSING,  # Pipeline stage: Preprocessing
        "description": "Preprocessing job posting webpage(s)",
        # Scrape job-posting pages -> extract content -> a structured JSON file w/t jobpostings
        "function": run_preprocessing_pipeline,  # Function to call for this pipeline
        "io": {  # Input/Output files for OpenAI and Anthropic providers
            "openai": {
                "job_posting_urls_file": JOB_POSTING_URLS_FILE,  # * ALL posting URLs
                "job_descriptions_json_file": JOB_DESCRIPTIONS_JSON_FILE,  # * scraped/cleaned descriptions
                "job_requirements_json_file": JOB_REQUIREMENTS_JSON_FILE,  # * extract requirements
            },
            "anthropic": {
                "job_posting_urls_file": JOB_POSTING_URLS_FILE,  # * ALL posting URLs
                "job_descriptions_json_file": JOB_DESCRIPTIONS_JSON_FILE,  # * scraped/cleaned descriptions
                "job_requirements_json_file": JOB_REQUIREMENTS_JSON_FILE,  # * extract requirements
            },
        },
    },
    "1_async": {
        "stage": PipelineStage.PREPROCESSING,  # Async preprocessing stage
        "description": "Async preprocessing job posting webpage(s)",
        # Async version: Scrape job-posting pages -> extract content -> a structured JSON file
        # w/t jobpostings
        "function": run_preprocessing_pipeline_async,  # Async function to call
        "io": {
            "openai": {
                # "job_posting_urls_file": JOB_POSTING_URLS_FILE,  # * ALL posting URLs
                "job_posting_urls_file": JOB_POSTING_URLS_FILTERED_FILE,  # todo: use filtered urls file for now (temp) / change back later
                "job_descriptions_json_file": JOB_DESCRIPTIONS_JSON_FILE,  # * scraped/cleaned descriptions
                "job_requirements_json_file": JOB_REQUIREMENTS_JSON_FILE,  # * extract requirements
            },
            "anthropic": {
                # "job_posting_urls_file": JOB_POSTING_URLS_FILE,  # * ALL posting URLs
                "job_posting_urls_file": JOB_POSTING_URLS_FILTERED_FILE,  # todo: use filtered urls file for now (temp) / change back later
                "job_descriptions_json_file": JOB_DESCRIPTIONS_JSON_FILE,  # * scraped/cleaned descriptions
                "job_requirements_json_file": JOB_REQUIREMENTS_JSON_FILE,  # * extract requirements
            },
        },
    },
    "2a": {
        "stage": PipelineStage.EVALUATION,  # Evaluation stage
        "description": "Create/upsert mapping file for iteration 0",
        "function": run_upserting_mapping_file_iter0_mini_pipeline,
        # * Function to create/update mapping file
        "io": {
            "openai": {
                "job_descriptions_file": JOB_DESCRIPTIONS_JSON_FILE,
                "iteration_dir": ITERATE_0_OPENAI_DIR,  # Directory for OpenAI iteration 0
                "iteration": 0,
                "mapping_file_name": mapping_file_name,  # Reference file w/t file paths in the dir
            },
            "anthropic": {
                "job_descriptions_file": JOB_DESCRIPTIONS_JSON_FILE,
                "iteration_dir": ITERATE_0_ANTHROPIC_DIR,  # Directory for Anthropic iteration 0
                "iteration": 0,
                "mapping_file_name": mapping_file_name,  # Reference file w/t file paths in the dir
            },
        },
    },
    "2b": {
        "stage": PipelineStage.EVALUATION,
        "description": "Flatten responsibilities and requirements files",
        # * Nested resume files -> flatten JSON / responsibilities only files
        # * Nested job requirements -> flatted JSON & job requirements only files
        "function": run_flatten_resps_reqs_processing_mini_pipeline,
        # File: /pipelines/run_flatten_resps_reqs_processing_mini_pipeline.py
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # Reference file w/t file paths in the dir
                "job_requirements_file": JOB_REQUIREMENTS_JSON_FILE,
                # Flatted JSON file with job requirements of all jobpostings
                "resume_json_file": RESUME_JSON_FILE,  # Nested resume JSON file
            },
            "anthropic": {
                "mapping_file": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # Reference file w/t file paths in the dir
                "job_requirements_file": JOB_REQUIREMENTS_JSON_FILE,
                # Flatted JSON file with job requirements of all jobpostings
                "resume_json_file": RESUME_JSON_FILE,  # Nested resume JSON file
            },
        },
    },
    "2c": {
        "stage": PipelineStage.EVALUATION,
        "description": "Resume evaluation in iteration 0 (add similarity metrics)",
        "function": run_metrics_processing_pipeline,
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name,
            },
            "anthropic": {
                "mapping_file": ITERATE_0_ANTHROPIC_DIR / mapping_file_name,
            },
        },
        "kwargs": {  # Generic place for additional arguments (like callables)
            "generate_metrics": generate_metrics_from_flat_json
        },
    },
    "2c_async": {
        "stage": PipelineStage.EVALUATION,
        "description": "Async resume evaluation in iteration 0 (add similarity metrics)",
        # Async version: Match responsibilities and requirements and cacluate similarity metrics
        "function": run_metrics_processing_pipeline_async,
        # Function to match & create similarity metrics
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
        },
        "kwargs": {  # Generic place for additional arguments (like callables)
            "generate_metrics": generate_metrics_from_flat_json_async
        },
    },
    "2d": {
        "stage": PipelineStage.EVALUATION,
        "description": "Add extra indices (composite and PCA scores) to metrics files based \
on similarity metrics",
        "function": run_multivariate_indices_processing_mini_pipeline,
        # Function to add extra metrics to similarity metrics files
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
        },
        "kwargs": {  # Generic place for additional arguments (like callables)
            "add_indices_func": add_multivariate_indices
        },
    },
    "2d_async": {  # Run async to save time
        "stage": PipelineStage.EVALUATION,
        "description": "Asynchronously add extra indices (composite and PCA scores) to \
metrics files based on similarity metrics",
        # Async: Add extra indices (composite and PCA scores) to similarity files
        "function": run_multivariate_indices_processing_mini_pipeline_async,
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # Referene file w/t file paths in the dir
            },
        },
        "kwargs": {  # Generic place for additional arguments (like callables)
            "add_indices_func": add_multivariate_indices
        },
    },
    "2e": {
        "stage": PipelineStage.EVALUATION,
        "description": "Cleaning similarity metrics files by removing empty rows (iter 0)",
        "function": run_cleaning_similarity_metrics_files_pipeline,
        "io": {
            "openai": {"mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name},
            "anthropic": {"mapping_file": ITERATE_0_ANTHROPIC_DIR / mapping_file_name},
        },
    },
    "2f": {
        "stage": PipelineStage.EVALUATION,
        "description": "Copy and exclude responsibilities to pruned_responsibilities folder",
        "function": [
            run_copying_resps_to_pruned_resps_dir_mini_pipeline,
            run_excluding_resps_mini_pipeline,
        ],  # 2 functions
        "io": {
            "openai": {"mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name},
            "anthropic": {"mapping_file": ITERATE_0_ANTHROPIC_DIR / mapping_file_name},
        },
    },
    "3a": {
        "stage": PipelineStage.EDITING,
        "description": "Create/upsert mapping file for iteration 1",
        # Check if new urls (jobpostings) need to be added, create or update the reference file
        # w/t file paths in iteration 1
        "function": run_upserting_mapping_file_iter1_mini_pipeline,
        # function to check, create, or update the reference file of the dir
        "io": {
            "openai": {
                "job_descriptions_file": JOB_DESCRIPTIONS_JSON_FILE,
                # JSON file with ALL job descriptions (inc. urls)
                "iteration_dir": ITERATE_1_OPENAI_DIR,  # Directory of iternation 1
                "iteration": 1,  # Iteration number
                "mapping_file_name": mapping_file_name,  # Reference file w/t all file paths in the dir
            },
            "anthropic": {
                "job_descriptions_file": JOB_DESCRIPTIONS_JSON_FILE,
                # JSON file with ALL job descriptions (inc. urls)
                "iteration_dir": ITERATE_1_ANTHROPIC_DIR,  # Directory of iternation 1
                "iteration": 1,  # Iternation nubmer
                "mapping_file_name": mapping_file_name,  # Reference file w/t all file paths in the dir
            },
        },
    },
    "3b": {
        "stage": PipelineStage.EDITING,
        "description": "Modify responsibilities based on requirements using LLM",
        # Text alignment with 3-step LLM prompts: responsibilities -> job requirements
        "function": run_resume_editing_pipeline,
        # functio to run text alignmetn pipeline
        "llm_provider": "openai",
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # File path reference file of previous iteration
                "mapping_file_curr": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # File path reference file of current iteration
            },
            "anthropic": {
                "mapping_file_prev": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file of previous iteration
                "mapping_file_curr": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file of current iteration
            },
        },
    },
    "3b_async": {
        "stage": PipelineStage.EDITING,
        "description": "Async modification of responsibilities based on requirements",
        # Async: Text alignment with 3-step LLM prompts: responsibilities -> job requirements
        "function": run_resume_editing_pipeline_async,
        # functio to async run text alignment pipeline_async
        "llm_provider": "openai",
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # File path reference file of previous iteration
                "mapping_file_curr": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # File path reference file of current iteration
            },
            "anthropic": {
                "mapping_file_prev": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file of previous iteration
                "mapping_file_curr": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file of current iteration
            },
        },
    },
    "3c": {
        "stage": PipelineStage.EDITING,
        "description": "Copy requirements from iteration 0 to iteration 1",
        # Copy requirements from prev iteration to current iteration
        "function": run_copying_reqs_to_next_iter_mini_pipeline,
        # Funciton to copy files to current iteration folder
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR
                / mapping_file_name,  # File path reference file from previous iteration
                "mapping_file_curr": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # File path reference file from current iteration
            },
            "anthropic": {
                "mapping_file_prev": ITERATE_0_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file from previous iteration
                "mapping_file_curr": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # File path reference file from current iteration
            },
        },
    },
    "3d": {
        "stage": PipelineStage.EDITING,
        "description": "Resume evaluation in iteration 1 (add similarity metrics)",
        # Match edited responsibilities against requirements again to compute &
        # add similarity scores
        "function": run_metrics_re_processing_pipeline_async,
        # Function to compute similarity scores
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # Reference file w/t all file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # Reference file w/t all file paths in the dir
            },
        },
        "kwargs": {"generate_metrics": generate_metrics_from_nested_json},
    },
    "3d_async": {
        "stage": PipelineStage.EDITING,
        "description": "Async resume evaluation in iteration 1 (add similarity metrics)",
        # Async: Match edited responsibilities against requirements again to compute &
        # add similarity scores
        "function": run_metrics_re_processing_pipeline_async,
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # Reference file w/t all file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # Reference file w/t all file paths in the dir
            },
        },
        "kwargs": {"generate_metrics": generate_metrics_from_nested_json_async},
    },
    "3e": {
        "stage": PipelineStage.EDITING,
        "description": "Adding multivariate indices to metrics files in iteration 1",
        # Add extra indices (composite and PCA scores) to similarity files
        "function": run_multivariate_indices_processing_mini_pipeline,
        # Function to calculate and add extra indices
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # Reference file with all file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # Reference file with all file paths in the dir
            },
        },
        "kwargs": {"add_indices_func": add_multivariate_indices},
    },
    "3e_async": {
        "stage": PipelineStage.EDITING,
        "description": "Adding multivariate indices to metrics files in iteration 1",
        # Add extra indices (composite and PCA scores) to similarity files
        "function": run_multivariate_indices_processing_mini_pipeline_async,
        # Function to calculate and add extra indices
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR
                / mapping_file_name,  # Reference file with all file paths in the dir
            },
            "anthropic": {
                "mapping_file": ITERATE_1_ANTHROPIC_DIR
                / mapping_file_name,  # Reference file with all file paths in the dir
            },
        },
        "kwargs": {"add_indices_func": add_multivariate_indices},
    },
    "3f": {
        "stage": PipelineStage.EVALUATION,
        "description": "Cleaning similarity metrics files by removing empty rows (iter 1).",
        "function": run_cleaning_similarity_metrics_files_pipeline,
        "io": {
            "openai": {"mapping_file": ITERATE_1_OPENAI_DIR / mapping_file_name},
            "anthropic": {"mapping_file": ITERATE_1_ANTHROPIC_DIR / mapping_file_name},
        },
    },
}
