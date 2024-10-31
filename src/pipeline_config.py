# pipeline_config.py

from enum import Enum
from config import (
    resume_json_file,
    job_posting_urls_file,
    job_descriptions_json_file,
    job_requirements_json_file,
    mapping_file_name,
    elbow_curve_plot_file_name,
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
    ITERATE_0_CLAUDE_DIR,
    REQS_FILES_ITERATE_0_CLAUDE_DIR,
    RESPS_FILES_ITERATE_0_CLAUDE_CLAUDE,
    PRUNED_RESPS_FILES_ITERATE_0_DIR_CLAUDE,
    SIMILARITY_METRICS_ITERATE_0_CLAUDE_DIR,
    ITERATE_1_CLAUDE_DIR,
    REQS_FILES_ITERATE_1_CLAUDE_DIR,
    RESPS_FILES_ITERATE_1_CLAUDE_DIR,
    SIMILARITY_METRICS_ITERATE_1_CLAUDE_DIR,
    PRUNED_RESPS_FILES_ITERATE_1_CLAUDE_DIR,
)


# Enum for defining pipeline stages
class PipelineStage(Enum):
    PREPROCESSING = "preprocessing"
    EVALUATION = "evaluation"
    EDITING = "editing"


# Dictionary of configurations for each pipeline stage
PIPELINE_CONFIG = {
    "1": {
        "stage": PipelineStage.PREPROCESSING,
        "description": "Preprocessing job posting webpage(s)",
        "function": "run_preprocessing_pipeline",
        "llm_provider": "openai",  # Modify as needed in main
        "io": {
            "openai": {
                "existing_url_list_file": job_descriptions_json_file,
                "url_list_file": job_posting_urls_file,
            },
            "claude": {
                "existing_url_list_file": job_descriptions_json_file,
                "url_list_file": job_posting_urls_file,
            },
        },
    },
    "1_async": {
        "stage": PipelineStage.PREPROCESSING,
        "description": "Async preprocessing job posting webpage(s)",
        "function": "run_preprocessing_pipeline_async",
        "llm_provider": "openai",  # Modify as needed in main
        "io": {
            "openai": {
                "existing_url_list_file": job_descriptions_json_file,
                "url_list_file": job_posting_urls_file,
            },
            "claude": {
                "existing_url_list_file": job_descriptions_json_file,
                "url_list_file": job_posting_urls_file,
            },
        },
    },
    "2a": {
        "stage": PipelineStage.EVALUATION,
        "description": "Create/upsert mapping file for iteration 0",
        "function": "run_upserting_mapping_file_pipeline_iter0",
        "io": {
            "openai": {
                "job_descriptions_file": job_descriptions_json_file,
                "iteration_dir": ITERATE_0_OPENAI_DIR,
                "iteration": 0,
                "mapping_file_name": mapping_file_name,
            },
            "claude": {
                "job_descriptions_file": job_descriptions_json_file,
                "iteration_dir": ITERATE_0_CLAUDE_DIR,
                "iteration": 0,
                "mapping_file_name": mapping_file_name,
            },
        },
    },
    "2b": {
        "stage": PipelineStage.EVALUATION,
        "description": "Flatten responsibilities and requirements files",
        "function": "run_flat_requirements_and_responsibilities_pipeline",
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name,
                "job_requirements_file": job_requirements_json_file,
                "resume_json_file": resume_json_file,
            },
            "claude": {
                "mapping_file": ITERATE_0_CLAUDE_DIR / mapping_file_name,
                "job_requirements_file": job_requirements_json_file,
                "resume_json_file": resume_json_file,
            },
        },
    },
    "2c": {
        "stage": PipelineStage.EVALUATION,
        "description": "Resume evaluation in iteration 0 (add similarity metrics)",
        "function": "run_resume_comparison_pipeline",
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file": ITERATE_0_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "2c_async": {
        "stage": PipelineStage.EVALUATION,
        "description": "Async resume evaluation in iteration 0 (add similarity metrics)",
        "function": "run_resume_comparison_pipeline",
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file": ITERATE_0_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "2d": {
        "stage": PipelineStage.EVALUATION,
        "description": "Add indices to metrics files based on similarity metrics",
        "function": "run_adding_multivariate_indices_mini_pipeline",
        "io": {
            "openai": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
            },
            "claude": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_0_CLAUDE_DIR,
            },
        },
    },
    "2d_async": {
        "stage": PipelineStage.EVALUATION,
        "description": "Async add indices to metrics files based on similarity metrics",
        "function": "run_adding_multivariate_indices_mini_pipeline_async",
        "io": {
            "openai": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_0_OPENAI_DIR,
            },
            "claude": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_0_CLAUDE_DIR,
            },
        },
    },
    "2e": {
        "stage": PipelineStage.EVALUATION,
        "description": "Copy and exclude responsibilities to pruned_responsibilities folder",
        "function": "run_excluding_responsibilities_mini_pipeline",
        "io": {
            "openai": {
                "mapping_file": ITERATE_0_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file": ITERATE_0_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3a": {
        "stage": PipelineStage.EDITING,
        "description": "Create/upsert mapping file for iteration 1",
        "function": "run_upserting_mapping_file_pipeline_iter1",
        "io": {
            "openai": {
                "job_descriptions_file": job_descriptions_json_file,
                "iteration_dir": ITERATE_1_OPENAI_DIR,
                "iteration": 1,
                "mapping_file_name": mapping_file_name,
            },
            "claude": {
                "job_descriptions_file": job_descriptions_json_file,
                "iteration_dir": ITERATE_1_CLAUDE_DIR,
                "iteration": 1,
                "mapping_file_name": mapping_file_name,
            },
        },
    },
    "3b": {
        "stage": PipelineStage.EDITING,
        "description": "Modify responsibilities based on requirements using LLM",
        "function": "run_resume_editing_pipeline",
        "llm_provider": "openai",
        "model_id": "gpt-4-turbo",
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file_prev": ITERATE_0_CLAUDE_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3b_async": {
        "stage": PipelineStage.EDITING,
        "description": "Async modification of responsibilities based on requirements",
        "function": "run_resume_editing_pipeline_async",
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file_prev": ITERATE_0_CLAUDE_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3c": {
        "stage": PipelineStage.EDITING,
        "description": "Copy requirements from iteration 0 to iteration 1",
        "function": "run_copying_requirements_to_next_iteration_mini_pipeline",
        "io": {
            "openai": {
                "mapping_file_prev": ITERATE_0_OPENAI_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file_prev": ITERATE_0_CLAUDE_DIR / mapping_file_name,
                "mapping_file_curr": ITERATE_1_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3d": {
        "stage": PipelineStage.EDITING,
        "description": "Resume evaluation in iteration 1 (add similarity metrics)",
        "function": "run_resume_comparison_pipeline",
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file": ITERATE_1_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3d_async": {
        "stage": PipelineStage.EDITING,
        "description": "Async resume evaluation in iteration 1 (add similarity metrics)",
        "function": "re_run_resume_comparison_pipeline_async",
        "io": {
            "openai": {
                "mapping_file": ITERATE_1_OPENAI_DIR / mapping_file_name,
            },
            "claude": {
                "mapping_file": ITERATE_1_CLAUDE_DIR / mapping_file_name,
            },
        },
    },
    "3e": {
        "stage": PipelineStage.EDITING,
        "description": "Adding multivariate indices to metrics files in iteration 1",
        "function": "run_adding_multivariate_indices_mini_pipeline",
        "io": {
            "openai": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_1_OPENAI_DIR,
            },
            "claude": {
                "csv_files_dir": SIMILARITY_METRICS_ITERATE_1_CLAUDE_DIR,
            },
        },
    },
}
