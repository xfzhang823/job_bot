# pipelines/run_pipelines.py
import os
import logging
import inspect
from typing import Optional
from tqdm import tqdm
import asyncio


from utils.generic_utils import fetch_new_urls
from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS

# Pipeline functions
from pipelines.preprocessing_pipeline_async import (
    run_pipeline_async as run_preprocessing_pipeline_async,
)
from pipelines.preprocessing_pipeline import (
    run_pipeline as run_preprocessing_pipeline,
)
from pipelines.upserting_mapping_mini_pipeline_iter0 import (
    run_pipeline as run_upserting_mapping_file_pipeline_iter0,
)
from pipelines.flattened_resps_reqs_processing_mini_pipeline import (
    run_pipeline as run_flat_requirements_and_responsibilities_mini_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_processing_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline import (
    multivariate_indices_processing_mini_pipeline as run_adding_multivariate_indices_mini_pipeline,
)
from pipelines.exclude_responsibilities_mini_pipeline import (
    run_pipeline as run_excluding_responsibilities_mini_pipeline,
)
from pipelines.resume_pruning_pipeline import (
    run_pipeline as run_resume_pruning_pipeline,
)
from pipelines.copying_resps_to_pruned_resps_dir_mini_pipeline import (
    run_pipe_line as run_copying_resps_to_pruned_resps_mini_pipeline,
)
from pipelines.copying_reqs_to_next_iter_mini_pipeline import (
    run_pipeline as run_copying_requirements_to_next_iteration_mini_pipeline,
)
from pipelines.upserting_mapping_mini_pipeline_iter1 import (
    run_pipeline as run_upserting_mapping_file_pipeline_iter1,
)
from pipelines.resume_editing_pipeline import (
    run_pipeline as run_resume_editing_pipeline,
)
from pipelines.resume_eval_pipeline import (
    metrics_re_processing_pipeline as re_run_resume_comparison_pipeline,
)
from pipelines.resume_eval_pipeline_async import (
    metrics_processing_pipeline_async as run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    metrics_re_processing_pipeline_async as re_run_resume_comparison_pipeline_async,
)
from pipelines.resume_eval_pipeline_async import (
    multivariate_indices_processing_mini_pipeline_async as run_adding_multivariate_indices_mini_pipeline_async,
)
from pipelines.resume_editing_pipeline_async import (
    run_pipeline_async as run_resume_editting_pipeline_async,
)


# Set logger
logger = logging.getLogger(__name__)


def run_pipeline(
    pipeline_id: str, llm_provider: str = "openai", model_id: Optional[str] = None
):
    """
    *Thi is the sync version

    Executes a dynamically specified pipeline based on its configuration in `PIPELINE_CONFIG`.

    This function facilitates running various pipeline functions with unique configurations,
    dynamically handling both the function selection and required arguments:
    - Based on the 'pipeline_id', the function retrieves a specific function name
    and configuration details from `PIPELINE_CONFIG`.
    - Using 'globals()', the function name (stored as a string) is dynamically converted to
    a callable function.
    - Additionally, `run_pipeline` leverages Python's `inspect` library to examine the function's
    signature, ensuring only the necessary arguments are passed.

    Note:
        *'globals()' is used here as a lookup tool to dynamically access function objects
        *based on their names stored as strings, not to make a function accessible outside
        *of a local function.

    This approach supports flexible configurations, enabling `run_pipeline` to handle
    multiple pipelines with distinct argument requirements without the need to hardcode
    specific functions or argument lists. This makes `run_pipeline` adaptable to different
    processes and configurations, promoting modularity and reusability.

    Args:
        - pipeline_id (str): The identifier of the pipeline to execute. This ID is used to look up
        the configuration and function name in `PIPELINE_CONFIG`.
        - llm_provider (str): The LLM provider to use, typically `"openai"` or `"claude"`.
        Defaults to `"openai"`.
        - model_id (str, optional): The specific model ID to use. If `None`, the function uses
        a default based on `llm_provider` or the pipeline configuration.

    Raises:
        FileNotFoundError: If required input/output configurations are missing for the specified
            `llm_provider` in `PIPELINE_CONFIG`.
        ValueError: If a pipeline function is specified but is not found in the global namespace.

    Example:
        Suppose `PIPELINE_CONFIG` has a pipeline with ID `"3b"` configured as follows:

            PIPELINE_CONFIG = {
                "3b": {
                    "function": "run_resume_editing_pipeline",
                    "io": {
                        "openai": {
                            "mapping_file_prev": "path/to/prev/file",
                            "mapping_file_curr": "path/to/curr/file"
                        }
                    }
                }
            }

        Then, to run this pipeline with OpenAI as the provider and a specific model:

        *    run_pipeline("3b", llm_provider="openai", model_id="gpt-4-turbo")

        This will execute `run_resume_editing_pipeline` with the following arguments:
        - `mapping_file_prev` (from `io_config`)
        - `mapping_file_curr` (from `io_config`)
        - `llm_provider="openai"`
        - `model_id="gpt-4-turbo"`

    How It Works:
    1. The function retrieves the specific configuration for `pipeline_id` from
    `PIPELINE_CONFIG`. This includes the function name and the relevant I/O configurations for
    the given `llm_provider`.
    2. The `func_name` is retrieved as a string and converted to a function object
    using 'globals()'.
    3. Using `inspect.signature`, the function examines the parameters required by 'func_name'.
    4. A dictionary `kwargs` is initialized with values from `io_config`.
    5. The `llm_provider` and `model_id` arguments are conditionally added to `kwargs`
    only if they are expected by the function, preventing unexpected argument errors.
    6. Finally, the dynamically selected function is executed with `func(**kwargs)`.

    This dynamic approach ensures that `run_pipeline` can adapt to different functions,
    configurations, and argument requirements, promoting flexibility and reducing
    the need for code duplication.
    """

    config = PIPELINE_CONFIG[pipeline_id]
    func_name = config["function"]
    io_config = config["io"][llm_provider]

    # Use the provided model_id or fallback to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    logger.info(
        f"Running pipeline '{config['description']}' for provider '{llm_provider}' with model ID '{model_id}'"
    )

    # Prepare arguments dynamically based on the function's signature
    func = globals()[func_name]  # Retrieve the function object from globals
    func_signature = inspect.signature(func)  # Get function's parameters
    kwargs = io_config.copy()  # Start with io_config as the base arguments

    # Add llm_provider and model_id only if the function accepts them
    if "llm_provider" in func_signature.parameters:
        kwargs["llm_provider"] = llm_provider
    if "model_id" in func_signature.parameters:
        kwargs["model_id"] = model_id

    # Log for debugging purposes
    logger.info(f"Calling function '{func_name}' with kwargs: {kwargs}")

    # Call the function with the dynamically prepared kwargs
    func(**kwargs)


# Async function
async def run_pipeline_async(
    pipeline_id: str, llm_provider: str = "openai", model_id: Optional[str] = None
):
    """
    *Async version of the run_pipelines function

    Executes an asynchronous pipeline based on its configuration in `PIPELINE_CONFIG`.
    Similar to `run_pipeline`, but designed for async pipelines.

    Args:
        pipeline_id (str): The identifier of the pipeline to execute.
        llm_provider (str): The LLM provider to use, typically "openai" or "claude".
        model_id (str, optional): The specific model ID to use; if None, defaults will be used.
    """
    config = PIPELINE_CONFIG[pipeline_id]
    func_name = config["function"]
    io_config = config["io"][llm_provider]

    # Set model_id to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    logger.info(
        f"Running async pipeline '{config['description']}' for provider '{llm_provider}' with model ID '{model_id}'"
    )

    # Retrieve the async function object and examine its signature
    func = globals()[func_name]
    func_signature = inspect.signature(func)
    kwargs = io_config.copy()  # Start with io_config as the base arguments

    # Add llm_provider and model_id only if the async function accepts them
    if "llm_provider" in func_signature.parameters:
        kwargs["llm_provider"] = llm_provider
    if "model_id" in func_signature.parameters and model_id:
        kwargs["model_id"] = model_id

    # Await the async function call with the prepared kwargs
    await func(**kwargs)


def run_pipeline_1(llm_provider: str = "openai"):
    """
    Synchronous pipeline for preprocessing job posting webpages.

    This pipeline identifies and processes new URLs. For each new URL,
    it invokes the `run_preprocessing_pipeline` function to extract relevant
    data and save it. If no new URLs are found, the function logs and exits.
    """
    config = PIPELINE_CONFIG["1"]
    io_config = config["io"][llm_provider]

    new_urls = fetch_new_urls(
        existing_url_list_file=io_config["existing_url_list_file"],
        url_list_file=io_config["url_list_file"],
    )
    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):
        logger.info(f"Processing for URL: {url}")
        run_preprocessing_pipeline(
            job_description_url=url,
            job_descriptions_json_file=io_config["existing_url_list_file"],
            requirements_json_file=io_config["url_list_file"],
        )


def run_pipeline_2a(llm_provider: str = "openai"):
    """
    Pipeline to create/update the mapping file for iteration 0.

    This function updates the mapping file and copies the requirements files from
    the previous directory to the current one.
    """
    config = PIPELINE_CONFIG["2a"]
    io_config = config["io"][llm_provider]

    run_upserting_mapping_file_pipeline_iter0(
        job_descriptions_file=io_config["job_descriptions_file"],
        iteration_dir=io_config["iteration_dir"],
        iteration=io_config["iteration"],
        mapping_file_name=io_config["mapping_file_name"],
    )


def run_pipeline_2b(llm_provider: str = "openai"):
    """
    Pipeline to flatten responsibilities and requirements files for iteration 0.
    """
    config = PIPELINE_CONFIG["2b"]
    io_config = config["io"][llm_provider]

    run_flat_requirements_and_responsibilities_mini_pipeline(
        mapping_file=io_config["mapping_file"],
        job_requirements_file=io_config["job_requirements_file"],
        resume_json_file=io_config["resume_json_file"],
    )


def run_pipeline_2c(llm_provider: str = "openai"):
    """
    Pipeline to evaluate resumes against job requirements and generate similarity metrics.
    """
    config = PIPELINE_CONFIG["2c"]
    io_config = config["io"][llm_provider]

    if not os.path.exists(io_config["mapping_file"]):
        logger.error(f"Mapping file not found: {io_config['mapping_file']}")
        raise FileNotFoundError(
            f"Mapping file not found at {io_config['mapping_file']}"
        )

    run_resume_comparison_pipeline(io_config["mapping_file"])


def run_pipeline_2d(llm_provider: str = "openai"):
    """
    Pipeline to add indices to metrics files based on similarity metrics.
    """
    config = PIPELINE_CONFIG["2d"]
    io_config = config["io"][llm_provider]

    run_adding_multivariate_indices_mini_pipeline(io_config["mapping_file"])


def run_pipeline_2e(llm_provider: str = "openai"):
    """
    Pipeline to copy files in responsibilities folder to pruned_responsibilities folder,
    and exclude certain responsibilities.
    """
    config = PIPELINE_CONFIG["2e"]
    io_config = config["io"][llm_provider]

    run_copying_resps_to_pruned_resps_mini_pipeline(io_config["mapping_file"])
    run_excluding_responsibilities_mini_pipeline(io_config["mapping_file"])


def run_pipeline_3a(llm_provider: str = "openai"):
    """
    Pipeline to create or upsert the mapping file for iteration 1.
    """
    config = PIPELINE_CONFIG["3a"]
    io_config = config["io"][llm_provider]

    run_upserting_mapping_file_pipeline_iter1(
        job_descriptions_file=io_config["job_descriptions_file"],
        iteration_dir=io_config["iteration_dir"],
        iteration=io_config["iteration"],
        mapping_file_name=io_config["mapping_file_name"],
    )


def run_pipeline_3b(llm_provider: str = "openai"):
    """
    Pipeline to modify responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    config = PIPELINE_CONFIG["3b_async"]
    io_config = config["io"][llm_provider]

    # Retrieve and validate model_id
    model_id = config.get("model_id")
    if model_id is None:
        raise ValueError(
            f"Model ID is not defined for pipeline '3b_async' using provider '{llm_provider}'."
        )

    # Validate other essential inputs
    mapping_file_prev = io_config.get("mapping_file_prev")
    mapping_file_curr = io_config.get("mapping_file_curr")
    if not mapping_file_prev or not mapping_file_curr:
        raise ValueError(
            f"One or more required I/O configurations are missing for provider '{llm_provider}'."
        )

    run_resume_editting_pipeline(
        mapping_file_prev=mapping_file_prev,
        mapping_file_curr=mapping_file_curr,
        llm_provider=llm_provider,
        model_id=model_id,
    )


def run_pipeline_3c(llm_provider: str = "openai"):
    """
    Pipeline to copy requirements files from iteration 0 to iteration 1.
    """
    config = PIPELINE_CONFIG["3c"]
    io_config = config["io"][llm_provider]

    run_copying_requirements_to_next_iteration_mini_pipeline(
        mapping_file_prev=io_config["mapping_file_prev"],
        mapping_file_curr=io_config["mapping_file_curr"],
    )


def run_pipeline_3d(llm_provider: str = "openai"):
    """
    Pipeline to match resume's responsibilities to job postings' requirements
    and generate similarity metrics.
    """
    config = PIPELINE_CONFIG["3d"]
    io_config = config["io"][llm_provider]

    re_run_resume_comparison_pipeline(io_config["mapping_file"])


def run_pipeline_3e(llm_provider: str = "openai"):
    """
    Pipeline to add multivariate indices to metrics files in iteration 1.
    """
    config = PIPELINE_CONFIG["3e"]
    io_config = config["io"][llm_provider]

    run_adding_multivariate_indices_mini_pipeline(io_config["mapping_file"])


async def run_pipeline_1_async(llm_provider: str = "openai"):
    """
    Async pipeline for preprocessing job posting webpage(s).
    """
    config = PIPELINE_CONFIG["1_async"]
    io_config = config["io"][llm_provider]

    new_urls = fetch_new_urls(
        existing_url_list_file=io_config["existing_url_list_file"],
        url_list_file=io_config["url_list_file"],
    )
    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    for url in tqdm(new_urls, desc="Processing job postings", unit="job"):
        await run_preprocessing_pipeline_async(
            job_description_url=url,
            job_descriptions_json_file=io_config["existing_url_list_file"],
            requirements_json_file=io_config["url_list_file"],
        )


async def run_pipeline_2c_async(llm_provider: str = "openai"):
    """
    Async pipeline for resume evaluation in iteration 0.
    """
    config = PIPELINE_CONFIG["2c_async"]
    io_config = config["io"][llm_provider]

    if not os.path.exists(io_config["mapping_file"]):
        raise FileNotFoundError(
            f"Mapping file not found at {io_config['mapping_file']}"
        )

    await run_resume_comparison_pipeline_async(io_config["mapping_file"])


async def run_pipeline_2d_async(llm_provider: str = "openai"):
    """
    Async pipeline for adding indices to metrics files.
    """
    config = PIPELINE_CONFIG["2d_async"]
    io_config = config["io"][llm_provider]

    await run_adding_multivariate_indices_mini_pipeline_async(
        io_config["csv_files_dir"]
    )


async def run_pipeline_3b_async(llm_provider: str, model_id: str):
    """
    Async pipeline for modifying responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    config = PIPELINE_CONFIG["3b_async"]
    io_config = config["io"][llm_provider]

    # Validate other essential inputs
    mapping_file_prev = io_config.get("mapping_file_prev")
    mapping_file_curr = io_config.get("mapping_file_curr")
    if not mapping_file_prev or not mapping_file_curr:
        raise ValueError(
            f"One or more required I/O configurations are missing for provider '{llm_provider}'."
        )
    await run_resume_editting_pipeline_async(
        mapping_file_prev=mapping_file_prev,
        mapping_file_curr=mapping_file_curr,
        llm_provider=llm_provider,
        model_id=model_id,
    )


async def run_pipeline_3d_async(llm_provider: str):
    """
    Async version of the resume comparison pipeline for iteration 1.
    """
    config = PIPELINE_CONFIG["3d_async"]
    io_config = config["io"][llm_provider]

    await re_run_resume_comparison_pipeline_async(io_config["mapping_file"])
