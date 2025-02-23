# pipelines/run_pipelines.py
import os
import logging
import inspect
from typing import Optional
from tqdm import tqdm
import asyncio


from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS

# Pipeline functions
from pipelines.resume_editing_pipeline import (
    run_resume_editing_pipeline as run_resume_editing_pipeline,
)
from pipelines.resume_eval_pipeline import (
    run_metrics_re_processing_pipeline as re_run_resume_comparison_pipeline,
)

from pipelines.resume_editing_pipeline_async import (
    run_resume_editing_pipeline_async as run_resume_editting_pipeline_async,
)
from project_config import (
    OPENAI,
    ANTHROPIC,
    CLAUDE_HAIKU,
    CLAUDE_SONNET,
    GPT_35_TURBO,
    GPT_4_TURBO,
)
from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS

# Set logger
logger = logging.getLogger(__name__)


# Sync run pipeline function
def run_pipeline(
    pipeline_id: str,
    llm_provider: str = OPENAI,
    model_id: Optional[str] = GPT_4_TURBO,
):
    """
    *This is the sync version

    Executes a dynamically specified pipeline based on its configuration in `PIPELINE_CONFIG`.

    This function facilitates running various pipeline functions with unique configurations,
    dynamically handling both the function selection and required arguments:
    - Based on the 'pipeline_id', the function retrieves a specific function name
    and configuration details from `PIPELINE_CONFIG`.
    - Additionally, `run_pipeline` leverages Python's 'inspect' library to examine the function's
    signature, ensuring only the necessary arguments are passed.

    *For example, "llm_provider" and "model_id" are not required for all pipelines; however, they
    *are passed to all pipelines in a standard format:
    - The pipeline retrieves the function (funcs = config.get("function")) and extracts
    input/output settings (io_config) for the specified llm_provider, if applicable.
    - It then checks the function signature (inspect.signature(func)) to determine
    whether llm_provider or model_id is needed, passing only the required arguments
    when calling the function (func(**kwargs)).

    This approach supports flexible configurations, enabling `run_pipeline` to handle
    multiple pipelines with distinct argument requirements without the need to hardcode
    specific functions or argument lists.

    Args:
        - pipeline_id (str): The identifier of the pipeline to execute. This ID is used
        to look up the configuration and function name in `PIPELINE_CONFIG`.
        - llm_provider (str): The LLM provider to use, typically `"openai"` or `"claude"`.
        Defaults to `"openai"`.
        - model_id (str, optional): The specific model ID to use. If `None`,
        the function uses a default based on `llm_provider` or the pipeline
        configuration.

    Raises:
        - FileNotFoundError: If required input/output configurations are missing for
        the specified `llm_provider` in `PIPELINE_CONFIG`.
        - ValueError: If a pipeline function is specified but is not found in
        the global namespace.

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

        * run_pipeline("3b", llm_provider="openai", model_id="gpt-4-turbo")

        This will execute `run_resume_editing_pipeline` with the following arguments:
        - `mapping_file_prev` (from `io_config`)
        - `mapping_file_curr` (from `io_config`)
        - `llm_provider="openai"`
        - `model_id="gpt-4-turbo"`

    How It Works:
    1. The function retrieves the specific configuration for `pipeline_id` from
    `PIPELINE_CONFIG`. This includes the function name and the relevant I/O
    configurations for
    the given `llm_provider`.
    2. The `funcs` key is retrieved and ensured to be a list.
    3. Using `inspect.signature`, the function examines the parameters required by
    each function.
    4. A dictionary `kwargs` is initialized with values from `io_config`.
    5. The `llm_provider` and `model_id` arguments are conditionally added to `kwargs`
    only if they are expected by the function, preventing unexpected argument errors.
    6. Finally, each function is executed sequentially with `func(**kwargs)`.

    This dynamic approach ensures that `run_pipeline` can adapt to different functions,
    configurations, and argument requirements, promoting flexibility and reducing
    the need for code duplication.
    """
    config = PIPELINE_CONFIG.get(pipeline_id)
    if not config:
        raise ValueError(f"Pipeline '{pipeline_id}' not found in configuration.")

    # * Retrieve function reference(s) from pipeline config
    funcs = config.get("function", [])
    if not funcs:
        raise ValueError(f"No function found for pipeline '{pipeline_id}'.")

    # Ensure `funcs` is always a list (convert single function into a list if needed)
    if not isinstance(funcs, list):
        funcs = [funcs]

    # Validate that all elements in the list are callable
    for f in funcs:
        if not callable(f):
            raise TypeError(
                f"Expected a callable function for pipeline '{pipeline_id}', but got {type(f)}"
            )

    # * Extract I/O configuration for the selected LLM provider
    io_config = config.get("io", {}).get(llm_provider, {})
    kwargs = io_config.copy()  # Copy input file paths into kwargs

    # * Set model_id to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    # * Dynamically include any additional arguments (e.g., callables)
    if "kwargs" in config:
        kwargs.update(config["kwargs"])

    # * Execute each function in the list
    for func in funcs:
        local_kwargs = kwargs.copy()  # Create a fresh copy for each function

        # Check function signature and add necessary arguments
        func_signature = inspect.signature(func)
        if "llm_provider" in func_signature.parameters:
            local_kwargs["llm_provider"] = llm_provider
        if "model_id" in func_signature.parameters:
            local_kwargs["model_id"] = model_id

        logger.info(
            f"Running pipeline '{pipeline_id}' with function '{func.__name__}' and kwargs: {local_kwargs}"
        )

        func(**local_kwargs)  # Run the function with appropriate arguments


# Async version
async def run_pipeline_async(
    pipeline_id: str, llm_provider: str = OPENAI, model_id: Optional[str] = None
):
    """
    * Async version: see the sync version docstring for more details.
    Runs an asynchronous pipeline function dynamically.

    Args:
        - pipeline_id (str): The identifier for the pipeline.
        - llm_provider (str): The LLM provider (default: "openai").
        - model_id (Optional[str]): The model ID to use. If None, a default is selected.

    Returns:
        None
    """
    # Step 1: Retrieve pipeline config
    config = PIPELINE_CONFIG.get(pipeline_id)
    if not config:
        raise ValueError(f"Pipeline '{pipeline_id}' not found in configuration.")

    # * Step 2: Retrieve function reference(s) from pipeline config
    funcs = config.get("function", [])
    if not funcs:
        raise ValueError(f"No function found for pipeline '{pipeline_id}'.")

    # Ensure `funcs` is always a list (convert single function into a list if needed)
    if not isinstance(funcs, list):
        funcs = [funcs]

    # Step 3: Validate that all elements in the list are callable
    for f in funcs:
        if not callable(f):
            raise TypeError(
                f"Expected a callable function for pipeline '{pipeline_id}', but got {type(f)}"
            )

    # * Step 4: Extract I/O configuration for the selected LLM provider
    io_config = config.get("io", {}).get(llm_provider, {})
    kwargs = io_config.copy()  # Copy input file paths into kwargs

    # * Step 5: Set model_id to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    # Step 6: Include additional arguments (like callables) dynamically
    if "kwargs" in config:
        kwargs.update(config["kwargs"])

    # * Step 7: Execute each function individually
    for func in funcs:
        logger.info(
            f"Preparing to run async pipeline '{pipeline_id}' with function '{func.__name__}'."
        )

        # Step 7.1: Create a fresh copy of kwargs to avoid mutation issues
        local_kwargs = kwargs.copy()  # Create a fresh copy for each function

        # Step 7.2: Check function signature and filter out unexpected arguments
        func_signature = inspect.signature(func)
        valid_params = func_signature.parameters.keys()
        local_kwargs = {k: v for k, v in local_kwargs.items() if k in valid_params}

        # Step 7.3: Pass `llm_provider` and `model_id` if the function expects them
        if "llm_provider" in func_signature.parameters:
            local_kwargs["llm_provider"] = llm_provider
        if "model_id" in func_signature.parameters:
            local_kwargs["model_id"] = model_id

        logger.info(
            f"Running async pipeline '{pipeline_id}' with function '{func.__name__}' and kwargs: {local_kwargs}"
        )

        # Step 8: Run function asynchronously or in a separate thread if sync
        if inspect.iscoroutinefunction(func):
            logger.info(
                f"Awaiting async function '{func.__name__}' with args: {local_kwargs}"
            )
            await func(**local_kwargs)
        else:
            logger.warning(
                f"Running sync function '{func.__name__}' in a thread to prevent blocking."
            )
            await asyncio.to_thread(func, **local_kwargs)

    logger.info(f"Pipeline '{pipeline_id}' execution completed.")


def run_pipeline_1(llm_provider: str = OPENAI, model_id: str = GPT_4_TURBO) -> None:
    """
    Synchronous pipeline for preprocessing job posting webpages.

    This pipeline identifies and processes new URLs. For each new URL,
    it invokes the `run_preprocessing_pipeline` function to extract relevant
    data and save it. If no new URLs are found, the function logs and exits.
    """
    run_pipeline("1", llm_provider=llm_provider, model_id=model_id)


def run_pipeline_2a(llm_provider: str = OPENAI):
    """
    Pipeline to create/update the mapping file for iteration 0.

    This function updates the mapping file and copies the requirements files from
    the previous directory to the current one.
    """
    run_pipeline("2a", llm_provider=llm_provider)


def run_pipeline_2b(llm_provider: str = OPENAI):
    """
    Pipeline to flatten responsibilities and requirements files for iteration 0.
    """
    run_pipeline("2b", llm_provider=llm_provider)


def run_pipeline_2c(llm_provider: str = OPENAI, model_id: str = GPT_4_TURBO):
    """
    Pipeline to evaluate resumes against job requirements and generate similarity metrics.
    """
    run_pipeline("2c", llm_provider=llm_provider, model_id=model_id)


def run_pipeline_2d(llm_provider: str = OPENAI):
    """
    Pipeline to add extra indices (composite scores and PCA scores) to metrics files
    based on similarity & entailment metrics.
    """
    run_pipeline("2d", llm_provider=llm_provider)


def run_pipeline_2e(llm_provider: str = OPENAI):
    """
    Pipeline to clean up metrics csv files by removing empty rows.
    """
    run_pipeline("2e", llm_provider=llm_provider)


def run_pipeline_2f(llm_provider: str = OPENAI):
    """
    Pipeline to copy files in responsibilities folder to pruned_responsibilities folder,
    and exclude certain responsibilities.
    """
    # Calls `run_pipeline("2f")`, which runs both mini pipelines from PIPELINE_CONFIG
    run_pipeline("2f", llm_provider=llm_provider)


def run_pipeline_3a(llm_provider: str = OPENAI):
    """
    Pipeline to create or upsert the mapping file for iteration 1.
    """
    run_pipeline("3a", llm_provider=llm_provider)


def run_pipeline_3b(llm_provider: str = OPENAI, model_id=GPT_4_TURBO):
    """
    Pipeline to modify responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    run_pipeline("3b", llm_provider=llm_provider, model_id=model_id)


def run_pipeline_3c(llm_provider: str = OPENAI):
    """
    Pipeline to copy requirements files from iteration 0 to iteration 1.
    """
    run_pipeline("3c", llm_provider=llm_provider)


def run_pipeline_3d(llm_provider: str = OPENAI):
    """
    Pipeline to evaluate (modified resumes against job requirements and generate similarity metrics.
    """
    run_pipeline("3d", llm_provider=llm_provider)


def run_pipeline_3e(llm_provider: str = OPENAI):
    """
    Pipeline to add multivariate indices to metrics files in iteration 1.
    """
    run_pipeline("3e", llm_provider=llm_provider)


def run_pipeline_3f(llm_provider: str = OPENAI):
    """
    Pipeline to clean up metrics csv files by removing empty rows.
    """
    run_pipeline("3f", llm_provider=llm_provider)


# * Async Functions
async def run_pipeline_1_async(
    llm_provider: str = OPENAI, model_id: str = GPT_4_TURBO
) -> None:
    """
    Asynchronous pipeline for preprocessing job posting webpages.

    This pipeline identifies and processes new URLs. For each new URL, it invokes the
    `run_preprocessing_pipeline_async` function to extract relevant data and save it.
    If no new URLs are found, the function logs and exits.
    """
    asyncio.run(
        run_pipeline_async("1_async", llm_provider=llm_provider, model_id=model_id)
    )


async def run_pipeline_2c_async(llm_provider: str = OPENAI):
    """
    Async pipeline to evaluate resumes against job requirements and generate similarity \
metrics in iter 0.
    """
    asyncio.run(run_pipeline_async("2_async", llm_provider=llm_provider))


async def run_pipeline_2d_async(llm_provider: str = OPENAI):
    """
    Async ipeline to add extra indices (composite scores and PCA scores) to metrics
    files based on similarity & entailment metrics.
    """
    asyncio.run(run_pipeline_async("2d_async", llm_provider=llm_provider))


async def run_pipeline_3b_async(
    llm_provider: str = OPENAI, model_id: str = GPT_4_TURBO
):
    """
    Async pipeline for modifying responsibilities text based on requirements using LLM.

    Args:
        llm_provider (str): The LLM provider, e.g., 'openai' or 'claude'.
    """
    asyncio.run(
        run_pipeline_async("3b_async", llm_provider=llm_provider, model_id=model_id)
    )


async def run_pipeline_3d_async(llm_provider: str = OPENAI):
    """
    Async pipeline to evaluate resumes against job requirements and generate
    similarity metrics in iter 1.
    """
    asyncio.run(run_pipeline_async("3d_async", llm_provider=llm_provider))


def run_pipeline_3e_async(llm_provider: str = OPENAI):
    """
    Async pipeline to add multivariate indices to metrics files in iteration 1.
    """
    asyncio.run(run_pipeline_async("3e_async", llm_provider=llm_provider))
