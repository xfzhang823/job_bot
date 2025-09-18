# pipelines/run_pipelines.py
import logging
import asyncio
import inspect
from typing import Optional
from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS
from project_config import (
    OPENAI,
    ANTHROPIC,
    GPT_4_1_NANO,
    GPT_35_TURBO,
    CLAUDE_HAIKU,
    CLAUDE_SONNET_3_5,
)

# Set logger
logger = logging.getLogger(__name__)


# Sync run pipeline function
def run_pipeline(
    pipeline_id: str,
    llm_provider: str = OPENAI,
    model_id: Optional[str] = GPT_4_1_NANO,
    filter_keys: Optional[
        list[str]
    ] = None,  # ✅ NEW (for processing a small batch of urls)
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

        # * ✅ New: add filter_keys if the function supports it
        if filter_keys and "filter_keys" in func_signature.parameters:
            local_kwargs["filter_keys"] = filter_keys

        logger.info(
            f"Running pipeline '{pipeline_id}' with function '{func.__name__}' and kwargs: {local_kwargs}"
        )

        func(**local_kwargs)  # Run the function with appropriate arguments


# Async version
async def run_pipeline_async(
    pipeline_id: str,
    llm_provider: str = OPENAI,
    model_id: Optional[str] = None,
    filter_keys: Optional[
        list[str]
    ] = None,  # ✅ NEW (for processing a small batch of urls)
) -> None:
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
    logger.info(f"Starting async pipeline: {pipeline_id}")

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

        # * ✅ New: add filter_keys if the function supports it
        if filter_keys and "filter_keys" in func_signature.parameters:
            local_kwargs["filter_keys"] = filter_keys

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
