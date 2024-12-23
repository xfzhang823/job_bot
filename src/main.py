# main.py
import logging
import asyncio  # Add this line to import asyncio

from pipelines.run_pipelines import (
    run_pipeline,
    run_pipeline_async,
)
from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS
from project_config import CLAUDE_SONNET

# Set up logger
logger = logging.getLogger(__name__)


# High level function to call the run_pipeline functions (sync and async)
def execute_pipeline(pipeline_id, llm_provider="openai", model_id=None):
    """
    *Executes a pipeline by dynamically selecting between synchronous and
    *asynchronous execution paths.

    This function serves as the main entry point for executing pipelines
    based on their ID and provider. It references configurations stored in
    'PIPELINE_CONFIG' and calls one of two modular functions:
    - 'run_pipeline` for synchronous pipelines,
    - or `run_pipeline_async` for asynchronous ones.
    (clean separation between sync and async pipeline handling, modularity, code reusability.)

    `execute_pipeline` uses the `_async` suffix in `pipeline_id` to determine
    whether a pipeline is asynchronous:
    - If `is_async` is `True`, the function calls `run_pipeline_async`,
    wrapped in `asyncio.run()` to handle the
    event loop.
    - Otherwise, it calls `run_pipeline` directly for synchronous execution.

    Organization:
        * 'main.py': The `execute_pipeline` function orchestrates pipeline execution,
        * choosing between `run_pipeline` and `run_pipeline_async` based on `is_async`.
        - `run_pipelines.py`: Contains `run_pipeline` and `run_pipeline_async`,
        which dynamically lookup and execute specific pipeline functions
        (e.g., `run_pipeline_3b_async` or `run_pipeline_2c_async`)
        using `globals()`.
        - It enables us to manage multiple pipelines without needing to import each
        individual pipeline function into `main.py`.

    Args:
        - pipeline_id (str): The identifier of the pipeline to execute.
        This ID maps to specific configurations and functions in `PIPELINE_CONFIG`.
        - llm_provider (str): The LLM provider to use, typically "openai" or "claude".
        Defaults to "openai".
        - model_id (str, optional): The specific model ID to use:
            If `None`, the function retrieves a default model ID based on `llm_provider`
            or `PIPELINE_CONFIG`.

    Raises:
        - FileNotFoundError: If required input/output configurations are missing for
        the specified `llm_provider` in `PIPELINE_CONFIG`.
        - ValueError: If a pipeline function specified in `PIPELINE_CONFIG` cannot be
        found in the global namespace.

    Example:
        To execute a synchronous pipeline with ID "3b" and OpenAI as the provider:
            execute_pipeline("3b", llm_provider="openai", model_id="gpt-4-turbo")

        To execute an asynchronous pipeline with ID "3d_async" and Claude as the provider:
            execute_pipeline("3d_async", llm_provider="claude")
    """
    config = PIPELINE_CONFIG.get(pipeline_id)
    if config is None:
        logger.error(f"No pipeline configuration found for ID: {pipeline_id}")
        return

    # Set model_id to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    description = config["description"]
    is_async = "_async" in pipeline_id

    logger.info(
        f"Executing Pipeline ID '{pipeline_id}' - {description} with provider \
            '{llm_provider}' and model ID '{model_id}'"
    )

    # Run the appropriate pipeline function based on async/sync type
    if is_async:
        asyncio.run(
            run_pipeline_async(
                pipeline_id, llm_provider=llm_provider, model_id=model_id
            )
        )
    else:
        run_pipeline(pipeline_id, llm_provider=llm_provider, model_id=model_id)


def main_claude():
    """Executing the pipeline"""
    # execute_pipeline("1")  # Run pipeline 1 with default OpenAI

    # execute_pipeline("2a", llm_provider="claude")  # Run pipeline 2a in Claude I/O

    # execute_pipeline("3a", llm_provider="claude")  # Run pipeline 3b in Claude I/O

    # execute_pipeline("3b", llm_provider="claude")  # Run pipeline 3b using Claude

    # execute_pipeline(
    #     "3b_async", llm_provider="claude"
    # )  # Run pipeline 3b_async using Claude

    # execute_pipeline("3c", llm_provider="claude")  # Run pipeline 3c in Claude I/O

    # execute_pipeline(
    #     "3d_async", llm_provider="claude"
    # )  # Run async pipeline 3d in Claude I/O

    execute_pipeline("3e", llm_provider="claude")  # Run pipeline 3e using Claude


if __name__ == "__main__":
    main_claude()
