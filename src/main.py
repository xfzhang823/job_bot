# main.py
import logging
import asyncio  # Add this line to import asyncio

from pipelines.run_pipelines import (
    run_pipeline,
    run_pipeline_1_async,
    run_pipeline_2c_async,
    run_pipeline_3b_async,
    run_pipeline_3d_async,
)
from pipeline_config import PIPELINE_CONFIG

# Set up logger
logger = logging.getLogger(__name__)


def execute_pipeline(pipeline_id, llm_provider="openai"):
    """
    Executes a pipeline based on its ID and specified provider.

    Args:
        pipeline_id (str): The identifier of the pipeline to run.
        provider (str): The LLM provider to use, default is 'openai'.
    """
    config = PIPELINE_CONFIG.get(pipeline_id)
    if config is None:
        logger.error(f"No pipeline configuration found for ID: {pipeline_id}")
        return

    description = config["description"]
    is_async = "_async" in pipeline_id

    logger.info(
        f"Executing Pipeline ID '{pipeline_id}' - {description} with provider '{llm_provider}'"
    )

    # Run the pipeline as async or sync based on ID
    if is_async:
        # Use async for pipelines ending in '_async'
        if pipeline_id == "1_async":
            asyncio.run(run_pipeline_1_async())
        elif pipeline_id == "2c_async":
            asyncio.run(run_pipeline_2c_async())
        elif pipeline_id == "3b_async":
            asyncio.run(run_pipeline_3b_async(llm_provider))
        elif pipeline_id == "3d_async":
            asyncio.run(run_pipeline_3d_async())
    else:
        # Sync pipelines
        run_pipeline(pipeline_id, llm_provider=llm_provider)


if __name__ == "__main__":
    # Example of running various pipelines
    # execute_pipeline("1")  # Run pipeline 1 with default OpenAI
    execute_pipeline("2a", llm_provider="openai")  # Run pipeline 2a using Claude
    execute_pipeline("3a", llm_provider="openai")  # Run pipeline 3b using Claude
    # execute_pipeline(
    #     "3d_async", llm_provider="openai"
    # )  # Run async pipeline 3d using OpenAI
