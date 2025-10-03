"""
Filename: main.py
Author: Xiao-Fei Zhang

This module serves as the entry point for executing various pipeline processes.
It dynamically selects and executes a specific pipeline based on the given pipeline ID.

Key Steps:
1. **main.py**: The entry point of the execution. It calls the `execute_pipeline` function
   to start the process.

2. **execute_pipeline**: This function dynamically selects the appropriate pipeline
   to run based on the `pipeline_id` passed to it. It automatically detects whether
   the pipeline is synchronous or asynchronous (by `_async` suffix) and routes
   execution accordingly.

3. **run_pipeline / run_pipeline_async**: These functions retrieve pipeline configurations
from `pipeline_config.py` and map them to the corresponding implementation functions
(e.g., `run_preprocessing_pipeline`, `run_resume_editing_pipeline_async`).

4. **pipeline_config.py**: This file contains the configuration that maps pipeline IDs to
   specific functions and input/output files, defining how the data flows and is processed
   through various stages.

5. **Run each individual function pipeline**:
   - For example, `run_preprocessing_pipeline`: This pipeline processes job posting URLs,
     checks for new URLs, and extracts job descriptions.

6. **Optional filtering with filter_keys**:
   - The `execute_pipeline` function supports a `filter_keys` argument,
     allowing users to selectively run pipelines on specific job URLs.
   - This improves flexibility by avoiding full-batch processing when testing or debugging.

7. **Move onto the next step**: Once the pipeline finishes, it proceeds to the next stage
   as defined in the pipeline configuration.

This structure allows for easy expansion of new pipelines with minimal code changes,
facilitating modular, reusable code for various stages of the process.
"""

# Dependencies
from job_bot.pipelines.hf_cache_refresh_and_lock_pipeline import (
    run_hf_cache_refresh_and_lock_pipeline,
)

# * ✅ Force Transformers & BERTScore to use local cache

# Setting environment variables for Transformers to force the library to work entirely
# from the local cache only!
run_hf_cache_refresh_and_lock_pipeline(refresh_cache=False)


import logging
import asyncio  # Add this line to import asyncio
import matplotlib

# User defined
from job_bot.pipelines.run_pipelines import (
    run_pipeline,
    run_pipeline_async,
)
from job_bot.pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS
from job_bot.pipelines.filter_job_posting_urls_mini_pipeline import (
    run_filtering_job_posting_urls_mini_pipe_line,
)
from job_bot.config.project_config import (
    OPENAI,
    ANTHROPIC,
    CLAUDE_SONNET_3_5,
    GPT_35_TURBO,
    GPT_4_1_NANO,
    CLAUDE_HAIKU,
)


matplotlib.use("Agg")  # Prevent interactive mode

# Set up logger
logger = logging.getLogger(__name__)

# Optional
urls_to_filter = [
    "https://jobs.takeda.com/job/-/-/1113/79540724304",
    "https://jobs.johnsoncontrols.com/job/WD30231669",
]


# High level function to call the run_pipeline functions (sync and async)
def execute_pipeline(
    pipeline_id,
    llm_provider="openai",
    model_id=None,
    filter_keys: list[str] | None = None,  # ✅ NEW; selected urls to process
):
    """
    *Executes a pipeline by dynamically selecting between synchronous and
    *asynchronous execution paths.

    This function serves as the main entry point for executing pipelines
    based on their ID and provider. It references configurations stored in
    'PIPELINE_CONFIG' and calls one of two modular functions:
    - `run_pipeline` for synchronous pipelines,
    - or `run_pipeline_async` for asynchronous ones.
    (clean separation between sync and async pipeline handling, modularity,
    code reusability.)

    `execute_pipeline` uses the `_async` suffix in `pipeline_id` to determine
    whether a pipeline is asynchronous:
    - If `is_async` is `True`, the function calls `run_pipeline_async`,
      wrapped in `asyncio.run()` to handle the event loop.
    - Otherwise, it calls `run_pipeline` directly for synchronous execution.

    #* Optional Filtering
        - This function also supports selective execution via the `filter_keys` argument.
        - This allows running only a subset of job postings (by URL) instead of
        the full batch.

    Organization:
        * 'main.py': The `execute_pipeline` function orchestrates pipeline execution,
        * choosing between `run_pipeline` and `run_pipeline_async` based on `is_async`.
        - `run_pipelines.py`: Contains `run_pipeline` and `run_pipeline_async`,
        which dynamically lookup and execute specific pipeline functions
        (e.g., `run_resume_editing_pipeline_async` or `run_metrics_processing_pipeline_async`)
        using `PIPELINE_CONFIG`.
        - It enables us to manage multiple pipelines without needing to import each
        individual pipeline function into `main.py`.

    Args:
        - pipeline_id (str): The identifier of the pipeline to execute.
          This ID maps to specific configurations and functions in `PIPELINE_CONFIG`.
        - llm_provider (str): The LLM provider to use, typically "openai" or "anthropic".
          Defaults to "openai".
        - model_id (str, optional): The specific model ID to use.
          If `None`, the function retrieves a default model ID based on `llm_provider`
          or `PIPELINE_CONFIG`.
        - filter_keys (list[str] | None): Optional list of job posting URLs to process.
          If provided, only matching URLs will be run through the pipeline. Defaults to None.

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

        To run an async pipeline with a filtered subset of jobs:
            execute_pipeline("3b_async",
                            llm_provider="openai",
                            model_id="gpt-4",
                            filter_keys=["https://job1.com", "https://job2.com"])
    """

    logger.info(f"Starting execution for pipeline: {pipeline_id}")

    config = PIPELINE_CONFIG.get(pipeline_id)
    if config is None:
        logger.error(f"No pipeline configuration found for ID: {pipeline_id}")
        return

    # Set model_id to default if not provided
    model_id = model_id or config.get("model_id", DEFAULT_MODEL_IDS.get(llm_provider))

    description = config["description"]
    is_async = "_async" in pipeline_id

    logger.info(
        f"Executing Pipeline ID '{pipeline_id}' - {description} with provider '{llm_provider}' and model ID '{model_id}'"
    )

    # Run the appropriate pipeline function based on async/sync type
    try:
        if is_async:
            asyncio.run(
                run_pipeline_async(
                    pipeline_id,
                    llm_provider=llm_provider,
                    model_id=model_id,
                    filter_keys=filter_keys,
                )
            )
        else:
            run_pipeline(
                pipeline_id,
                llm_provider=llm_provider,
                model_id=model_id,
                filter_keys=filter_keys,
            )
    except Exception as e:
        logger.error(f"Error executing pipeline '{pipeline_id}': {e}", exc_info=True)

    logger.info(f"Finished execution for pipeline: {pipeline_id}")


def main_anthropic():
    """Executing the pipeline using Anthropic models (e.g., Claude)"""

    # #! ☑️ Step 0: Masking/Filter URLs to run a small batch (temporary step)
    # run_filtering_job_posting_urls_mini_pipe_line()

    # Define default Anthropic model
    model_id = CLAUDE_HAIKU

    # # ✅ Step 1: Preprocessing job posting webpages
    # execute_pipeline("1_async", llm_provider=ANTHROPIC, model_id=model_id)

    # # ✅ Step 2: Creating/updating mapping file for iteration 0
    # execute_pipeline("2a", llm_provider=ANTHROPIC)

    # ✅ Step 3: Extracting & Flattening Job Requirements and Responsibilities
    execute_pipeline("2b", llm_provider=ANTHROPIC)

    # # ✅ Step 4: Resume Evaluation (Calculate Similarity/Entailment Metrics)
    # execute_pipeline("2c_async", llm_provider=ANTHROPIC)

    # # ✅ Step 5: Add Composite Scores & PCA Scores to Metrics
    # execute_pipeline("2d_async", llm_provider=ANTHROPIC)

    # # ✅ Step 6: Clean Up Sim Metrics CSV Files (Removing Empty Rows)
    # execute_pipeline("2e", llm_provider=ANTHROPIC)

    # # ✅ Step 7: Copy & Prune Responsibilities
    # execute_pipeline("2f", llm_provider=ANTHROPIC)

    # ✅ Step 8: Creating/updating mapping file for iteration 1
    execute_pipeline("3a", llm_provider=ANTHROPIC)

    # # ✅ Step 9: Modify Responsibilities Based on Requirements from Iter 0
    # # & Save to Iter 1
    # execute_pipeline("3b_async", llm_provider=ANTHROPIC, model_id=model_id)

    # # ✅ Step 10: Copy Requirements from Iteration 0 to Iteration 1
    # execute_pipeline("3c", llm_provider=ANTHROPIC)

    # # ✅ Step 11: Async Resume Evaluation in Iteration 1
    # execute_pipeline("3d_async", llm_provider=ANTHROPIC, model_id=model_id)

    # # ✅ Step 12: Add Multivariate Indices to Metrics Files in Iteration 1
    # execute_pipeline("3e_async", llm_provider=ANTHROPIC, model_id=model_id)

    # # ✅ Step 13: Clean Metrics Files in Iteration 1
    # execute_pipeline("3f", llm_provider=ANTHROPIC, model_id=model_id)


def main_openai():
    """Executing the pipeline using OpenAI models (e.g., GPT)"""

    # Define default OpenAI model
    # model_id = GPT_4_TURBO

    # # ☑️ Step 0: Masking/Filter URLs to run a small batch (temporary step)
    # run_filtering_job_posting_urls_mini_pipe_line()

    # # ✅ Step 1: Preprocessing job posting webpages
    # execute_pipeline("1_async", llm_provider=OPENAI, model_id=GPT_35_TURBO)

    # # ✅ Step 2: Creating/updating mapping file for iteration 0
    # execute_pipeline("2a", llm_provider=OPENAI)

    # # ✅ Step 3: Extracting & Flattening Job Requirements and Responsibilities
    # execute_pipeline("2b", llm_provider=OPENAI)

    # # ✅ Step 4: Resume Evaluation (Calculate Similarity/Entailment Metrics)
    # execute_pipeline("2c_async", llm_provider=OPENAI, filter_keys=urls_to_filter)

    # # ✅ Step 5: Add Composite Scores & PCA Scores to Metrics
    # execute_pipeline("2d_async", llm_provider=OPENAI)

    # # ✅ Step 6: Clean Up Sim Metrics CSV Files (Removing Empty Rows)
    # execute_pipeline("2e", llm_provider=OPENAI)

    # # ✅ Step 7: Copy & Prune Responsibilities
    # execute_pipeline("2f", llm_provider=OPENAI)

    # ✅ Step 8: Creating/updating mapping file for iteration 1
    execute_pipeline("3a", llm_provider=OPENAI)

    # # ✅ Step 9: Modify Responsibilities Based on Requirements from Iter 0
    # # & Save to Iter 1
    # execute_pipeline(
    #     "3b_async",
    #     llm_provider=OPENAI,
    #     model_id=GPT_4_TURBO,
    #     filter_keys=urls_to_filter,
    # )

    # # ✅ Step 10: Copy Requirements from Iteration 0 to Iteration 1
    # execute_pipeline("3c", llm_provider=OPENAI)

    # # ✅ Step 11: Async Resume Evaluation in Iteration 1
    # execute_pipeline("3d_async", llm_provider=OPENAI, filter_keys=urls_to_filter)

    # # ✅ Step 11: Add Multivariate Indices to Metrics Files in Iteration 1
    # execute_pipeline("3e_async", llm_provider=OPENAI)

    # # ✅ Step 12: Clean Metrics Files in Iteration 1
    # execute_pipeline("3f", llm_provider=OPENAI)


if __name__ == "__main__":

    # main_openai()  # * Execute the OpenAI pipeline by calling main_openai

    main_anthropic()  # * Execute the OpenAI pipeline by calling anthropic
