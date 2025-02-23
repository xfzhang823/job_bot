"""
Filename: main.py

This module serves as the entry point for executing various pipeline processes. 
It dynamically selects and executes a specific pipeline based on the given pipeline ID. 

Key Steps:
1. **main.py**: The entry point of the execution. It calls the `execute_pipeline` function 
   to start the process.

2. **execute_pipeline**: This function dynamically selects the appropriate pipeline 
   to run based on the `pipeline_id` passed to it.

3. **run_pipeline**: This function retrieves the pipeline configuration from 
'pipeline_config.py' and maps it to the corresponding function 
(e.g., 'run_preprocessing_pipeline').

4. **pipeline_config.py**: This file contains the configuration that maps pipeline IDs to 
   specific functions and input/output files, defining how the data flows and is processed 
   through various stages.

5. **Run each individual function pipeline**: 
   - For example, `run_preprocessing_pipeline`: This pipeline processes job posting URLs, 
     checks for new URLs, and extracts job descriptions.

6. **Move onto the next step**: Once the pipeline finishes, it proceeds to the next stage 
as defined in the pipeline configuration.

This structure allows for easy expansion of new pipelines with minimal code changes, 
facilitating modular, reusable code for various stages of the process.
"""

# Dependencies
import logging
import asyncio  # Add this line to import asyncio
import matplotlib

# User defined
from pipelines.run_pipelines import (
    run_pipeline,
    run_pipeline_async,
)
from pipeline_config import PIPELINE_CONFIG, DEFAULT_MODEL_IDS
from project_config import OPENAI, ANTHROPIC, CLAUDE_SONNET, GPT_4_TURBO, CLAUDE_HAIKU

# Set up logger
logger = logging.getLogger(__name__)


matplotlib.use("Agg")


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
        - llm_provider (str): The LLM provider to use, typically "openai" or "anthropic".
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


def main_anthropic():
    """Executing the pipeline using Anthropic models (e.g., Claude)"""

    # Define default Anthropic model
    model_id = CLAUDE_HAIKU

    # ✅ Step 1: Preprocessing job posting webpages
    execute_pipeline("1_async", llm_provider=ANTHROPIC, model_id=model_id)

    # ✅ Step 2: Creating/updating mapping file for iteration 0
    execute_pipeline("2a", llm_provider=ANTHROPIC)

    # ✅ Step 3: Extracting & Flattening Job Requirements and Responsibilities
    execute_pipeline("2b", llm_provider=ANTHROPIC)

    # ✅ Step 4: Resume Evaluation (Calculate Similarity/Entailment Metrics)
    execute_pipeline("2c_async", llm_provider=ANTHROPIC)

    # ✅ Step 5: Add Composite Scores & PCA Scores to Metrics
    execute_pipeline("2d_async", llm_provider=ANTHROPIC)

    # ✅ Step 6: Clean Up Sim Metrics CSV Files (Removing Empty Rows)
    execute_pipeline("2e", llm_provider=ANTHROPIC)

    # ✅ Step 7: Copy & Prune Responsibilities
    execute_pipeline("2f", llm_provider=ANTHROPIC)

    # ✅ Step 8: Creating/updating mapping file for iteration 1
    execute_pipeline("3a", llm_provider=ANTHROPIC)

    # ✅ Step 9: Modify Responsibilities Based on Requirements from Iter 0
    # & Save to Iter 1
    execute_pipeline("3b_async", llm_provider=ANTHROPIC, model_id=model_id)

    # ✅ Step 10: Copy Requirements from Iteration 0 to Iteration 1
    execute_pipeline("3c", llm_provider=ANTHROPIC)

    # ✅ Step 11: Async Resume Evaluation in Iteration 1
    execute_pipeline("3d_async", llm_provider=ANTHROPIC, model_id=model_id)

    # ✅ Step 12: Add Multivariate Indices to Metrics Files in Iteration 1
    execute_pipeline("3e_async", llm_provider=ANTHROPIC, model_id=model_id)

    # ✅ Step 13: Clean Metrics Files in Iteration 1
    execute_pipeline("3f", llm_provider=ANTHROPIC, model_id=model_id)


def main_openai():
    """Executing the pipeline using OpenAI models (e.g., GPT)"""

    # Define default OpenAI model
    model_id = GPT_4_TURBO

    # ✅ Step 1: Preprocessing job posting webpages
    execute_pipeline("1_async", llm_provider=OPENAI, model_id=model_id)

    # ✅ Step 2: Creating/updating mapping file for iteration 0
    execute_pipeline("2a", llm_provider=OPENAI)

    # ✅ Step 3: Extracting & Flattening Job Requirements and Responsibilities
    execute_pipeline("2b", llm_provider=OPENAI)

    # ✅ Step 4: Resume Evaluation (Calculate Similarity/Entailment Metrics)
    execute_pipeline("2c_async", llm_provider=OPENAI)

    # ✅ Step 5: Add Composite Scores & PCA Scores to Metrics
    execute_pipeline("2d_async", llm_provider=OPENAI)

    # ✅ Step 6: Clean Up Sim Metrics CSV Files (Removing Empty Rows)
    execute_pipeline("2e", llm_provider=OPENAI)

    # ✅ Step 7: Copy & Prune Responsibilities
    execute_pipeline("2f", llm_provider=OPENAI)

    # ✅ Step 8: Iteration 1 - Modify Responsibilities Based on Requirements
    execute_pipeline("3a", llm_provider=OPENAI)
    execute_pipeline("3b_async", llm_provider=OPENAI, model_id=model_id)

    # ✅ Step 9: Copy Requirements from Iteration 0 to Iteration 1
    execute_pipeline("3c", llm_provider=OPENAI)

    # ✅ Step 10: Async Resume Evaluation in Iteration 1
    execute_pipeline("3d_async", llm_provider=OPENAI, model_id=model_id)

    # ✅ Step 11: Add Multivariate Indices to Metrics Files in Iteration 1
    execute_pipeline("3e_asyc", llm_provider=OPENAI, model_id=model_id)

    # ✅ Step 12: Clean Metrics Files in Iteration 1
    execute_pipeline("3f", llm_provider=OPENAI, model_id=model_id)


if __name__ == "__main__":
    # main_openai()  # Execute the OpenAI pipeline by calling main_openai
    main_anthropic()
