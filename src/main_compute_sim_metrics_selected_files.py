# Dependencies
import time
from pipelines.hf_cache_refresh_and_lock_pipeline import (
    run_hf_cache_refresh_and_lock_pipeline,
)

# * ✅ Force Transformers & BERTScore to use local cache

# Setting environment variables for Transformers to force the library to work entirely
# from the local cache only!
run_hf_cache_refresh_and_lock_pipeline(refresh_cache=False)

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict
from pydantic import ValidationError
from project_config import (
    ITERATE_0_OPENAI_DIR,
    ITERATE_0_ANTHROPIC_DIR,
    ITERATE_1_OPENAI_DIR,
    ITERATE_1_ANTHROPIC_DIR,
    mapping_file_name,
    OPENAI,
    ANTHROPIC,
)
from pipelines.resume_eval_pipeline_async import (
    run_metrics_processing_pipeline_async,
    run_metrics_re_processing_pipeline_async,
    run_multivariate_indices_processing_mini_pipeline_async,
)
from utils.pydantic_model_loaders_from_files import (
    load_job_file_mappings_model,
)
from utils.generic_utils_async import read_and_validate_json_async, read_json_file_async
from models.resume_job_description_io_models import (
    Requirements,
    NestedResponsibilities,
    JobFileMappings,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ Define URL Lists (Easier Selection)


CUSTOM_URLS: List[str] = [
    # "https://job-boards.greenhouse.io/airtable/jobs/7603873002?gh_src=aef790d02us",
    # "https://www.accenture.com/us-en/careers/jobdetails?id=R00251798_en&src=LINKEDINJP",
    # "https://advisor360.breezy.hr/p/2e1636328c7d-senior-product-manager-ai-analytics-insights",
    # "https://jobs.smartrecruiters.com/Blend360/744000042638791-director-ai-strategy?trid=2d92f286-613b-4daf-9dfa-6340ffbecf73",
    # "https://boards.greenhouse.io/gleanwork/jobs/4425502005?source=LinkedIn",
    # "https://advisor360.breezy.hr/p/2e1636328c7d-senior-product-manager-ai-analytics-insights",
    # "https://careers.veeva.com/job/365ff44c-8e0a-42b4-a117-27b409a77753/director-crossix-analytics-services-boston-ma/?lever-source=Linkedin",
    # "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site/job/California---San-Francisco/Vice-President--Product-Research---Insights_JR279859?source=LinkedIn_Jobs",
    # "https://careers.spglobal.com/jobs/310832?lang=en-us&utm_source=linkedin",
    # "https://bostonscientific.eightfold.ai/careers/job/563602800464180?domain=bostonscientific.com",
    # "https://careers.thomsonreuters.com/us/en/job/THTTRUUSJREQ188456EXTERNALENUS/Director-of-AI-Content-Innovation?utm_source=linkedin&utm_medium=phenom-feeds",
    # "https://jobs.us.pwc.com/job/-/-/932/76741173104?utm_source=linkedin.com&utm_campaign=core_media&utm_medium=social_media&utm_content=job_posting&ss=paid&dclid=CjgKEAjwy46_BhD8-aeS9rzGtzsSJAAuE6pojXgWgT7LeiCns3H71Hqcb3dqchcqskpnFxz8njxwwPD_BwE",
    "https://zendesk.wd1.myworkdayjobs.com/en-US/zendesk/job/San-Francisco-California-United-States-of-America/Competitive-Intelligence-Manager_R30346?source=LinkedIn",
]


async def load_with_default_url(file: Path, model, url: str):
    """
    Load and validate a JSON file using the given Pydantic model.
    If the "url" field is missing, add it using the provided URL.
    Returns the validated model instance or None on failure.
    """
    try:
        # First, try the standard async validation.
        validated = await read_and_validate_json_async(file, model)
        return validated
    except ValidationError as e:
        logger.warning(f"Validation error for {file}: {e}. Attempting to patch 'url'.")
        try:
            content = await read_json_file_async(file)
        except Exception as exc:
            logger.error(f"Failed to read file {file}: {exc}")
            return None
        # If "url" is missing, add it.
        if "url" not in content:
            content["url"] = url
        try:
            # Validate again with the patched data.
            validated = model(**content)
            return validated
        except ValidationError as e2:
            logger.error(f"Re-validation failed for {file} even after patching: {e2}")
            return None


# * timeout is set at 800->for large batch/max concurrent workers, timneout needs to increase
async def compute_similarity_metrics(
    custom_urls: List[str],
    llm_provider: str,
    pipeline_type: str = "flat",
    batch_size: int = 2,
    max_concurrent: int = 3,
) -> None:
    """
    Runs the similarity metrics computation pipeline with strict filtering.
    Ensures that each requirements (or responsibilities) file has a "url" field by patching it if needed.

    Args:
        custom_urls (List[str]): Specific job URLs to process.
        llm_provider (str): LLM provider to use (e.g., "openai", "anthropic").
        pipeline_type (str): Choose "flat" for initial processing, "nested" for reprocessing.
        batch_size (int): Number of tasks per batch.
        max_concurrent (int): Maximum concurrent API calls.

    Returns:
        None
    """
    # Define mapping directories using a lookup dictionary
    mapping_dirs = {
        (OPENAI, "flat"): ITERATE_0_OPENAI_DIR,
        (OPENAI, "nested"): ITERATE_1_OPENAI_DIR,
        (ANTHROPIC, "flat"): ITERATE_0_ANTHROPIC_DIR,
        (ANTHROPIC, "nested"): ITERATE_1_ANTHROPIC_DIR,
    }

    try:
        original_mapping_file = (
            mapping_dirs[(llm_provider.lower(), pipeline_type)] / mapping_file_name
        )
    except KeyError:
        raise ValueError(
            f"Invalid LLM provider ({llm_provider}) or pipeline type ({pipeline_type})."
        )

    logger.info(f"Loading mapping file: {original_mapping_file}")

    file_mapping_model = load_job_file_mappings_model(original_mapping_file)
    if file_mapping_model is None:
        logger.error(f"Failed to load mapping file {original_mapping_file}")
        return

    # Convert mapping to dictionary (keys are URLs)
    mapping_dict: Dict[str, Dict] = {
        str(url): job_file_paths.model_dump()
        for url, job_file_paths in file_mapping_model.root.items()
    }

    logger.info(f"Available URLs in mapping file: {list(mapping_dict.keys())[:5]}")

    # Strictly filter to only process the custom_urls
    selected_mappings = {
        url: mapping_dict[url] for url in custom_urls if url in mapping_dict
    }

    logger.info(f"Filtered mappings (STRICT MATCH): {list(selected_mappings.keys())}")

    if not selected_mappings:
        logger.warning("No matching URLs found in the mapping file. Exiting.")
        return

    # Validate that required files exist and have correct structure
    valid_mappings = {}
    for url, paths in selected_mappings.items():
        reqs_file = Path(paths["reqs"])
        resps_file = Path(paths["resps"])

        valid_reqs = await load_with_default_url(reqs_file, Requirements, url)
        valid_resps = await load_with_default_url(
            resps_file, NestedResponsibilities, url
        )

        if valid_reqs and valid_resps:
            valid_mappings[url] = paths
        else:
            logger.warning(
                f"Skipping invalid files for URL {url}: {reqs_file}, {resps_file}"
            )

    if not valid_mappings:
        logger.error("No valid job descriptions found after filtering. Exiting.")
        return

    # Create a temporary filtered mapping file
    filtered_mapping_file = Path("filtered_mapping.json")
    with open(filtered_mapping_file, "w") as f:
        json.dump(valid_mappings, f, indent=2)

    logger.info(f"Created filtered mapping file: {filtered_mapping_file}")

    # Run the selected similarity metrics pipeline with the filtered mapping file
    pipeline_map = {
        "flat": run_metrics_processing_pipeline_async,
        "nested": run_metrics_re_processing_pipeline_async,
    }
    try:
        pipeline_function = pipeline_map[pipeline_type]
    except KeyError:
        raise ValueError(
            "Invalid pipeline_type. Use 'flat' for initial processing or 'nested' for reprocessing."
        )

    logger.info(
        f"🚀 Running {pipeline_type.upper()} JSON-based similarity metrics computation..."
    )
    await pipeline_function(
        mapping_file=filtered_mapping_file,
        batch_size=batch_size,
        max_concurrent=max_concurrent,
    )
    logger.info(f"✅ Similarity metrics computation ({pipeline_type}) completed.")

    # Now, run the multivariate indices pipeline on the same filtered mapping file
    logger.info("🚀 Running multivariate indices pipeline...")
    await run_multivariate_indices_processing_mini_pipeline_async(
        mapping_file=filtered_mapping_file
    )
    logger.info("✅ Multivariate indices pipeline completed.")


def main():
    """
    Main function to start the similarity computation pipeline with custom URL selection.
    """
    start_time = time.time()
    custom_urls: List[str] = (
        CUSTOM_URLS  # Choose between CUSTOM_URLS_1 or CUSTOM_URLS_2
    )

    llm_provider = OPENAI  # Change to "anthropic" if needed
    pipeline_type = (
        "nested"  # Use "flat" for initial processing or "nested" for reprocessing
    )

    asyncio.run(
        compute_similarity_metrics(
            custom_urls=custom_urls,
            llm_provider=llm_provider,
            pipeline_type=pipeline_type,
        )
    )

    elapsed_time = time.time() - start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    logger.info(f"Total time: {elapsed_time}")


if __name__ == "__main__":
    main()
