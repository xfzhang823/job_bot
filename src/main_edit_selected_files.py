import asyncio
import json
import logging
import tempfile
from pathlib import Path
from typing import Tuple, Dict, List
from pydantic import HttpUrl
from utils.pydantic_model_loaders_from_files import (
    load_job_file_mappings_model,
)
from pipelines.resume_editing_pipeline_async import run_resume_editing_pipeline_async
from project_config import (
    ITERATE_0_OPENAI_DIR,
    ITERATE_0_ANTHROPIC_DIR,
    ITERATE_1_OPENAI_DIR,
    ITERATE_1_ANTHROPIC_DIR,
    mapping_file_name,
    OPENAI,
    GPT_4_TURBO,
)

# Set up logging
logger = logging.getLogger(__name__)


def get_mapping_file_and_dirs(llm_provider: str) -> Tuple[Path, Path]:
    """Return the correct mapping file and directories based on the selected LLM provider."""
    if llm_provider.lower() == "openai":
        return (
            ITERATE_0_OPENAI_DIR / mapping_file_name,
            ITERATE_1_OPENAI_DIR / mapping_file_name,
        )
    elif llm_provider.lower() == "anthropic":
        return (
            ITERATE_0_ANTHROPIC_DIR / mapping_file_name,
            ITERATE_1_ANTHROPIC_DIR / mapping_file_name,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


def create_filtered_mapping(
    original_mapping: Dict[str, Dict], custom_urls: List[str]
) -> Dict[str, Dict]:
    """
    Filter the original mapping dictionary so that only entries with keys in custom_urls
    are retained.
    """
    return {
        url: original_mapping[url] for url in custom_urls if url in original_mapping
    }


async def edit_selected_files(
    custom_urls: List[str],
    mapping_file_prev: Path,
    mapping_file_curr: Path,
    llm_provider: str,
    model_id: str,
) -> None:
    """
    Run the asynchronous resume editing pipeline for a selected list of URLs.

    This function creates two simulated mapping files—one for the previous iteration
    and one for the current iteration—each containing only the URLs you want to process,
    while preserving their respective file paths.

    Args:
        custom_urls (List[str]): List of job posting URLs to process.
        mapping_file_prev (Path): Path to the previous iteration mapping file.
        mapping_file_curr (Path): Path to the current iteration mapping file.
        llm_provider (str): The LLM provider to use.
        model_id (str): The specific model ID.
    """
    # Load previous iteration's mappping file
    logger.info(f"Loading previous mapping file: {mapping_file_prev}")
    file_mapping_prev = load_job_file_mappings_model(mapping_file_prev)
    if file_mapping_prev is None:
        logger.error(f"Failed to load mapping file {mapping_file_prev}")
        return

    # Load current mapping file
    logger.info(f"Loading current mapping file: {mapping_file_curr}")
    file_mapping_curr = load_job_file_mappings_model(mapping_file_curr)
    if file_mapping_curr is None:
        logger.error(f"Failed to load mapping file {mapping_file_curr}")
        return

    # Convert each mapping file into dictionaries with URL keys
    mapping_dict_prev: Dict[str, Dict] = {
        str(url): job_file_paths.model_dump()
        for url, job_file_paths in file_mapping_prev.root.items()
    }
    mapping_dict_curr: Dict[str, Dict] = {
        str(url): job_file_paths.model_dump()
        for url, job_file_paths in file_mapping_curr.root.items()
    }

    # Debug
    logger.info(
        f"Available URLs in previous mapping: {list(mapping_dict_prev.keys())[:5]}"
    )
    logger.info(
        f"Available URLs in current mapping: {list(mapping_dict_curr.keys())[:5]}"
    )

    # Filter the mappings for only the custom URLs
    filtered_mapping_prev = create_filtered_mapping(mapping_dict_prev, custom_urls)
    filtered_mapping_curr = create_filtered_mapping(mapping_dict_curr, custom_urls)

    if not filtered_mapping_prev or not filtered_mapping_curr:
        logger.warning("No matching URLs found in one of the mapping files. Exiting.")
        return
    # # Normalize URLs and filter for matches
    # selected_mappings: Dict[str, Dict] = {
    #     url: mapping_dict[url] for url in custom_urls if url in mapping_dict
    # }

    # Create two temporary files for the filtered mappings using tempfile.NamedTemporaryFile
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_prev:
        json.dump(filtered_mapping_prev, temp_prev, indent=2)
        temp_prev_mapping_file = Path(temp_prev.name)
    logger.info(f"Filtered previous mapping saved to {temp_prev_mapping_file}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as temp_curr:
        json.dump(filtered_mapping_curr, temp_curr, indent=2)
        temp_curr_mapping_file = Path(temp_curr.name)
    logger.info(f"Filtered current mapping saved to {temp_curr_mapping_file}")

    # Run the async editing pipeline using the simulated mapping files
    logger.info(
        f"Starting editing process with {llm_provider} ({model_id}) using filtered mappings..."
    )
    await run_resume_editing_pipeline_async(
        mapping_file_prev=temp_prev_mapping_file,
        mapping_file_curr=temp_curr_mapping_file,
        llm_provider=llm_provider,
        model_id=model_id,
    )
    logger.info("Editing process completed.")

    # Optionally, delete the temporary files if no longer needed:
    # temp_prev_mapping_file.unlink()
    # temp_curr_mapping_file.unlink()


def main() -> None:
    """Main function for customization and user input."""
    # Customization (User Input)
    custom_urls_1: List[str] = [
        # "https://boards.greenhouse.io/gleanwork/jobs/4425502005?source=LinkedIn",
        # "https://job-boards.greenhouse.io/airtable/jobs/7603873002?gh_src=aef790d02us",
        # "https://www.accenture.com/us-en/careers/jobdetails?id=R00251798_en&src=LINKEDINJP",
    ]

    custom_urls_2: List[str] = [
        # "https://advisor360.breezy.hr/p/2e1636328c7d-senior-product-manager-ai-analytics-insights",
        # "https://careers.veeva.com/job/365ff44c-8e0a-42b4-a117-27b409a77753/director-crossix-analytics-services-boston-ma/?lever-source=Linkedin",
        # "https://jobs.smartrecruiters.com/Blend360/744000042638791-director-ai-strategy?trid=2d92f286-613b-4daf-9dfa-6340ffbecf73",
        # "https://jobs.us.pwc.com/job/-/-/932/76741173104?utm_source=linkedin.com&utm_campaign=core_media&utm_medium=social_media&utm_content=job_posting&ss=paid&dclid=CjgKEAjwy46_BhD8-aeS9rzGtzsSJAAuE6pojXgWgT7LeiCns3H71Hqcb3dqchcqskpnFxz8njxwwPD_BwE",
        # "https://zendesk.wd1.myworkdayjobs.com/en-US/zendesk/job/San-Francisco-California-United-States-of-America/Competitive-Intelligence-Manager_R30346?source=LinkedIn",
        # "https://salesforce.wd12.myworkdayjobs.com/External_Career_Site/job/California---San-Francisco/Vice-President--Product-Research---Insights_JR279859?source=LinkedIn_Jobs",
        # "https://careers.thomsonreuters.com/us/en/job/THTTRUUSJREQ188456EXTERNALENUS/Director-of-AI-Content-Innovation?utm_source=linkedin&utm_medium=phenom-feeds",
        "https://careers.spglobal.com/jobs/310832?lang=en-us&utm_source=linkedin",
        # "https://bostonscientific.eightfold.ai/careers/job/563602800464180?domain=bostonscientific.com",
    ]

    # Choose which URL list to process
    custom_urls: List[str] = custom_urls_2  # Change to custom_urls_2 if needed

    # LLM settings
    llm_provider: str = OPENAI  # Change to "anthropic" if needed
    model_id: str = GPT_4_TURBO  # Adjust based on provider

    # Get correct mapping file paths
    try:
        mapping_file_prev, mapping_file_curr = get_mapping_file_and_dirs(llm_provider)
    except ValueError as e:
        logger.error(f"Invalid LLM provider: {e}")
        return

    # Run the editing process
    asyncio.run(
        edit_selected_files(
            custom_urls, mapping_file_prev, mapping_file_curr, llm_provider, model_id
        )
    )


if __name__ == "__main__":
    main()
