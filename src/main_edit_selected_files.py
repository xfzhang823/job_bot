import asyncio
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List
from pydantic import HttpUrl
from evaluation_optimization.create_mapping_file import load_mappings_model_from_json
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


async def edit_selected_files(
    custom_urls: List[str],
    mapping_file_prev: Path,
    mapping_file_curr: Path,
    llm_provider: str,
    model_id: str,
) -> None:
    """
    Run the asynchronous resume editing pipeline for a selected list of URLs.

    Args:
        custom_urls (List[str]): List of job posting URLs to process.
        mapping_file_prev (Path): Path to the previous iteration mapping file.
        mapping_file_curr (Path): Path to the current iteration mapping file.
        llm_provider (str): The LLM provider to use.
        model_id (str): The specific model ID.
    """
    logger.info(f"Loading mapping file: {mapping_file_prev}")
    file_mapping_model = load_mappings_model_from_json(mapping_file_prev)

    if file_mapping_model is None:
        logger.error(f"Failed to load mapping file {mapping_file_prev}")
        return

    # Convert HttpUrl keys to str and JobFilePaths (Pydantic model) to dict
    mapping_dict: Dict[str, Dict] = {
        str(url): job_file_paths.model_dump()
        for url, job_file_paths in file_mapping_model.root.items()
    }

    logger.info(
        f"Available URLs in mapping file: {list(mapping_dict.keys())[:5]}"
    )  # Debugging

    # Normalize URLs and filter for matches
    selected_mappings: Dict[str, Dict] = {
        url: mapping_dict[url] for url in custom_urls if url in mapping_dict
    }

    if not selected_mappings:
        logger.warning("No matching URLs found in the mapping file. Exiting.")
        return

    # Save filtered mapping to a temporary file
    temp_mapping_file = Path("filtered_mapping.json")
    with open(temp_mapping_file, "w") as f:
        json.dump({"root": selected_mappings}, f, indent=2)

    logger.info(f"Filtered mapping saved to {temp_mapping_file}")

    # Run the async editing pipeline
    logger.info(f"Starting editing process with {llm_provider} ({model_id})...")
    await run_resume_editing_pipeline_async(
        mapping_file_prev=mapping_file_prev,
        mapping_file_curr=mapping_file_curr,
        llm_provider=llm_provider,
        model_id=model_id,
    )
    logger.info("Editing process completed.")


def main() -> None:
    """Main function for customization and user input."""
    # Customization (User Input)
    custom_urls_1: List[str] = [
        # "https://boards.greenhouse.io/gleanwork/jobs/4425502005?source=LinkedIn",
        # "https://job-boards.greenhouse.io/airtable/jobs/7603873002?gh_src=aef790d02us",
        # "https://www.accenture.com/us-en/careers/jobdetails?id=R00251798_en&src=LINKEDINJP",
    ]

    custom_urls_2: List[str] = [
        "https://advisor360.breezy.hr/p/2e1636328c7d-senior-product-manager-ai-analytics-insights",
        "https://jobs.smartrecruiters.com/Blend360/744000042638791-director-ai-strategy?trid=2d92f286-613b-4daf-9dfa-6340ffbecf73",
    ]

    # Choose which URL list to process
    custom_urls: List[str] = custom_urls_1  # Change to custom_urls_2 if needed

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
