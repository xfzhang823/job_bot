""" 
The pipeline module contains all high-level functions.

This module is to be called by main.py.
"""

# Import libraries
import os
import json
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Union
import openai
from utils.generic_utils import (
    fetch_new_urls,
    pretty_print_json,
    load_or_create_json,
    add_to_json_file,
    save_to_json_file,
)
from llm_providers.llm_api_utils_async import call_openai_api_async

# from utils.webpage_reader import process_webpages_to_json
from utils.webpage_reader import process_webpages_to_json

from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from project_config import GPT_35_TURBO, GPT_4_TURBO

# Set up logging
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


async def extract_job_requirements_with_gpt(
    job_description: str, model_id: str = GPT_4_TURBO
):
    """
    Extracts key requirements from the job description using GPT, leveraging the async
    API call utility.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities, and skills.
    """
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)

    try:
        # Make the API call asynchronously using the utility function
        response_model = await call_openai_api_async(
            prompt=prompt,
            model_id=model_id,
            expected_res_type="json",
            json_type="editing",  # assuming we need to validate it as "editing"
            temperature=0.3,
            max_tokens=1500,  # Adjust max_tokens as needed
        )

        # Return the validated response as parsed JSON
        return (
            response_model.data
        )  # Assuming the result is structured in the `data` field
    except Exception as e:
        logger.error(f"Error extracting job requirements: {e}")
        return None


def process_single_url(
    llm_provider: str,
    model_id: str,
    job_description_url: str,
    job_descriptions_json_file: Union[Path, str],
    job_requirements_json_file: Union[Path, str],
):
    """
    Processes a single job description URL: fetches the content from the URL,
    converts it to JSON format, and extracts key job requirements, saving both pieces
    of data to their respective JSON files.

    Args:
        - llm_provider (str): openai or anthropic
        - model_id (str): llm model id (gpt-4, claude-3-sonnet, etc.)
        - job_description_url (str): The URL of the job description to fetch and process.
        - job_descriptions_json_file (Union[Path, str]): Path to the JSON file where
        the job description content will be saved.
        The content is stored in JSON format.
        - job_requirements_json_file (Union[Path, str]): Path to the JSON file where
        the extracted job requirements will be saved. The requirements are extracted using GPT.

    Returns:
        None: This function performs actions that modify external files but does not
        return any values.
    """
    # Ensure paths are Path obj.
    job_descriptions_json_file, job_requirements_json_file = Path(
        job_descriptions_json_file
    ), Path(job_requirements_json_file)

    # Initialize key variables
    job_descriptions = {}
    job_description_json = {}
    job_requirements_json = {}
    # resume_json = {}

    # Step 1: Check if job posting json file exists or not, and
    # check if the url already exists or not
    job_descriptions, is_existing = load_or_create_json(
        job_descriptions_json_file, job_description_url
    )

    # Check if the current job description already exists by unique ID (URL)
    if is_existing:
        logger.info(
            f"Job description for URL:\n '{job_description_url}' \n"
            f"already exists. Skipping the rest of the preprocessing steps."
        )
        job_description_json = job_descriptions[job_description_url]
    else:
        # *Step 2: Fetch the job description from the URL and save to JSON file
        logger.info(f"Fetching job description from {job_description_url}...")

        # Convert job description text to JSON
        job_description_json = process_webpages_to_json(job_description_url)

        add_to_json_file(job_description_json, job_descriptions_json_file)

        logger.info(
            "job posting webpage(s) processed; job descriptions JSON file updated."
        )

    # Step 3: Extract key requirements from job description

    # Check if the JSON file exists and load it or create a new one
    job_requirements_json, is_existing = load_or_create_json(
        job_requirements_json_file, key=job_description_url
    )

    if is_existing:
        logger.info(f"{job_requirements_json_file} already exists. Loaded data.")
    else:
        logger.info(f"Extract requirements from job description.")
        job_requirements_json = extract_job_requirements_with_gpt(
            job_description_json, model_id="gpt-3.5-turbo"
        )
        add_to_json_file(
            {job_description_url: job_requirements_json}, job_requirements_json_file
        )


# Orchestrator function: run preprocessing pipeline (check new URLs and process them)
def run_preprocessing_pipeline(
    job_posting_urls_file: Union[Path, str],
    job_descriptions_json_file: Union[Path, str],
    job_requirements_json_file: Union[Path, str],
    llm_provider: str = "openai",  # llm_provider is passed from the orchestrating function
    model_id: str = GPT_4_TURBO,  # default to gpt 4 turbo
) -> None:
    """
    Orchestrates the preprocessing pipeline, checking for new URLs and processing them.

    This function handles checking for new URLs and calling process_single_url function to
    - fetch job descriptions from those URLs, and
    - extract relevant information using GPT.

    It invokes process_single_url for each new URL.

    Args:
        - llm_provider (str): The LLM provider to use (default is "openai").
        - job_posting_urls_file: Union[Path, str]: Path to the job postings JSON file with URLs.
        - job_descriptions_json_file Union[Path, str]: Path to the job description JSON file.
        - requirements_json_file Union[Path, str]: Path to the extracted job requirements JSON file.

    Returns:
        None
    """
    logger.info(f"Starting preprocessing pipeline using provider {llm_provider}")

    # Change file paths to Path if not already
    job_descriptions_json_file, job_posting_urls_file, job_requirements_json_file = (
        Path(job_descriptions_json_file),
        Path(job_posting_urls_file),
        Path(job_requirements_json_file),
    )

    # Fetch new URLs to process
    new_urls = fetch_new_urls(
        existing_url_list_file=job_requirements_json_file,
        url_list_file=job_posting_urls_file,
    )

    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    logger.info(f"New URLs found: \n{new_urls}")  # Debugging

    for url in new_urls:
        logger.info(f"Processing for URL: {url}")
        process_single_url(
            llm_provider=llm_provider,
            model_id=model_id,
            job_description_url=url,
            job_descriptions_json_file=job_descriptions_json_file,
            job_requirements_json_file=job_requirements_json_file,
        )

    logger.info("Finished preprocessing pipeline.")
