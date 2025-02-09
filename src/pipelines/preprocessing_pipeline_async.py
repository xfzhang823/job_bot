""" 
The pipeline module contains all high-level functions.

This module is to be called by main.py.
"""

# Import libraries
import os
import json
import logging
from dotenv import load_dotenv
from typing import Union
from pathlib import Path
import asyncio
import nest_asyncio
import openai
from llm_providers.llm_api_utils_async import (
    call_anthropic_api_async,
    call_openai_api_async,
)
from utils.generic_utils import (
    fetch_new_urls,
    pretty_print_json,
    load_or_create_json,
    read_from_json_file,
    add_to_json_file,
    save_to_json_file,
)
from utils.webpage_reader_async import process_webpages_to_json_async
from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from models.llm_response_models import JobSiteResponse, RequirementsResponse
from project_config import GPT_4_TURBO, GPT_35_TURBO, CLAUDE_HAIKU, CLAUDE_SONNET

# Set up logging
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


async def extract_job_requirements_with_openai_async(
    job_description: str, model_id: str = GPT_35_TURBO
):
    """
    Extracts key requirements from the job description using GPT.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities,
        and skills.
    """
    if not job_description:
        logger.error("job_description text is empty or invalid.")
        raise ValueError("job_description text cannot be empty.")

    # Set up the prompt
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)
    logger.info(f"Prompt to extract job requirements:\n{prompt}")

    # Call the async OpenAI API
    response_model = await call_openai_api_async(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        json_type="requirements",
        temperature=0.3,
        max_tokens=2000,
    )
    logger.info(f"Validated LLM Response Model: {response_model}")

    # Validate that the response is in the expected JobSiteResponseModel format
    if not isinstance(response_model, RequirementsResponse):
        logger.error(
            "Received response is not in expected JobSiteResponseModel format."
        )
        raise ValueError(
            "Received response is not in expected JobSiteResponseModel format."
        )

    # Return the model-dumped dictionary (from Pydantic obj to dict)
    return response_model.model_dump()


async def extract_job_requirements_with_anthropic_async(
    job_description: str, model_id: str = GPT_35_TURBO
):
    """
    Extracts key requirements from the job description using Anthropic.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities,
        and skills.
    """
    if not job_description:
        logger.error("job_description text is empty or invalid.")
        raise ValueError("job_description text cannot be empty.")

    # Set up the prompt
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)
    logger.info(f"Prompt to extract job requirements:\n{prompt}")

    # Call the async OpenAI API
    response_model = await call_anthropic_api_async(
        prompt=prompt,
        model_id=model_id,
        expected_res_type="json",
        json_type="requirements",
        temperature=0.3,
        max_tokens=2000,
    )
    logger.info(f"Validated LLM Response Model: {response_model}")

    # Validate that the response is in the expected JobSiteResponseModel format
    if not isinstance(response_model, RequirementsResponse):
        logger.error(
            "Received response is not in expected JobSiteResponseModel format."
        )
        raise ValueError(
            "Received response is not in expected JobSiteResponseModel format."
        )

    # Return the model-dumped dictionary (from Pydantic obj to dict)
    return response_model.model_dump()


async def process_single_url_async(
    job_description_url: str,
    job_descriptions_json_file: Path,
    job_requirements_json_file: Path,
    lock: asyncio.Lock,
    llm_provider: str = "opeanai",
    model_id: str = GPT_4_TURBO,
) -> None:
    """
    Asynchronously processes a single job posting URL, fetches the job description,
    and extracts job requirements using GPT.

    Args:
        - job_description_url (str): The job description URL to process.
        - job_descriptions_json_file (Path): Path to the job descriptions JSON file.
        - job_requirements_json_file (Path): Path to the job requirements JSON file.
        - lock (asyncio.Lock): A lock to ensure thread safety during critical sections.
        #* The lock is put in the function b/c each URL processing must be done before
        #* it is saved to file.
        - llm_provider (str): The LLM provider to use (default is "openai").
        - model_id (str): The model ID to use for GPT.

    Returns:
        None
    """
    logger.info(f"Processing URL: {job_description_url}")

    # Step 1: Webpage -> job description JSON format
    job_description_dict = await process_webpages_to_json_async(
        urls=job_description_url, llm_provider=llm_provider, model_id=model_id
    )  # Returns a dict: Dict[str, Dict[str, Any]]
    logger.info(f"Job description fetched for URL: {job_description_url}")
    logger.info(job_description_dict)

    #! According to OpenAI, you shouldn't passing JSON str into LLM;
    #! howeever, according to Anthropic and DeepSeek, JSON input is fine.
    # Step 2: Converst job_description from dict to json string
    job_description_json = json.dumps(job_description_dict, indent=2)

    # Extract job requirements using openai or anthropic
    if llm_provider.lower() == "openai":
        requirements_dict = await extract_job_requirements_with_openai_async(
            job_description=job_description_json, model_id=model_id
        )
    elif llm_provider.lower() == "anthropic":
        requirements_dict = await extract_job_requirements_with_anthropic_async(
            job_description=job_description_json, model_id=model_id
        )
    else:
        raise ValueError(f"{llm_provider} is not a supported API.")

    logger.info(f"Job Requirements: \n{requirements_dict}")

    # Use the lock when writing to files to avoid race conditions
    async with lock:
        add_to_json_file(
            job_description_dict, job_descriptions_json_file
        )  # Add the dict to job descriptions JSON file)
        add_to_json_file(
            {job_description_url: requirements_dict}, job_requirements_json_file
        )

    logger.info(
        f"Job description and requirements processed for URL: {job_description_url}"
    )


async def run_preprocessing_pipeline_async(
    job_posting_urls_file: Union[Path, str],
    job_descriptions_json_file: Union[Path, str],
    job_requirements_json_file: Union[Path, str],
    llm_provider: str = "openai",  # llm_provider is passed from the orchestrating function
    model_id: str = GPT_4_TURBO,  # default to gpt 4 turbo
):
    """
    Asynchronous pipeline for preprocessing job posting webpage(s):
    - Orchestrates the entire pipeline for modifying a resume based on a job description.
    - If there are multiple URL links, iterate through them and run the `process_single_url`
    function for each.
    - The pipeline processes job descriptions and extracts job requirements for each URL.

    Args:
        - job_posting_urls_file (str or Path): Path to the job postings JSON file with URLs.
        - job_descriptions_json_file (str or Path): Path to the job description JSON file.
        - job_requirements_json_file (str or Path): Path to the extracted job requirements
        JSON file.
        - llm_provider (str): The LLM provider to use (default is "openai").
        - model_id (str): The model ID to use for GPT (default is "gpt-4-turbo").

    Returns:
        None
    """
    logger.info("Starting preprocessing pipeline async...")

    # Convert file paths to Path obj. if not already
    job_descriptions_json_file, job_posting_urls_file, job_requirements_json_file = (
        Path(job_descriptions_json_file),
        Path(job_posting_urls_file),
        Path(job_requirements_json_file),
    )

    # Step 1: Error handling - check if job posting json file exists or not
    if not job_posting_urls_file.exists():
        raise FileNotFoundError(
            f"Job posting URLs file not found: {job_posting_urls_file}"
        )

    # Step 2: Check if requirements JSON file exists or not
    if job_requirements_json_file.exists():
        # If so, fetch new URLs to process
        new_urls = fetch_new_urls(
            existing_url_list_file=job_requirements_json_file,
            url_list_file=job_posting_urls_file,
        )
    else:
        # If not, all URLs are new -> to be processed
        data = read_from_json_file(job_posting_urls_file)
        jobs = data.get("job", [])
        new_urls = list(set([jobs.get("url") for job in jobs]))  # Unique URLs only

    if not new_urls:
        logger.info("No new URLs found. Skipping...")
        return

    # Step 3: Process URLs

    # Set list of urls to be processed
    job_posting_urls = new_urls
    logger.info(f"Found {len(job_posting_urls)} URLs to process.")

    # Create an async lock for file writes to avoid race conditions
    lock = asyncio.Lock()

    # Process each URL asynchronously with a lock to avoid race conditions when writing
    # to files: each task is a single URL processing; whenever a task is done,
    # it's saved (no need to wait for other url processing)
    tasks = []
    for job_description_url in job_posting_urls:
        tasks.append(
            process_single_url_async(
                job_description_url=job_description_url,
                job_descriptions_json_file=job_descriptions_json_file,
                job_requirements_json_file=job_requirements_json_file,
                lock=lock,  # pass the lock
                llm_provider=llm_provider,
                model_id=model_id,
            )
        )  # * task.append to set up the coroutine

    # Await the completion of all the tasks (URLs processed)
    await asyncio.gather(*tasks)  # * task.gather to execute the coroutine

    logger.info("Preprocessing pipieline finished.")

    # logger.info("Finished processing all URLs.")

    # # Check if the current job description already exists by unique ID (URL)
    # if is_existing:
    #     logger.info(
    #         f"Job description for URL:\n '{job_description_url}' \n"
    #         f"already exists. Skipping the rest of the preprocessing steps."
    #     )
    #     job_description_json = job_descriptions[job_description_url]
    # else:
    #     # **Step 2: Async fetch the job description from the URL and save it**
    #     logger.info(f"Fetching job description from {job_description_url}...")

    #     # Convert job description text to JSON
    #     job_description_json = await process_webpages_to_json_async(job_description_url)
    #     logger.info(f"job description json: {job_description_json}")  # debugging

    #     add_to_json_file(job_description_json, job_descriptions_json_file)

    #     logger.info(
    #         "job posting webpage(s) processed; job descriptions JSON file updated."
    #     )

    # # Step 3: Extract key requirements from job description

    # # Check if the JSON file exists and load it or create a new one
    # job_requirements_json, is_existing = load_or_create_json(
    #     job_requirements_json_file, key=job_description_url
    # )

    # if is_existing:
    #     logger.info(f"{job_requirements_json_file} already exists. Loaded data.")
    # else:
    #     logger.info(f"Extract requirements from job description.")
    #     job_requirements_json = extract_job_requirements_with_gpt(
    #         job_description_json, model_id="gpt-3.5-turbo"
    #     )
    #     add_to_json_file(
    #         {job_description_url: job_requirements_json}, job_requirements_json_file
    #     )

    # def extract_flatten_resps_and_reqs(resume_json_file, requirements_json_file):
    # """Function to:
    # * read resume and job requirments JSON files
    # * flatten and nornalize them in dictionary format

    # Args:
    #     resume_json_file (str.): resume JSON file path.
    #     requirements_json_file (str.): extracted requirements (from job posting) JSON file path.

    # Return:
    #     responsibilities (dict), requiremnts (dict)

    # Example format:
    # {"json index": "text"}
    # """
    # # SParse and flatten responsibilities from resume (as a dict)
    # resume_parser = ResumeParser(resume_json_file)
    # resps_flat = (
    #     resume_parser.extract_and_flatten_responsibilities()
    # )  # extract as a dict

    # # Parse and flatten job requirements (as a dict) or
    # # parse/flatten/conncactenate into a single string
    # job_reqs_parser = JobRequirementsParser(
    #     requirements_json_file
    # )  # todo: missing url parameter
    # reqs_flat = job_reqs_parser.extract_flatten_reqs()  # extract as a dict

    # logger.info("Responsibilities and requirements extracted and flatted.")
    # return resps_flat, reqs_flat
