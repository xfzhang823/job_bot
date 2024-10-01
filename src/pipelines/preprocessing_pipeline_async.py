""" 
The pipeline module contains all high-level functions.

This module is to be called by main.py.
"""

# Import libraries
import os
import json
import logging
from dotenv import load_dotenv
import asyncio
import nest_asyncio
import openai
from utils.generic_utils import (
    pretty_print_json,
    load_or_create_json,
    add_to_json_file,
    save_to_json_file,
)

from utils.webpage_reader_async import process_webpages_to_json_async
from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser


nest_asyncio.apply()

# Set up logging
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def fetch_new_urls(existing_url_list_file, url_list_file):
    """
    Fetch job posting URLs from the new jobs file, compare them with the existing jobs file,
    and return only the new URLs that have not been previously processed.

    Args:
        job_descriptions_json_file (str): File path of the JSON file containing already processed job postings.
                                          Typically from 'job_postings.json'.
        list_of_urls_file (str): File path of the JSON file containing the list of new job posting URLs.

    Returns:
        list: A list of URLs that are new and have not been processed yet.

    Notes:
        - Both files must be in valid JSON format.
        - The existing job descriptions file should use URLs as keys.
        - The new jobs file should have a structure where job postings are listed under "jobs" and each job has a "url" field.
    """

    # Load new job posting URLs from the file
    with open(url_list_file, "r") as f:
        new_job_data = json.load(f)

    # Extract URLs from the new job postings
    urls_to_check = [job["url"] for job in new_job_data["jobs"]]

    # Load the existing job descriptions from the existing JSON file
    with open(existing_url_list_file, "r") as f:
        existing_job_data = json.load(f)

    # Extract existing URLs (assuming URLs are the keys)
    existing_urls = list(existing_job_data.keys())

    # Filter out URLs that already exist
    new_urls = [url for url in urls_to_check if url not in existing_urls]

    if not new_urls:
        logger.info("No new URLs found.")  # Optional logging or handling

    return new_urls


def extract_job_requirements_with_gpt(job_description, model_id="gpt-3.5-turbo"):
    """
    Extracts key requirements from the job description using GPT.

    Args:
        job_description (str): The full text of the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Extracted requirements including qualifications, responsibilities, and skills.
    """
    prompt = EXTRACT_JOB_REQUIREMENTS_PROMPT.format(content=job_description)

    try:
        response = openai.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Lower temperature for more factual extraction
        )
        response_text = response.choices[0].message.content.strip()

        logger.info("Raw GPT response generated.")

        # Parse JSON response
        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logger.info("Error: Unable to parse JSON from the model's output.")
        logger.info(f"Raw response that caused JSONDecodeError: {response_text}")
        logger.error(f"JSON decoding failed: {e}")
        return None


def extract_flatten_resps_and_reqs(resume_json_file, requirements_json_file):
    """Function to:
    * read resume and job requirments JSON files
    * flatten and nornalize them in dictionary format

    Args:
        resume_json_file (str.): resume JSON file path.
        requirements_json_file (str.): extracted requirements (from job posting) JSON file path.

    Return:
        responsibilities (dict), requiremnts (dict)

    Example format:
    {"json index": "text"}
    """
    # SParse and flatten responsibilities from resume (as a dict)
    resume_parser = ResumeParser(resume_json_file)
    resps_flat = (
        resume_parser.extract_and_flatten_responsibilities()
    )  # extract as a dict

    # Parse and flatten job requirements (as a dict) or
    # parse/flatten/conncactenate into a single string
    job_reqs_parser = JobRequirementsParser(requirements_json_file)
    reqs_flat = job_reqs_parser.extract_flatten_reqs()  # extract as a dict

    logger.info("Responsibilities and requirements extracted and flatted.")
    return resps_flat, reqs_flat


async def run_pipeline_async(
    job_description_url: list,
    job_descriptions_json_file: str,
    requirements_json_file: str,
    resume_json_file: str,
    responsibilities_flat_json_file: str,
    requirements_flat_json_file: str,
):
    """
    Orchestrates the entire pipeline for modifying a resume based on a job description.
    The webpage reading/converting to JSON part is async.

    Args:
        - job_description_url (str): URL of the job description.
        - job_descriptions_json_file (str): Path to the job description JSON file.
        - requirements_json_file (str): Path to the extracted job requirements JSON file.
        - resume_json_file (str): Path to the resume JSON file.
        - text_file_holder (str): Path to the temporary text file holder for job description content.

    Returns:
        None
    """
    # Initialize key variables
    job_descriptions = {}
    job_description_json = {}
    requirements_json = {}
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
        # **Step 2: Async fetch the job description from the URL and save it**
        logger.info(f"Fetching job description from {job_description_url}...")

        # Convert job description text to JSON
        job_description_json = await process_webpages_to_json_async(job_description_url)
        logger.info(f"job description json: {job_description_json}")  # debugging

        await add_to_json_file(job_description_json, job_descriptions_json_file)

        logger.info(
            "job posting webpage(s) processed; job descriptions JSON file updated."
        )

    # Step 3: Extract key requirements from job description

    # Check if the JSON file exists and load it or create a new one
    requirements_json, is_existing = load_or_create_json(
        requirements_json_file, key=job_description_url
    )

    if is_existing:
        logger.info(f"{requirements_json_file} already exists. Loaded data.")
    else:
        logger.info(f"Extract requirements from job description.")
        requirements_json = extract_job_requirements_with_gpt(
            job_description_json, model_id="gpt-3.5-turbo"
        )
        add_to_json_file(
            {job_description_url: requirements_json}, requirements_json_file
        )

    # Step 4: Extract, flatten, and save responsibilities and requirements
    # to JSON files

    # Check if the json files exist already
    if os.path.exists(responsibilities_flat_json_file) and os.path.exists(
        requirements_flat_json_file
    ):
        logger.info("Responsibilities and requirements flat JSON files already exist.")

    else:
        resps_flat, reqs_flat = extract_flatten_resps_and_reqs(
            resume_json_file=resume_json_file,
            requirements_json_file=requirements_json_file,
        )

        save_to_json_file(resps_flat, responsibilities_flat_json_file)
        logger.info("Responsibilities in flat dictionary format extracted and saved.")

        save_to_json_file(reqs_flat, requirements_flat_json_file)
        logger.info("Requirments in flat dictionary format extracted and saved.")
