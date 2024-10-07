""" 
The pipeline module contains all high-level functions.

This module is to be called by main.py.
"""

# Import libraries
import os
import json
import logging
from dotenv import load_dotenv
import openai
from utils.generic_utils import (
    pretty_print_json,
    load_or_create_json,
    add_to_json_file,
    save_to_json_file,
)

# from utils.webpage_reader import process_webpages_to_json
from utils.webpage_reader import process_webpages_to_json

from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser


# Set up logging
logger = logging.getLogger(__name__)

# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


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


def run_pipeline(
    job_description_url,
    job_descriptions_json_file,
    requirements_json_file,
    # resume_json_file,
):
    """
    Orchestrates the entire pipeline for modifying a resume based on a job description.

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
        # **Step 2: Fetch the job description from the URL and save it**
        logger.info(f"Fetching job description from {job_description_url}...")

        # Convert job description text to JSON
        job_description_json = process_webpages_to_json(job_description_url)

        add_to_json_file(job_description_json, job_descriptions_json_file)

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
