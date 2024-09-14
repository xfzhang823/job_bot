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
)
from utils.llm_data_utils import (
    read_and_clean_webpage_wt_llama3,
    convert_to_json_wt_gpt,
)
from evaluation_optimization.text_similarity_finder import TextSimilarity
from prompts.prompt_templates import EXTRACT_JOB_REQUIREMENTS_PROMPT


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

        logging.info("Raw GPT response generated.")

        # Parse JSON response
        return json.loads(response_text)

    except json.JSONDecodeError as e:
        logging.info("Error: Unable to parse JSON from the model's output.")
        logging.info(f"Raw response that caused JSONDecodeError: {response_text}")
        logging.error(f"JSON decoding failed: {e}")
        return None


def run_pipeline(
    job_description_url,
    job_descriptions_json_file,
    requirements_json_file,
    resume_json_file,
    text_file_holder,
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
    resume_json = {}

    # Step 1: Check if job posting json file exists or not, and
    # check if the url already exists or not
    job_descriptions, is_existing = load_or_create_json(
        job_descriptions_json_file, job_description_url
    )

    # Check if the current job description already exists by unique ID (URL)
    if is_existing:
        logging.info(
            f"Job description for URL:\n '{job_description_url}' \n"
            f"already exists. Skipping the rest of the preprocessing steps."
        )
        job_description_json = job_descriptions[job_description_url]
    else:
        # **Step 2: Fetch the job description from the URL and save it**
        logging.info(f"Fetching job description from {job_description_url}...")
        job_description_text = read_and_clean_webpage_wt_llama3(job_description_url)

        # Save text in a temp text file (for prototyping/debugging)
        with open(text_file_holder, "w", encoding="utf-8") as f:
            f.write(job_description_text)

        # Convert job description text to JSON
        job_description_json = convert_to_json_wt_gpt(
            job_description_text, primary_key=job_description_url
        )
        add_to_json_file(job_description_json, job_descriptions_json_file)

    # Step 3: Extract key requirements from job description

    # Check if the JSON file exists and load it or create a new one
    requirements_json, is_existing = load_or_create_json(
        requirements_json_file, key=job_description_url
    )

    if is_existing:
        print(f"{requirements_json_file} already exists. Loaded data.")
    else:
        print(f"Extract requirements from job description.")
        requirements_json = extract_job_requirements_with_gpt(
            job_description_json, model_id="gpt-3.5-turbo"
        )
        add_to_json_file(
            {job_description_url: requirements_json}, requirements_json_file
        )

    pretty_print_json(requirements_json)
