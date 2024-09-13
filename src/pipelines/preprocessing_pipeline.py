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
from utils.utils import (
    pretty_print_json,
    load_or_create_json,
    read_and_clean_webpage_wt_llama3,
    convert_to_json_wt_gpt,
    add_to_json_file,
    read_from_json_file,
)
from matching.text_similarity_finder import TextSimilarity
from prompts.prompts import EXTRACT_JOB_REQUIREMENTS_PROMPT


# Load API keys
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def analyze_resume_against_requirements(
    resume_json, requirements, model_id="gpt-4-turbo"
):
    """
    Analyzes the resume JSON against the job requirements and suggests modifications.

    Args:
        resume_json (dict): The JSON object containing the resume details.
        requirements (dict): The extracted requirements from the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        str: JSON-formatted suggestions for resume modifications.
    """
    prompt = (
        f"Analyze the following resume JSON and identify sections that match the key job requirements. "
        f"Suggest modifications to better align the resume with the job description.\n\n"
        f"Resume JSON:\n{resume_json}\n\n"
        f"Key Requirements JSON:\n{requirements}\n\n"
        "Provide your suggestions in JSON format, with modifications highlighted."
    )

    response = openai.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.3
    )

    return response.choices[0].message.content


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


def modify_resume_section(section_json, requirements, model_id="gpt-3.5-turbo"):
    """
    Modifies a specific section of the resume to better align with job requirements.

    Args:
        section_json (dict): The JSON object containing the resume section details.
        requirements (dict): The extracted requirements from the job description.
        model_id (str): The model ID for OpenAI (default is gpt-3.5-turbo).

    Returns:
        dict: Modified resume section.
    """
    prompt = (
        f"Modify the following resume section in JSON format to better align with the job requirements. "
        f"Make it more concise and impactful while highlighting relevant skills and experiences:\n\n"
        f"Current Section JSON:\n{section_json}\n\n"
        f"Job Requirements JSON:\n{requirements}\n\n"
        "Return the modified section in JSON format."
    )

    response = openai.chat.completions.create(
        model=model_id, messages=[{"role": "user", "content": prompt}], temperature=0.7
    )

    return json.loads(response["choices"][0]["message"]["content"])


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

    # Step 1: Load or create job descriptions JSON file
    job_descriptions, is_existing = load_or_create_json(job_descriptions_json_file)

    # Check if the current job description already exists by unique ID (URL)
    if job_description_url in job_descriptions:
        logging.info(
            f"Job description for URL '{job_description_url}' already exists. Skipping fetching step."
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
