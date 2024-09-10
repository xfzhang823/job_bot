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

    # Step 4. Load resume JSON file
    resume_json = read_from_json_file(resume_json_file)
    pretty_print_json(resume_json)

    # Step 5:
    # # Step 4: Analyze resume against job requirements
    # analysis = analyze_resume_against_requirements(resume_json, requirements)
    # print("Analysis and Suggestions:", analysis)

    # # Step 6: Modify relevant resume sections based on analysis
    # for section in resume_json["experience"]:  # Example: Modify experience sections
    #     modified_section = modify_resume_section(section, requirements)
    #     section.update(modified_section)  # Update the section in place

    # # Step 6: Save the modified resume to a file
    # add_to_json_file(resume_json, "modified_resume.json")
    # print("Resume modification completed and saved to 'modified_resume.json'.")
    # steps = [
    #     (
    #         "Gather Requirements",
    #         lambda: extract_job_requirements_with_gpt(job_description),
    #     ),
    #     ("Read and Compare Resume", lambda: read_resume(resume_file)),
    #     (
    #         "Match Resume to Job Description",
    #         lambda resume_data, job_data: match_resume_to_job(
    #             resume_data, job_description_json
    #         ),
    #     ),
    # ]


# # file paths in plain text
# txt_f_path = "/Resume_Xiaofei_Zhang_2024_template_for_LLM.txt"
# job_descrip_file_path = r"C:\Users\xzhan\My Drive\Job Search\Job Postings\Adobe Design-Principal AI Strategist.txt"

# # Define the file path for the output
# output_file_path = r"C:\Users\xzhan\My Drive\Job Search\Resumes\Resume Xiao-Fei Zhang 2023 Adobe-AI Strategist.txt"

# # read files
# # resume = extract_content(resume_file_path)
# job_description = extract_content(job_descrip_file_path)


# with open(output_file_path, "w") as file:
#     # Step 1: Extract Key Requirements from Job Description
#     job_descrip_prompt = f"""I will give you a job description. As a career coach, identify the key skills and experiences required for this job.
#     Job Description:
#     {job_description}"""
#     job_descrip_response = make_query([{"role": "user", "content": job_descrip_prompt}])
#     file.write("Job Description Analysis:\n" + job_descrip_response + "\n\n")

#     # Step 2: Match Resume to Job Requirements
#     resume_matching_prompt = f"""Based on the following key requirements identified: {job_descrip_response}, how well does this resume match?
#     {resume}"""
#     resume_matching_response = make_query(
#         [{"role": "user", "content": resume_matching_prompt}]
#     )
#     file.write("Resume Matching Analysis:\n" + resume_matching_response + "\n\n")

#     #     # Step 3: Tailor Resume to Job Requirements
#     #     # resume_tailoring_prompt = f"""Based on the matching: {resume_matching_response},
#     #     # please tailor my resume to show capabilities, impact, and metrics,
#     #     # as well as optimizing it for the hiring company's Applicant Tracking System (ATS).
#     #     # Please exclude education and personal contact info."""
#     #     resume_tailor_prompt = f"Based on the above analysis, can you suggest specific changes to the resume to better align it with the job requirements?"

#     #     resume_tailoring_response = make_query(
#     #         [{"role": "user", "content": resume_tailor_prompt}]
#     #     )
#     #     file.write("Edited Resume:\n" + resume_tailoring_response + "\n\n")

#     # # ... [Previous steps and code] ...

#     # Step 3: Request Specific Tailoring Suggestions
#     # Include a summary of key findings from job description analysis and resume matching
#     resume_tailor_prompt = f"""Given the key skills and experiences required for the job as identified:
#     {job_descrip_response}

#     And the analysis of how the current resume matches these:
#     {resume_matching_response}

#     Can you suggest specific changes to the resume to better align it with the job requirements, focusing on showcasing capabilities, impact, and metrics? Please exclude education and personal contact info."""

#     tailor_response = make_query([{"role": "user", "content": resume_tailor_prompt}])
#     file.write("Tailoring Suggestions:\n" + tailor_response + "\n\n")

# # print(f"Responses written to {output_file_path}")
