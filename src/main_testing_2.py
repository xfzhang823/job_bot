""" Temp for testing """

import logging
import os
from pipelines.resume_eval_pipeline import run_pipeline
from utils.generic_utils import load_or_create_json, pretty_print_json


def main():
    """just for testing b/c I have to test from src dir"""

    job_description_url = "https://www.google.com/about/careers/applications/jobs/results/113657145978692294-ai-market-intelligence-principal/?src=Online/LinkedIn/linkedin_us&utm_source=linkedin&utm_medium=jobposting&utm_campaign=contract&utm_medium=jobboard&utm_source=linkedin"

    # Define paths using os.path.join for cross-platform compatibility
    job_descriptions_json_path = os.path.join("..", "data", "jobpostings.json")

    print(job_descriptions_json_path)

    job_descriptions, is_existing = load_or_create_json(
        job_descriptions_json_path, key=job_description_url
    )
    pretty_print_json(job_descriptions)

    if is_existing:
        logging.info(
            f"Job description for URL '{job_description_url}' already exists. Skipping the next step."
        )
        job_description_json = job_descriptions[job_description_url]
    else:
        print("go through the process!")

    pretty_print_json(job_description_json)


if __name__ == "__main__":
    main()
