# main.py

import os
from pipeline import run_pipeline


def main():
    # Define input data sources
    job_description_url = "https://www.google.com/about/careers/applications/jobs/results/113657145978692294-ai-market-intelligence-principal/?src=Online/LinkedIn/linkedin_us&utm_source=linkedin&utm_medium=jobposting&utm_campaign=contract&utm_medium=jobboard&utm_source=linkedin"
    # Define paths using os.path.join for cross-platform compatibility
    description_text_holder = os.path.join("data", "jobposting_text_holder.txt")
    job_descriptions_json_path = os.path.join("data", "jobpostings.json")
    resume_json_path = os.path.join(
        "data", "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
    )
    requirments_json_path = os.path.join("data", "extracted_job_requirements.json")

    # Run the pipeline
    run_pipeline(
        job_description_url=job_description_url,
        job_descriptions_json_file=job_descriptions_json_path,
        requirements_json_file=requirments_json_path,
        resume_json_file=resume_json_path,
        text_file_holder=description_text_holder,
    )


if __name__ == "__main__":
    main()
