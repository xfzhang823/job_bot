# main.py

import os
import logging
from pipelines.preprocessing_pipeline import run_pipeline as run_preprocessing_pipeline
from pipelines.resume_job_comparison_pipeline import (
    run_pipeline as run_resume_comparison_pipeline,
)


def run_pipeline_1():
    # Run pipeline 1: preprocessing
    # Define input data sources
    job_description_url = "https://www.google.com/about/careers/applications/jobs/results/113657145978692294-ai-market-intelligence-principal/?src=Online/LinkedIn/linkedin_us&utm_source=linkedin&utm_medium=jobposting&utm_campaign=contract&utm_medium=jobboard&utm_source=linkedin"
    # Define paths using os.path.join for cross-platform compatibility
    description_text_holder = os.path.join("..", "data", "jobposting_text_holder.txt")
    job_descriptions_json_path = os.path.join("..", "data", "jobpostings.json")
    resume_json_path = os.path.join(
        "..", "data", "Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
    )
    requirments_json_path = os.path.join(
        "..", "data", "extracted_job_requirements.json"
    )

    # Run the pipeline
    run_preprocessing_pipeline(
        job_description_url=job_description_url,
        job_descriptions_json_file=job_descriptions_json_path,
        requirements_json_file=requirments_json_path,
        resume_json_file=resume_json_path,
        text_file_holder=description_text_holder,
    )


def run_pipeline_2():
    """just for texting b/c I have to test from src dir"""
    logging.info("Running pipeline 2: Matching Resume")

    resume_json_path = (
        r"C:\github\job_bot\data\Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
    )

    reqs_json_path = r"C:\github\job_bot\data\extracted_job_requirements.json"

    # CSV output file
    dir_path = r"C:\github\job_bot\input_output\resume_comparison"
    csv_output_f_path = os.path.join(dir_path, "output_seg_by_seg_sim_matrix_v2.csv")

    run_resume_comparison_pipeline(
        requirements_json_file=reqs_json_path,
        resume_json_file=resume_json_path,
        csv_file=csv_output_f_path,
    )


def main():
    """main to run the pipelines"""
    run_pipeline_1()
    run_pipeline_2()

    # Run pipeline 2: Compare resume pipeline


if __name__ == "__main__":
    run_pipeline_2()
