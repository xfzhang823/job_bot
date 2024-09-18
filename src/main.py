# main.py

import os
import logging
from pipelines.preprocessing_pipeline import run_pipeline as run_preprocessing_pipeline
from pipelines.resume_eval_pipeline import (
    run_pipeline as run_resume_comparison_pipeline,
)
from pipelines.resume_editing_pipeline import (
    run_pipeline as run_resume_editting_pipeline,
)
from config import (
    resume_json_file,
    job_descriptions_json_file,
    job_requirements_json_file,
    description_text_holder,
    responsibilities_flat_json_file,
    requirements_flat_json_file,
    resp_req_sim_metrics_file,
    excluded_from_modification_file,
)


def run_pipeline_1():
    """Pipeline for preprocessing job posting webpage"""
    # Run pipeline 1: preprocessing
    # Define input data sources
    job_description_url = "https://www.google.com/about/careers/applications/jobs/results/113657145978692294-ai-market-intelligence-principal/?src=Online/LinkedIn/linkedin_us&utm_source=linkedin&utm_medium=jobposting&utm_campaign=contract&utm_medium=jobboard&utm_source=linkedin"
    # Define paths using os.path.join for cross-platform compatibility

    # Run the pipeline
    run_preprocessing_pipeline(
        job_description_url=job_description_url,
        job_descriptions_json_file=job_descriptions_json_file,
        requirements_json_file=job_requirements_json_file,
        resume_json_file=resume_json_file,
        text_file_holder=description_text_holder,
        responsibilities_flat_json_file=responsibilities_flat_json_file,
        requirements_flat_json_file=requirements_flat_json_file,
    )


def run_pipeline_2():
    """Pipeline for resume evaluation"""
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


def run_pipeline_3():
    """ "Pipeline for editing responsibility text from resume"""
    sim_metrics_file = resp_req_sim_metrics_file
    import os

    dir_path = r"C:\github\job_bot\input_output\evaluation_optimization\output"
    print(os.listdir(dir_path))
    print(sim_metrics_file)
    run_resume_editting_pipeline(sim_metrics_file, excluded_from_modification_file)


def run_pipeline_4():
    pass


def main():
    """main to run the pipelines"""
    run_pipeline_1()
    run_pipeline_2()
    run_pipeline_3()

    # Run pipeline 2: Compare resume pipeline


if __name__ == "__main__":
    run_pipeline_1()
