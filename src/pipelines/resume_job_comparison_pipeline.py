import os
import json
import pandas as pd
import logging
from preprocessing.json_parser import (
    fetch_branches,
    fetch_subtrees,
    fetch_subtrees_under_subtrees,
    flatten_dict_and_list,
)
from utils.utils import pretty_print_json as pprint_json
from utils.field_mapping_utils import translate_column_names
from preprocessing.resume_parser import ResumeParser
from preprocessing.requirements_parser import JobRequirementsParser
from matching.text_similarity_finder import TextSimilarity
from matching.resume_matching import (
    calculate_resp_similarity_metrices,
    calculate_segment_resp_similarity_metrices,
    calculate_resps_reqs_bscore_precisions,
    calculate_segment_resp_bscore_precisions,
)
from matching.similarity_metrics_evaluator import categorize_scores_of_row
from IPython.display import display


def add_score_categoies(df):
    df = translate_column_names(
        df
    )  # Uses default COLUMN_NAMES_TO_VARS_MAPPING from utils

    # Apply high, mid, low categories to similarity metrics
    df = df.apply(categorize_scores_of_row, axis=1)
    return df


def run_pipeline(requirements_json_file, resume_json_file, csv_file):
    """just for texting b/c I have to test from src dir"""

    # Set file paths
    resume_json_path = resume_json_file
    reqs_json_path = requirements_json_file
    csv_path = csv_file

    # Check if resps comparison csv file exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Step 1: Parse and flatten responsibilities from resume (as a dict)
        resume_parser = ResumeParser(resume_json_path)
        resps_flat = (
            resume_parser.extract_and_flatten_responsibilities()
        )  # this is a dict

        # Step 2: Parse and flatten job requirements (as a dict) or
        # parse/flatten/conncactenate into a single string
        job_reqs_parser = JobRequirementsParser(reqs_json_path)
        reqs_flat = job_reqs_parser.extract_flatten_reqs()  # this is a dict
        reqs_flat_str = (
            job_reqs_parser.extract_flatten_concat_reqs()
        )  # concat into a single str.

        # # Step 3A: Calcualte and display BERTScore precisions
        # bscore_p_df = calculate_resps_reqs_bscore_precisions(resps_flat, reqs_flat_str)

        # # Step 3B: Calcualte and display BERTScore precisions - SEGMENT by SEGMENT
        # bscore_p_df = calculate_segment_resp_bscore_precisions(resps_flat, reqs_flat)

        # Step 4A. Calculate and display similarity metrices

        # Step 4B. Calculate and display similarity metrices - Segment by Segment
        similarity_df = calculate_segment_resp_similarity_metrices(
            resps_flat, reqs_flat
        )
        logging.info("Similarity metrics calcuated.")

        # Step 5. Add score category values (high, mid, low)
        # Translate DataFrame columns to match expected column names**
        similarity_df = add_score_categoies(similarity_df)

        # Display
        print("Similarity Metrics Dataframe:")
        # display(bscore_p_df)
        display(similarity_df.head(30))

        # Clean and save to csv
        # df_cleaned = bscore_p_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df_cleaned = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df_cleaned.to_csv(csv_path, index=False)
        logging.info(f"Similarity metrics saved to location {csv_path}.")

        # Load the dataframe
        df = pd.read_excel(csv_file)

    # Step 6. Filter out

    # Load csv_file


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
