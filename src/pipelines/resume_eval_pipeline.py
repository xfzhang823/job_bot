import os
import pandas as pd
import logging
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from evaluation_optimization.text_similarity_finder import TextSimilarity
from evaluation_optimization.resume_matching import (
    calculate_segment_resp_similarity_metrices,
)
from evaluation_optimization.similarity_metrics_eval import categorize_scores_for_df
from IPython.display import display


def run_pipeline(requirements_json_file, resume_json_file, csv_file):
    """run pipeline"""

    # Check if resps comparison csv file exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        # Step 1: Parse and flatten responsibilities from resume (as a dict)
        resume_parser = ResumeParser(resume_json_file)
        resps_flat = (
            resume_parser.extract_and_flatten_responsibilities()
        )  # this is a dict

        # Step 2: Parse and flatten job requirements (as a dict) or
        # parse/flatten/conncactenate into a single string
        job_reqs_parser = JobRequirementsParser(requirements_json_file)
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
        similarity_df = categorize_scores_for_df(similarity_df)

        # Clean and save to csv
        # df_cleaned = bscore_p_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df.to_csv(csv_file, index=False)
        logging.info(f"Similarity metrics saved to location {csv_file}.")

        # Load the dataframe
        df = pd.read_csv(csv_file)

    # Display the top rows of the dataframe for verification
    print("Similarity Metrics Dataframe:")
    display(df.head(30))

    # Step 6. Filter responsibilities to keep only the more relevant bullets

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
