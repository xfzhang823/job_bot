import json
import pandas as pd

from preprocessing.json_parser import (
    fetch_branches,
    fetch_subtrees,
    fetch_subtrees_under_subtrees,
    flatten_dict_and_list,
)
from utils.utils import pretty_print_json as pprint_json
from preprocessing.resume_parser import ResumeParser
from preprocessing.job_requirements_parser import JobRequirementsParser
from matching.text_similarity_finder import TextSimilarity
from matching.resume_matching import get_resps_reqs_similarity_metrices
from IPython.display import display


def main():
    resume_json_path = (
        r"C:\github\job_bot\data\Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
    )

    reqs_json_path = r"C:\github\job_bot\data\extracted_job_requirements.json"

    # Parse resume
    resume_parser = ResumeParser(resume_json_path)
    resps_flat = resume_parser.extract_and_flatten_responsibilities()

    # Parse job requirements, flatten, and concactenate into a single string
    job_reqs_parser = JobRequirementsParser(reqs_json_path)
    job_reqs_str = job_reqs_parser.extract_flatten_concat_reqs()

    # Calcualte and display similarity metrices
    similarity_df = get_resps_reqs_similarity_metrices(resps_flat, job_reqs_str)

    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.max_rows", None)

    print("Similarity Metrics Dataframe:")
    display(similarity_df)
    similarity_df.to_clipboard()

    # # Print merged result
    # print(json.dumps(merged_data, indent=4))
    # work_experiences = fetch_branches(nested_dict, search_key=search_by)
    # print(f"Branches with key {search_by}:")
    # pprint_json(work_experiences)

    # print()

    # summary_of_res = fetch_subtrees_under_subtrees(
    #     nested_dict, "work_experience", "summary"
    # )
    # print(f"summaries under work experience")
    # pprint_json(summary_of_res)


if __name__ == "__main__":
    main()
