""" Temp for testing """

import os
from pipelines.resume_job_comparison_pipeline import run_pipeline


def main():
    """just for texting b/c I have to test from src dir"""

    resume_json_path = (
        r"C:\github\job_bot\data\Resume_Xiaofei_Zhang_2024_template_for_LLM.json"
    )

    reqs_json_path = r"C:\github\job_bot\data\extracted_job_requirements.json"

    # CSV output file
    dir_path = r"C:\github\job_bot\data"
    csv_output_f_path = os.path.join(dir_path, "output_seg_by_seg_sim_matrix_v2.csv")

    run_pipeline(
        requirements_json_file=reqs_json_path,
        resume_json_file=resume_json_path,
        csv_file=csv_output_f_path,
    )

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
