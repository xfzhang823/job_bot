import os
import logging
import json
import pandas as pd

from evaluation_optimization.resume_editor import TextEditor
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.similarity_metrics_eval import categorize_scores
import logging_config

# Set up logging
logger = logging.getLogger(__name__)


def filter_responsibilities_by_low_scores(df, fields):
    """
    Filters out responsibilities where all specified fields have 'Low' scores for all requirements.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing responsibility and requirement data.
    - fields (list): A list of column names to check for 'Low' values.

    Returns:
    - responsibilities_to_optimize (np.array): An array of unique responsibilities
    that do not have 'Low' scores for all requirements in the specified fields.
    """
    # Group the data by Responsibility and aggregate the counts of "Low" scores for each specified field
    aggregation_dict = {field: lambda x: (x == "Low").sum() for field in fields}
    aggregation_dict["Requirement Key"] = (
        "count"  # Count the number of requirements associated with each responsibility
    )

    grouped = df.groupby("Responsibility").agg(aggregation_dict).reset_index()

    # Filter responsibilities where all specified fields have "Low" scores for all requirements
    filter_condition = grouped[fields[0]] == grouped["Requirement Key"]
    for field in fields[1:]:
        filter_condition &= grouped[field] == grouped["Requirement Key"]

    filtered_responsibilities = grouped[filter_condition]

    # Get responsibilities that don't match the above criteria
    responsibilities_to_optimize = df[
        ~df["Responsibility"].isin(filtered_responsibilities["Responsibility"])
    ]["Responsibility"].unique()

    return responsibilities_to_optimize


# # Example usage:
# # Assuming your dataframe is loaded into 'df'
# fields_to_check = ["soft_similarity_cat", "deberta_entailment_score_cat"]
# responsibilities_to_optimize = filter_responsibilities_by_low_scores(
#     df, fields_to_check
# )
# print(responsibilities_to_optimize)


def get_sim_score_categories(revised_text, reference_text):
    textsimilarity = AsymmetricTextSimilarity()

    # Get sim scores
    candidate_text = revised_text
    sim_scores = textsimilarity.short_text_similarity_metrics(
        candidate=candidate_text, reference=reference_text
    )
    logger.info("Similarity related scores calculated")

    sim_score_cats = categorize_scores(sim_scores)
    logger.info("Similarity related score categories assigned.")

    return sim_scores, sim_score_cats


def run_pipeline(sim_metrices_csv_file):
    """Pipeline to modify resume"""

    # Step 1. Read responsibility vs requirement similarity metrics csv file
    try:
        df = pd.read_csv(sim_metrices_csv_file)
        print("CSV file loaded successfully.")
        print(df.head(5))  # Print the first few rows
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")

    # Step 2. Exclude certain responsibilities from modification
    # (to be added back afterwards-factual statements like "promoted to ... in ...")
    df_to_exclude = df[
        df["Responsibility Key"] == "3.responsibilities.5"
    ]  # This is the "promoted to manger in ..."

    # Save on disk to be added back later
    df_to_exclude.to_csv()

    # df w/t resp. to modify w/t LLM
    df_to_optimize = df[~df_to_exclude]

    # Step 3. Loop over Optimize responsibilities

    responsibilities_to_optimize = filter_responsibilities_by_low_scores(df)
    print(responsibilities_to_optimize)
    # Step 4. Modify responsibilities by matching to requirements
    # Instantiate API object (do this outside the function to reduce overhead)


def other():
    gpt3 = "gpt-3.5-turbo"
    gpt4 = "gpt-4-turbo"

    # Set up dict holders
    all_revised_texts = {"revised_text": []}
    all_revised_scores = {"revised_scores": []}
    all_revised_score_categories = {"revised_score_categories": []}

    all_final_texts = {"final_text": []}
    all_final_scores = {"final_scores": []}
    all_final_score_categories = {"final_score_categories": []}

    requirement_list = [reqs_text1, reqs_text2, reqs_text3]

    print(f"Original text:\n{resp_text}")

    # Start looping
    id = 1
    for requirment in requirement_list:
        print(f"id: {id} \nMatch to {requirment}")
        try:
            revised_text_dict = edit_text_for_semantic_entailment(
                client=client,
                text_id=id,
                candidate_text=resp_text,
                reference_text=requirment,
                model_id=gpt3,
            )  # The function returns both id and the actual revised text

            # Debugging: Check keys in revised_text_dict
            print(f"Keys in revised_text_dict: {revised_text_dict.keys()}")

            if "optimized_text" not in revised_text_dict:
                raise KeyError("Key 'optimized_text' not found in revised_text_dict")

            revised_text = revised_text_dict["optimized_text"]
            logger.info(f"revised_text: {revised_text}")

            print("\n\n")

            final_text_dict = edit_text_for_dp(
                client=client,
                text_id=id,
                target_text=revised_text,
                source_text=resp_text,
                model_id=gpt3,
            )

            # Debugging: Check keys in final_text_dict
            print(f"Keys in final_text_dict: {final_text_dict.keys()}")

            if "optimized_text" not in final_text_dict:
                raise KeyError("Key 'optimized_text' not found in final_text_dict")

            final_text = final_text_dict["optimized_text"]
            logger.info(f"final_text: {final_text}")

            print("\n\n")

            sim_scores, sim_score_categories = get_sim_score_categories(
                revised_text, requirment
            )

            print("\n\n")

            final_sim_scores, final_sim_score_categories = get_sim_score_categories(
                final_text, requirment
            )

            print("Initial revision:")
            print(f"Scores:\n{sim_scores}")
            print(f"Scores_Cat:\n{sim_score_categories}")

            print("Final version")
            print(f"Scores:\n{final_sim_scores}")
            print(f"Scores_Cat:\n{final_sim_score_categories}")

            # Update combined dicts
            all_revised_texts["revised_text"].append(revised_text)
            all_revised_scores["revised_scores"].append(sim_scores)
            all_revised_score_categories["revised_score_categories"].append(
                sim_score_categories
            )

            all_final_texts["revised_text"].append(final_text)
            all_final_scores["revised_scores"].append(final_sim_scores)
            all_final_score_categories["revised_score_categories"].append(
                final_sim_score_categories
            )

        except Exception as e:
            print(f"An error occurred during revision 1: {e}")

        id += 1
        print("\n\n\n")

    # Combine and save to json file
    original_texts = {"original_text": resp_text}
    all_req_texts = {"requirments": [reqs_text1, reqs_text2, reqs_text3]}
    combined_dict_list_1 = [
        original_texts,
        all_req_texts,
        all_revised_texts,
        all_revised_scores,
        all_revised_score_categories,
    ]

    combined_dict_list_2 = [
        original_texts,
        all_req_texts,
        all_final_texts,
        all_final_scores,
        all_final_score_categories,
    ]

    f_path_1 = r"C:\github\job_bot\data\edited_resps_examples_1.json"
    with open(f_path_1, "w") as f:
        json.dump(combined_dict_list_1, f, indent=4)

    f_path_2 = r"C:\github\job_bot\data\edited_resps_examples_2.json"
    with open(f_path_2, "w") as f:
        json.dump(combined_dict_list_2, f, indent=4)
