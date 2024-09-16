""" Temp for testing """

import os
import logging
import json
from pipelines.resume_eval_pipeline import run_pipeline
from evaluation_optimization.resume_editing import (
    edit_text_for_dp,
    edit_text_for_semantic_entailment,
)
from evaluation_optimization.text_similarity_finder import AsymmetricTextSimilarity
from evaluation_optimization.similarity_metrics_eval import categorize_scores
from utils.llm_data_utils import get_openai_api_key
import logging_config
import openai
from openai import OpenAI

# Set up logging
logger = logging.getLogger(__name__)


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


def main():
    """just for testing b/c I have to test from src dir"""

    resp_text = "Provided strategic insights to a major global IT vendor, optimizing their service partner ecosystem in Asia Pacific for improved local implementation outcomes."

    reqs_text1 = "Ability to form and refine hypotheses, gather supporting data, and make recommendations"
    reqs_text2 = "Excellent problem solving and analysis skills, including opportunity identification, market segmentation, and framing of complex/ambiguous problems"
    reqs_text3 = "Leverage first party and third party market data to build assets and programs that surface valuable insights to our business stakeholders and help inform product roadmaps"

    # Instantiate API object (do this outside the function to reduce overhead)
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)  # Instantiate openai api chat completion class
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


if __name__ == "__main__":
    main()
