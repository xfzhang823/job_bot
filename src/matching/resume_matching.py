"""TBA"""

import pandas as pd
import logging
from matching.text_similarity_finder import (
    compute_bertscore_precision,
    AsymmetricTextSimilarity,
)


def calculate_resp_similarity_metrices(resps_flat, job_reqs_str):
    """
    Calculate text similarity metrics between resume responsibilities and job requirements,
    leveraging the AsymmetricTextSimilarity class due to the asymmetrical relationship btw
    experience/responsibilities and job requirements.

    Args:
        resps_flat (dict): Flattened responsibilities from the resume.
        job_reqs_str (str): Flattened job requirements string.

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics.
    """
    text_similarity = AsymmetricTextSimilarity()
    simiarity_results = []

    # Calcualter simiarity for each responsibility against the job requirments
    for key, value in resps_flat.items():
        print(f"Calculating similarity for: {value}")
        text1 = value
        text2 = job_reqs_str
        similarity_dict = text_similarity.all_metrics(text1, text2, context=None)

        # Append results ot list
        simiarity_results.append(
            {
                "Responsibility Key": key,
                "Responsibility": value,
                "Requirements": job_reqs_str,
                "Similarity Metrices": similarity_dict,
            }
        )

    # Convert list to a df
    df_similarity = pd.DataFrame(simiarity_results)
    return df_similarity


def calculate_segment_resp_similarity_metrices(
    responsibilities: list[str], requirements: list[str]
):
    """
    Calculate text similarity metrics:
    - between EACH resume responsibility and EACH job requirement;
    - leveraging the AsymmetricTextSimilarity class due to the asymmetrical relationship
    btw experience/responsibilities and job requirements.

    Args:
        responsibilities (list of str): List of flattened responsibilities from the resume.
        requirements (list of str): List of flattened job requirements from the job posting.

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics.
    """
    text_similarity = AsymmetricTextSimilarity()
    similarity_results = []

    # Debugging: Check length of responsibilities and requirements
    print(f"Number of responsibilities: {len(responsibilities)}")
    print(f"Number of requirements: {len(requirements)}")

    # Logging
    no_of_comparisons = len(responsibilities) * len(requirements)
    logging.info(f"Expected number of comparisons: {no_of_comparisons}")

    # Enumerate over responsibilities to capture both index and value
    for resp_key, resp in responsibilities.items():
        # Enumerate over requirements to capture both index and value
        for req_key, req in requirements.items():

            # Debugging
            print(f"Comparing Responsibility: {resp} \nwith Requirement: {req}")

            # Compute BERTScore Precision (assuming compute_bertscore_precision is defined)
            similarity_dict = text_similarity.short_text_similarity_metrics(resp, req)

            # Append results with responsibility index and value, and requirement index and value
            similarity_results.append(
                {
                    "Responsibility Key": resp_key,
                    "Responsibility": resp,
                    "Requirement Key": req_key,
                    "Requirement": req,
                    "BERTScore Precision": similarity_dict["bert_score_precision"],
                    "Soft Similarity": similarity_dict["soft_similarity"],
                    "Word Mover's Distance": similarity_dict["word_movers_distance"],
                    "Deberta Entailment Scoree": similarity_dict[
                        "deberta_entailment_score"
                    ],
                }
            )

            # Debugging: Check similarity output
            print(f"Similarity Metrics: {similarity_dict}")

    # Convert results to DataFrame for easy analysis
    df_results = pd.DataFrame(similarity_results)

    # Clean dataframe
    df_results = df_results.applymap(
        lambda x: str(x).replace("\n", " ") if isinstance(x, str) else x
    )

    # Display the full DataFrame content for debugging purposes
    print("Complete DataFrame:")
    print(df_results)

    return df_results


def calculate_resps_reqs_bscore_precisions(resps_flat, job_reqs_str):
    """
    Calculate the BERTScore (precision only) between:
    - Each resume responsibility.
    - Job requirements as a whole.

    Args:
        - resps_flat (dict): Dictionary of flattened responsibilities from the resume,
        where the key is an identifier and the value is the responsibility text.
        - job_reqs_str (str): Flattened job requirements string.

    Returns:
        pd.DataFrame: DataFrame containing BERTScore precision metrics for
        each responsibility against the job requirements.
    """
    bscore_p_results = []  # Corrected typo in variable name

    # Calculate similarity for each responsibility against the job requirements
    for key, value in resps_flat.items():
        print(f"Calculating BERTScore for: {value}")
        candidate_sent = value
        ref_parag = job_reqs_str

        # Debugging
        print(f"Candidate sententce: {candidate_sent}")
        print(f"Reference paragraph: {ref_parag}")

        # Compute BERTScore precision using the provided function
        bscore_p = compute_bertscore_precision(
            candidate_sent, ref_parag, candidate_context=None, reference_context=None
        )

        # Append results to the list
        bscore_p_results.append(
            {"Responsibility": value, "BERTScore Precision": bscore_p}
        )

    # Convert list to a DataFrame
    df_bscore_p = pd.DataFrame(bscore_p_results)
    return df_bscore_p


def calculate_segment_resp_bscore_precisions(
    responsibilities: list[str], requirements: list[str]
):
    """
    Calculate BERTScore Precision for each responsibility against each job requirement.

    Args:
        responsibilities (list of str): List of flattened responsibilities from the resume.
        requirements (list of str): List of flattened job requirements from the job posting.

    Returns:
        pd.DataFrame: DataFrame containing similarity scores for each responsibility and requirement pair.
    """
    results = []

    # Debugging: Check length of responsibilities and requirements
    print(f"Number of responsibilities: {len(responsibilities)}")
    print(f"Number of requirements: {len(requirements)}")

    # Expected number of combinations
    print(
        f"Expected number of comparisons: {len(responsibilities) * len(requirements)}"
    )

    # Enumerate over responsibilities to capture both index and value
    for resp_key, resp in responsibilities.items():
        # Enumerate over requirements to capture both index and value
        for req_key, req in requirements.items():

            # Debugging
            print(f"Comparing Responsibility: {resp} \nwith Requirement: {req}")

            # Compute BERTScore Precision (assuming compute_bertscore_precision is defined)
            bscore_p = compute_bertscore_precision(resp, req)

            # Append results with responsibility index and value, and requirement index and value
            results.append(
                {
                    "Responsibility Key": resp_key,
                    "Responsibility": resp,
                    "Requirement Key": req_key,
                    "Requirement": req,
                    "BERTScore Precision": bscore_p,
                }
            )

            # Debugging: Check BERTScore output
            print(f"BERTScore Precision: {bscore_p}")

    # Convert results to DataFrame for easy analysis
    df_results = pd.DataFrame(results)

    # Display the full DataFrame content for debugging purposes
    print("Complete DataFrame:")
    print(df_results)

    return df_results


