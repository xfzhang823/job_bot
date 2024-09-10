"""TBA"""

import pandas as pd
from matching.text_similarity_finder import TextSimilarity


def calculate_resps_reqs_similarity_metrices(resps_flat, job_reqs_str):
    """
    Calculate text similarity metrics between resume responsibilities and job requirements.

    Args:
        resps_flat (dict): Flattened responsibilities from the resume.
        job_reqs_str (str): Flattened job requirements string.

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics.
    """
    text_similarity = TextSimilarity()
    simiarity_results = []

    # Calcualter simiarity for each responsibility against the job requirments
    for key, value in resps_flat.items():
        print(f"Calculating similarity for: {value}")
        text1 = value
        text2 = job_reqs_str
        similarity_dict = text_similarity.all_similarities(text1, text2, context=None)

        # Append results ot list
        simiarity_results.append(
            {"Responsibility": value, "Similarity Metrices": similarity_dict}
        )

    # Convert list to a df
    df_similarity = pd.DataFrame(simiarity_results)
    return df_similarity
