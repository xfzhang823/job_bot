"""metrics_evaluator.py"""

# Dependencies
import pandas as pd
import logging
import logging_config
from evaluation_optimization.text_similarity_finder import (
    compute_bertscore_precision,
    AsymmetricTextSimilarity,
)
from utils.llm_data_utils import get_openai_api_key
from utils.field_mapping_utils import rename_df_columns, COLUMN_NAMES_TO_VARS_MAPPING


# Set logger
logger = logging.getLogger(__name__)

# Conventional metric criteria suggested by GPT
STANDARD_METRIC_CRITERIA = {
    # "BERTScore Precision"
    "bert_score_precision": {
        "range": (0, 1),
        "high_threshold": 0.85,
        "low_threshold": 0.70,
    },
    # "Soft Similarity (SBERT)"
    "soft_similarity": {
        "range": (-1, 1),
        "high_threshold": 0.7,
        "low_threshold": 0.4,
    },
    # "Word Mover's Distance"
    "word_movers_distance": {
        "range": (0, float("inf")),
        "high_threshold": 5,  # High score is considered "Low" for this metric
        "low_threshold": 15,  # Low score is considered "High" for this metric
        "reverse": True,  # Indicates smaller scores are better
    },
    # "NLI Entailment Score"
    "nli_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.7,
        "low_threshold": 0.3,
    },
    # "Jaccard Similarity"
    "jaccard_similarity": {
        "range": (0, 1),
        "high_threshold": 0.5,
        "low_threshold": 0.2,
    },
    # "DeBERTa Entailment Score"
    "deberta_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.8,
        "low_threshold": 0.4,
    },
}

# Define a more custom metric criteria in a dictionary
METRIC_CRITERIA = {
    # "BERTScore Precision"
    "bert_score_precision": {
        "range": (0, 1),
        "high_threshold": 0.85,
        "low_threshold": 0.75,
    },
    # "Soft Similarity (SBERT)"
    "soft_similarity": {
        "range": (-1, 1),
        "high_threshold": 0.5,
        "low_threshold": 0.2,
    },
    # "Word Mover's Distance"
    "word_movers_distance": {
        "range": (0, float("inf")),
        "high_threshold": 4,  # High score is considered "Low" for this metric
        "low_threshold": 7,  # Low score is considered "High" for this metric
        "reverse": True,  # Indicates smaller scores are better
    },
    # "DeBERTa Entailment Score"
    "deberta_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.70,
        "low_threshold": 0.20,
    },
    # # "Jaccard Similarity"
    # "jaccard_similarity": {
    #     "range": (0, 1),
    #     "high_threshold": 0.25,
    #     "low_threshold": 0.05,
    # },
}


def evaluate_score(metric_name, score):
    """
    Evaluate the score for a given metric name.

    Args:
        metric_name (str): The name of the metric.
        score (float): The score to evaluate.

    Returns:
        str: The category of the score ("High", "Medium", "Low").
    """
    if metric_name not in METRIC_CRITERIA:
        raise ValueError(f"Metric '{metric_name}' is not defined in the criteria.")

    criteria = METRIC_CRITERIA[metric_name]
    high_threshold = criteria["high_threshold"]
    low_threshold = criteria["low_threshold"]
    reverse = criteria.get("reverse", False)

    # For metrics where a lower score is better (like Word Mover's Distance)
    if reverse:
        if score <= high_threshold:
            return "High"
        elif score >= low_threshold:
            return "Low"
    else:
        if score >= high_threshold:
            return "High"
        elif score <= low_threshold:
            return "Low"

    return "Medium"


def categorize_scores(scores):
    """
    Categorize a dictionary of scores based on their metric names.

    Args:
        scores (dict): A dictionary where keys are metric names and values are scores.

    Returns:
        dict: A dictionary with the same keys but with "High", "Medium", or "Low" as values.
    """
    categorized_scores = {}
    for metric_name, score in scores.items():
        try:
            categorized_scores[metric_name] = evaluate_score(metric_name, score)
        except ValueError as e:
            print(e)
            categorized_scores[metric_name] = "Unknown"
    return categorized_scores


def categorize_scores_for_row(row, metrics):
    """
    Categorize scores for a single row based on provided metrics.

    Args:
        row (pd.Series): A single row from the DataFrame.
        metrics (list): A list of metric names to categorize.

    Returns:
        pd.Series: The row with new category columns added.
    """
    # Loop over each metric and apply the evaluate_score function
    for metric in metrics:
        # Create a new column name for the category
        category_col = f"{metric}_cat"
        # Apply evaluate_score to get 'High', 'Medium', 'Low' for each metric
        row[category_col] = evaluate_score(metric, row[metric])

    return row


def categorize_scores_for_df(df, metrics=None):
    """
    Apply category assignments to the entire DataFrame based on provided metrics.

    Args:
        - df (pd.DataFrame): The DataFrame to process.
        - metrics (list): A list of metric names to categorize.
        default to None (use default column names)

    Returns:
        pd.DataFrame: The DataFrame with new category columns added.
    """
    if metrics is None:
        metrics = [
            "bert_score_precision",
            "soft_similarity",
            "word_movers_distance",
            "deberta_entailment_score",
        ]
    df = rename_df_columns(
        df, COLUMN_NAMES_TO_VARS_MAPPING
    )  # Uses default COLUMN_NAMES_TO_VARS_MAPPING from utils

    # Apply high, mid, low categories to similarity metrics
    df = df.apply(lambda row: categorize_scores_for_row(row, metrics), axis=1)
    return df


def calculate_text_similarity_metrics(
    optimized_text: str, requirement_text: str
) -> dict:
    """
    Calculate all similarity scores between a single responsibility (optimized_text)
    and a single requirement (requirement_text).

    Args:
        optimized_text (str): The optimized responsibility text.
        requirement_text (str): The job requirement text.

    Returns:
        dict: A dictionary containing all calculated similarity scores.
    """
    # Instantiate the similarity calculator
    text_similarity = AsymmetricTextSimilarity()

    # Calculate the similarity metrics
    similarity_scores = text_similarity.short_text_similarity_metrics(
        optimized_text, requirement_text
    )

    return {
        "bert_score_precision": similarity_scores["bert_score_precision"],
        "soft_similarity": similarity_scores["soft_similarity"],
        "word_movers_distance": similarity_scores["word_movers_distance"],
        "deberta_entailment_score": similarity_scores["deberta_entailment_score"],
    }


def calculate_one_to_many_similarity_metrices(resps_flat, job_reqs_str):
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
                "responsibility_key": key,
                "responsibility": value,
                "requirements": job_reqs_str,
                "similarity_metrices": similarity_dict,
            }
        )

    # Convert list to a df
    df_similarity = pd.DataFrame(simiarity_results)
    return df_similarity


def calculate_many_to_many_similarity_metrices(
    responsibilities: dict[str, str], requirements: dict[str, str]
):
    """
    Calculate text similarity metrics:
    - between EACH resume responsibility and EACH job requirement;
    - leveraging the AsymmetricTextSimilarity class due to the asymmetrical relationship
    between experience/responsibilities and job requirements.

    Args:
        responsibilities (dict of str): Dictionary of flattened responsibilities from the resume.
        requirements (dict of str): Dictionary of flattened job requirements from the job posting.

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics.
    """
    text_similarity = AsymmetricTextSimilarity()
    similarity_results = []

    # Debugging: Check length of responsibilities and requirements
    logger.info(f"Number of responsibilities: {len(responsibilities)}")
    logger.info(f"Number of requirements: {len(requirements)}")

    # Logging
    no_of_comparisons = len(responsibilities) * len(requirements)
    logging.info(f"Expected number of comparisons: {no_of_comparisons}")

    # Enumerate over responsibilities to capture both index and value
    for resp_key, resp in responsibilities.items():
        # Enumerate over requirements to capture both index and value
        for req_key, req in requirements.items():

            # Debugging
            logger.info(f"Comparing responsibility: {resp} \nwith requirement: {req}")

            # Compute BERTScore Precision (assuming compute_bertscore_precision is defined)
            similarity_dict = text_similarity.short_text_similarity_metrics(resp, req)

            # Append results with responsibility index and value, and requirement index and value
            similarity_results.append(
                {
                    "responsibility_key": resp_key,
                    "responsibility": resp,
                    "requirement_key": req_key,
                    "requirement": req,
                    "BERTScore Precision": similarity_dict["bert_score_precision"],
                    "Soft Similarity": similarity_dict["soft_similarity"],
                    "Word Mover's Distance": similarity_dict["word_movers_distance"],
                    "Deberta Entailment Score": similarity_dict[
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
            {"responsibility": value, "BERTScore Precision": bscore_p}
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
            print(f"Comparing responsibility: {resp} \nwith requirement: {req}")

            # Compute BERTScore Precision (assuming compute_bertscore_precision is defined)
            bscore_p = compute_bertscore_precision(resp, req)

            # Append results with responsibility index and value, and requirement index and value
            results.append(
                {
                    "responsibility_key": resp_key,
                    "responsibility": resp,
                    "requirement_key": req_key,
                    "requirement": req,
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


class SimilarityScoreCalculator:
    """
    Calculate text similarity metrics:
    - In many-to-many mode: between EACH resume responsibility and EACH job requirement.
    - In one-to-one mode: between pre-matched resume responsibility and job requirement pairs.
    - Leverages the AsymmetricTextSimilarity class due to the asymmetrical relationship
      between experience/responsibilities and job requirements.

    Args:
        responsibilities (dict of str, optional): Dictionary of flattened responsibilities from the resume.
        requirements (dict of str, optional): Dictionary of flattened job requirements from the job posting.
        combined_list (list of dict, optional): List of pre-matched responsibility and requirement pairs.
        mode (str, optional): Specifies the operation mode.
                              - "many_to_many": Compares each responsibility to all requirements.
                              - "one_to_one": Compares pre-matched pairs in the combined_list.

    Returns:
        pd.DataFrame: DataFrame containing similarity metrics.
    """

    def __init__(self):
        """
        Initialize the comparator class. The similarity engine can be initialized here.
        """
        self.text_similarity = AsymmetricTextSimilarity()

    def calculate_similarity(self, responsibility, requirement):
        """
        Calculate similarity metrics between a responsibility and a requirement.

        Args:
            responsibility (str): The responsibility text.
            requirement (str): The requirement text.

        Returns:
            dict: Dictionary containing similarity metrics.
        """
        # Calculate similarity metrics
        return self.text_similarity.short_text_similarity_metrics(
            responsibility, requirement
        )

    def many_to_many(self, responsibilities: dict, requirements: dict) -> pd.DataFrame:
        """
        Perform many-to-many similarity comparisons between responsibilities and requirements.

        Args:
            responsibilities (dict): Dictionary of responsibilities.
            requirements (dict): Dictionary of requirements.

        Returns:
            pd.DataFrame: DataFrame containing similarity metrics for all pairs.
        """
        similarity_results = []
        logger.info(
            f"Starting many-to-many comparisons: {len(responsibilities)} responsibilities, {len(requirements)} requirements."
        )

        # Perform many-to-many comparison
        for resp_key, resp in responsibilities.items():
            for req_key, req in requirements.items():
                print(f"Comparing responsibility: {resp} \nwith requirement: {req}")

                # Calculate similarity metrics
                similarity_dict = self.calculate_similarity(resp, req)

                # Append results
                similarity_results.append(
                    {
                        "responsibility_key": resp_key,
                        "responsibility": resp,
                        "requirement_key": req_key,
                        "requirement": req,
                        "BERTScore Precision": similarity_dict["bert_score_precision"],
                        "Soft Similarity": similarity_dict["soft_similarity"],
                        "Word Mover's Distance": similarity_dict[
                            "word_movers_distance"
                        ],
                        "Deberta Entailment Score": similarity_dict[
                            "deberta_entailment_score"
                        ],
                    }
                )

        # Convert results to DataFrame
        return self._results_to_dataframe(similarity_results)

    def one_to_one(self, combined_list: list[dict]) -> pd.DataFrame:
        """
        Perform one-to-one similarity comparisons on pre-matched responsibility and requirement pairs.

        Args:
            combined_list (list of dict): List of dictionaries with pre-matched responsibility and requirement pairs.

        Returns:
            pd.DataFrame: DataFrame containing similarity metrics for each pair.
        """
        similarity_results = []
        logging.info(
            f"Starting one-to-one comparisons: {len(combined_list)} responsibility-requirement pairs."
        )

        # Perform one-to-one comparison
        for entry in combined_list:
            resp_key = entry["responsibility_key"]
            resp = entry["responsibility_text"]
            req_key = entry["requirement_key"]
            req = entry["requirement_text"]

            print(f"Comparing responsibility: {resp} \nwith requirement: {req}")

            # Calculate similarity metrics
            similarity_dict = self.calculate_similarity(resp, req)

            # Append results
            similarity_results.append(
                {
                    "responsibility_key": resp_key,
                    "responsibility": resp,
                    "requirement_Key": req_key,
                    "requirement": req,
                    "BERTScore Precision": similarity_dict["bert_score_precision"],
                    "Soft Similarity": similarity_dict["soft_similarity"],
                    "Word Mover's Distance": similarity_dict["word_movers_distance"],
                    "Deberta Entailment Score": similarity_dict[
                        "deberta_entailment_score"
                    ],
                }
            )

        # Convert results to DataFrame
        return self._results_to_dataframe(similarity_results)

    def _results_to_dataframe(self, results: list[dict]) -> pd.DataFrame:
        """
        Convert the similarity results to a Pandas DataFrame.

        Args:
            results (list of dict): List of dictionaries containing similarity metrics.

        Returns:
            pd.DataFrame: DataFrame with the cleaned similarity metrics.
        """
        df_results = pd.DataFrame(results)

        # Clean dataframe
        df_results = df_results.applymap(
            lambda x: str(x).replace("\n", " ") if isinstance(x, str) else x
        )

        print("Complete DataFrame:")
        print(df_results)

        return df_results


if __name__ == "__main__":
    # Example usage
    example_scores = {
        "bert_score_precision": 0.88,
        "soft_similarity": 0.45,
        "word_movers_distance": 4.2,
        "nli_entailment_score": 0.2,
        "jaccard_similarity": 0.6,
        "deberta_entailment_score": 0.75,
    }

    categorized = categorize_scores(example_scores)
    print("Categorized Scores:", categorized)