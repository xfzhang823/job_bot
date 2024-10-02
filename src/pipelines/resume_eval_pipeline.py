import os
from pathlib import Path
import pandas as pd
import re
import logging
import logging_config
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser
from evaluation_optimization.text_similarity_finder import TextSimilarity
from evaluation_optimization.similarity_metric_eval import (
    calculate_many_to_many_similarity_metrices,
    SimilarityScoreCalculator,
)
from evaluation_optimization.similarity_metric_eval import categorize_scores_for_df
from utils.generic_utils import read_from_json_file, pretty_print_json
from utils.get_file_names import get_file_names
from config import METRICS_OUTPUTS_EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR
from IPython.display import display


# Set up logger
logger = logging.getLogger(__name__)


def create_csv_file_name(company, job_title):
    """
    Creates a CSV file name from company and job title.

    Replaces illegal file name characters with underscores.

    Args:
        company (str): Company name
        job_title (str): Job title

    Returns:
        str: CSV file name
    """

    # Define illegal file name characters
    illegal_chars = r"[^a-zA-Z0-9._-]"

    # Replace illegal characters with underscores
    company = re.sub(illegal_chars, "_", company)
    job_title = re.sub(illegal_chars, "_", job_title)

    # Create CSV file name
    file_name = f"{company}_{job_title}.csv"

    return file_name


directory = METRICS_OUTPUTS_EVALUATION_OPTIMIZATION_INPUT_OUTPUT_DIR


def check_if_existing(dir, file_name):
    """
    Checks if a file exists in the specified directory.

    Args:
        dir (str): Directory path
        file_name (str): File name

    Returns:
        bool: True if file exists, False otherwise
    """
    existing_files = get_file_names(
        dir_path=dir, full_path=False, file_type_inclusive=True
    )
    if file_name in existing_files:
        logger.info(f"File ({file_name}) already exists. Skipping this step.")
        return True
    return False


def unpack_and_combine_json(nested_json, requirements_json):
    """
    Unpacks the nested responsibilities JSON and combines it with matching requirement texts
    from the requirements JSON. Outputs a list of dictionaries with responsibility and requirement texts.

    Args:
        nested_json (dict): JSON-like dictionary containing responsibility text structured in a nested format.
        requirements_json (dict): JSON-like dictionary containing requirement texts keyed by requirement IDs.

    Returns:
        list: A list of dictionaries containing responsibility keys, requirement keys,
              responsibility texts, and matched requirement texts.

    Error Handling:
        - If a requirement key is not found in the requirements JSON, it will skip that entry.
        - If a required field (e.g., 'optimized_text') is missing, it will skip that entry.
        - Logs warnings for missing fields and unmatched keys for better traceability.
    """
    results = []

    for resp_key, values in nested_json.items():  # Unpack the 1st level
        if not isinstance(values, dict):
            logger.info(
                f"Warning: Unexpected data structure under '{resp_key}'. Skipping entry."
            )
            continue

        for req_key, sub_value in values.items():  # Unpack the 2nd level
            if not isinstance(sub_value, dict):
                logger.info(
                    f"Warning: Unexpected data structure under '{req_key}'. Skipping entry."
                )
                continue

            # Extract the optimized_text
            if "optimized_text" in sub_value:
                optimized_text = sub_value["optimized_text"]
            else:
                logger.info(
                    f"Warning: Missing 'optimized_text' for '{req_key}'. Skipping entry."
                )
                continue

            # Perform the lookup in the requirements JSON
            requirement_text = requirements_json.get(req_key)
            if requirement_text is None:
                logger.info(
                    f"Warning: Requirement key '{req_key}' not found in requirements JSON. Skipping entry."
                )
                continue

            # Append results to a list for further processing
            results.append(
                {
                    "responsibility_key": resp_key,
                    "requirement_key": req_key,
                    "responsibility_text": optimized_text,
                    "requirement_text": requirement_text,
                }
            )

    return results


def run_pipeline(requirements_json_file, resume_json_file, csv_file):
    """1st time run pipeline to calculate responsibility vs requirement alignment scores"""

    logger.info("Start running responsibility/requirement alignment scoring pipeline.")

    # Check if resps comparison csv file (the results) exists already
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

        # Not needed for now
        # reqs_flat_str = (
        #     job_reqs_parser.extract_flatten_concat_reqs()
        # )  # concat into a single str.

        # Step 3. Calculate and display similarity metrices - Segment by Segment
        similarity_df = calculate_many_to_many_similarity_metrices(
            resps_flat, reqs_flat
        )
        logging.info("Similarity metrics calcuated.")

        # Step 4. Add score category values (high, mid, low)
        # Translate DataFrame columns to match expected column names**
        similarity_df = categorize_scores_for_df(similarity_df)
        logger.info("Similarity metrics score categories created.")

        # Step 5. Clean and save to csv
        # df_cleaned = bscore_p_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())
        df.to_csv(csv_file, index=False)
        logging.info(f"Similarity metrics saved to location {csv_file}.")

        # Load the dataframe
        df = pd.read_csv(csv_file)

        logger.info(f"Similarity metrics saved to file ({csv_file})")
    logger.info(
        "Finished running responsibility/requirement alignment scoring pipeline."
    )
    # Display the top rows of the dataframe for verification
    print("Similarity Metrics Dataframe:")
    display(df.head(10))


def re_run_pipeline(requirements_file, responsibilities_file, csv_file):
    """
    Re-run pipeline to calculate revised responsibility vs (job) requirement alignment scores
    """

    # Check if resps comparison csv file (the results) ALREADY exists
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)

        print(f"Similarity Metrics Dataframe: {df.head(10)}")

        logger.info("Output already exists, skipping pipeline.")
        return  # Early exit if output exists

    logger.info("Start running responsibility/requirement alignment scoring pipeline.")

    # Step 1: Read JSON files
    resps_dict = read_from_json_file(responsibilities_file)  # Nested dictionary
    reqs_dict = read_from_json_file(requirements_file)

    # Step 2: parse and combine 2 into a single list/dict
    combined_list = unpack_and_combine_json(
        nested_json=resps_dict, requirements_json=reqs_dict
    )

    # Step 3. Iterate through the Calculate and display similarity metrices - Segment by Segment
    similarity_calculator = SimilarityScoreCalculator()
    similarity_df = similarity_calculator.one_to_one(combined_list)

    logger.info("Similarity metrics calcuated.")

    # Step 4. Add score category values (high, mid, low)
    # Translate DataFrame columns to match expected column names**
    similarity_df = categorize_scores_for_df(similarity_df)
    logger.info("Similarity metrics score categories created.")

    # Step 5. Clean and save to csv
    # df_cleaned = bscore_p_df.applymap(lambda x: str(x).replace("\n", " ").strip())
    df = similarity_df.applymap(lambda x: str(x).replace("\n", " ").strip())
    df.to_csv(csv_file, index=False)

    logging.info(f"Similarity metrics saved to location {csv_file}.")

    # Print out for debugging
    print(f"Similarity Metrics Dataframe: {df.head(10)}")

    logger.info(f"Similarity scores saved to csv file ({csv_file})")
