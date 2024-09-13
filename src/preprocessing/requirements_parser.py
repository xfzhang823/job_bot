""" TBA """

import logging
from utils.utils import read_from_json_file
from preprocessing.json_parser import fetch_subtrees, flatten_dict_and_list


class JobRequirementsParser:
    def __init__(self, json_path):
        """
        Initialize the JobRequirementsParser with a JSON path.

        Args:
            json_path (str): Path to the JSON file containing the job requirements.
        """
        self.json_path = json_path
        self.job_reqs_dict = self.load_job_requirements_json()

    def load_job_requirements_json(self):
        """
        Load job requirements data from a JSON file and check for validity.

        Returns:
            dict or None: The job requirements data loaded from the JSON file,
                          or None if the data is invalid.
        """
        reqs_dict = read_from_json_file(self.json_path, key=None)

        # Check if reqs_dict is a valid JSON object (dict or list)
        if not isinstance(reqs_dict, (dict, list)):
            print("The provided data is not a valid JSON object (dict or list).")
            logging.error("Invalid JSON format provided. Exiting parsing.")
            return None

        logging.info("Job requirements JSON loaded from the file.")
        return reqs_dict

    def extract_down_to_earth(self):
        """
        Extract 'down to earth' requirements from the job requirements.

        Returns:
            dict: The extracted 'down to earth' requirements.
        """
        if self.job_reqs_dict:
            down_to_earth = fetch_subtrees(self.job_reqs_dict, "down_to_earth")
            logging.info("Fetched 'down to earth' job requirements.")
            return down_to_earth
        else:
            logging.error("Invalid job requirements data.")
            return None

    def extract_bare_minimum(self):
        """
        Extract 'bare minimum' requirements from the job requirements.

        Returns:
            dict: The extracted 'bare minimum' requirements.
        """
        if self.job_reqs_dict:
            bare_minimum = fetch_subtrees(self.job_reqs_dict, "bare_minimum")
            logging.info("Fetched 'bare minimum' job requirements.")
            return bare_minimum
        else:
            logging.error("Invalid job requirements data.")
            return None

    def extract_pie_in_the_sky(self):
        """
        Extract 'pie in the sky' requirements from the job requirements.

        Returns:
            dict: The extracted 'pie in the sky' requirements.
        """
        if self.job_reqs_dict:
            pie_in_the_sky = fetch_subtrees(self.job_reqs_dict, "pie_in_the_sky")
            logging.info("Fetched 'pie in the sky' job requirements.")
            return pie_in_the_sky
        else:
            logging.error("Invalid job requirements data.")
            return None

    def extract_other(self):
        """
        Extract other categories of job requirements from the job requirements.

        Returns:
            dict: The extracted requirements of other categories.
        """
        if self.job_reqs_dict:
            other_categories = fetch_subtrees(self.job_reqs_dict, "other")
            logging.info("Fetched 'other categories' job requirements.")
            return other_categories
        else:
            logging.error("Invalid job requirements data.")
            return None

    def extract_flatten_reqs(self):
        """
        Extract and flatten more relevant job requirements into a dictionary.
        Each requirement is keyed by a unique identifier.

        Returns:
            dict: A dictionary of flattened job requirements with unique keys.

        Relevant requirement categories include:
        - pie_in_sky
        - down_to_earth
        - other
        """
        # Extract requirements with error handling
        try:
            pie_in_sky_reqs = self.extract_pie_in_the_sky()  # List of dicts
            down_to_earth_reqs = self.extract_down_to_earth()  # List of dicts
            other_reqs = self.extract_other()  # Corrected method call (List of dicts)
        except Exception as e:
            logging.error(f"Error extracting requirements: {e}")
            return []

        # Debugging: Check the output of each extraction function
        print(f"Extracted 'pie_in_the_sky' requirements: {pie_in_sky_reqs}")
        print(f"Number of 'pie_in_the_sky' requirements: {len(pie_in_sky_reqs)}")

        print(f"Extracted 'down_to_earth' requirements: {down_to_earth_reqs}")
        print(f"Number of 'down_to_earth' requirements: {len(down_to_earth_reqs)}")

        print(f"Extracted 'other' requirements: {other_reqs}")
        print(f"Number of 'other' requirements: {len(other_reqs)}")

        # Check if any of the extracted lists are None or empty
        if not pie_in_sky_reqs:
            logging.warning("No 'pie_in_sky' requirements found.")
            pie_in_sky_reqs = []
        if not down_to_earth_reqs:
            logging.warning("No 'down_to_earth' requirements found.")
            down_to_earth_reqs = []
        if not other_reqs:
            logging.warning("No 'other' requirements found.")
            other_reqs = []

        # Combine all requirements into a single list
        combined_reqs = pie_in_sky_reqs + down_to_earth_reqs + other_reqs

        # Use the existing flatten_dict_and_list function to flatten the combined list
        flattened_reqs_dict = flatten_dict_and_list(combined_reqs)

        logging.info("Extracted and flattened job requirements.")
        print(
            f"Total number of flattened job requirements: {len(flattened_reqs_dict)}"
        )  # Debugging: Check total count
        print(f"Flattened job requirements: {flattened_reqs_dict}")
        return flattened_reqs_dict

    def extract_flatten_concat_reqs(self):
        """
        Extract, flatten, and concatenate more relevant job requirements into a single string.
        (only in a single text string format can it be properly text processed.)

        Returns:
            str: Concatenated string of all job requirements.

        Relevant requirement categories include:
        - pie_in_sky
        - down_to_earth
        - other
        """

        # Use the first function to get the flattened dictionary of requirements
        merged_flat_dict = self.extract_flatten_reqs()

        # Check if the dictionary is empty to avoid concatenating an empty string
        if not merged_flat_dict:
            logging.warning("No job requirements found to concatenate.")
            return ""

        # Convert dictionary values to a single string with newline separation
        job_reqs_str = "\n".join(merged_flat_dict.values())

        logging.info("Extracted, flattened, and concatenated job requirements.")
        return job_reqs_str

    def extract_all(self):
        """
        Extract all job requirements from the job requirements dictionary, excluding the first-level key.

        Returns:
            dict: A dictionary containing all job requirements, excluding the first-level key.
        """
        if self.job_reqs_dict:
            # Assuming the first level key is the URL or identifier
            # Extract the first (and only) value from the dictionary without the key
            all_requirements = next(iter(self.job_reqs_dict.values()), {})
            logging.info(
                "Fetched all job requirements (excluding the first-level key)."
            )
            return all_requirements
        else:
            logging.error("Invalid job requirements data.")
            return None
