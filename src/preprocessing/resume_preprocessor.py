""" TBA """

import re
import logging
from utils.generic_utils import pretty_print_json
from utils.dict_utils import fetch_subtrees
from utils.generic_utils import read_from_json_file
from utils.dict_utils import fetch_subtrees, flatten_dict_and_list
import logging


class ResumeParser:
    def __init__(self, json_path):
        """
        Initialize the ResumeParser with a JSON path.

        Args:
            json_path (str): Path to the JSON file containing the resume.
        """
        self.json_path = json_path
        self.resume_dict = self.load_resume_json()

    def load_resume_json(self):
        """
        Load resume data from a JSON file.

        Returns:
            dict: The resume data loaded from the JSON file.
        """
        self.resume_dict = read_from_json_file(self.json_path, key=None)
        logging.info("Resume JSON format loaded.")
        return self.resume_dict

    def extract_responsibilities(self, key="responsibilities"):
        """
        Extract job responsibilities from the resume.

        Args:
            key (str): The key to search for in the resume data. Default is 'responsibilities'.

        Returns:
            dict: The extracted job responsibilities.
        """
        search_by = key
        resume_resps = fetch_subtrees(self.resume_dict, search_key=search_by)
        logging.info(f"Fetched all sections under {search_by}:")
        return resume_resps

    def extract_and_flatten_responsibilities(self):
        """
        Parse the resume JSON file to extract responsibilities.

        Returns:
            dict: Flattened responsibilities dictionary.
        """

        # Extract responsibilities
        resps_dict = self.extract_responsibilities()

        # Flatten the extracted responsibilities
        resps_flat = flatten_dict_and_list(resps_dict)
        logging.info("Extracted & fattened responsibilities.")
        return resps_flat

    def extract_profile(self, key="professional_profile"):
        """
        Extract the professional profile from the resume.

        Args:
            key (str): The key to search for in the resume data. Default is 'professional_profile'.

        Returns:
            dict: The extracted professional profile.
        """
        resume_profile = fetch_subtrees(self.resume_dict, key=key)
        logging.info("Fetched professional profile.")
        return resume_profile

    def extract_keywords(self):
        """
        Extract keywords from the resume.

        Returns:
            None: This method needs to be implemented.
        """
        # Placeholder for the actual keyword extraction logic
        return

    def fetch_core_competencies(self):
        """
        Fetch core competencies from the resume.

        Returns:
            dict: The extracted core competencies.
        """
        core_comp_dict = fetch_subtrees(
            self.resume_dict, search_key="core_compentencies"
        )
        logging.info("Fetched core competencies.")
        return core_comp_dict

    def fetch_titles(self):
        """
        Fetch job titles from the resume.

        Returns:
            None: This method needs to be implemented.
        """
        # Placeholder for the actual title fetching logic
        return
