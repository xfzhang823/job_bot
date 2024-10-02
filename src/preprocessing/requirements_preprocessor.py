""" TBA """

import logging
from utils.generic_utils import read_from_json_file
from utils.dict_utils import fetch_subtrees, flatten_dict_and_list


import logging
from utils.generic_utils import read_from_json_file
from utils.dict_utils import fetch_subtrees, flatten_dict_and_list


class JobRequirementsParser:
    def __init__(self, json_path, url):
        """
        Initialize the JobRequirementsParser with a JSON path and a specific job posting URL.

        Args:
            json_path (str): Path to the JSON file containing job requirements.
            url (str): The specific URL for the job posting to be parsed.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> print(parser.url)
            'https://example.com/job1'
        """
        self.json_path = json_path
        self.url = url
        self.job_reqs_dict = self.load_single_job_posting()

    def load_single_job_posting(self):
        """
        Load the specific job posting data from the JSON file based on the provided URL.

        Returns:
            dict or None: The job posting data loaded from the JSON file,
                          or None if the data is invalid or the URL is not found.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> job_posting = parser.load_single_job_posting()
            >>> print(job_posting['down_to_earth'])
            ["5+ years of experience in software development", "Strong knowledge of Python"]
        """
        all_reqs_dict = read_from_json_file(self.json_path, key=None)

        # Check if all_reqs_dict is a valid JSON object (dict)
        if not isinstance(all_reqs_dict, dict):
            print("The provided data is not a valid JSON object.")
            logging.error("Invalid JSON format provided. Exiting parsing.")
            return None

        # Fetch the job posting for the specific URL
        job_posting = all_reqs_dict.get(self.url, None)
        if job_posting is None:
            print(f"No job posting found for the URL: {self.url}")
            logging.error(f"No job posting found for the URL: {self.url}")
            return None

        logging.info(f"Job requirements for {self.url} loaded from the file.")
        return job_posting

    def extract_down_to_earth(self):
        """
        Extract 'down to earth' requirements from the job posting.

        Returns:
            list or None: A list of 'down to earth' requirements, or None if not found.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> down_to_earth_reqs = parser.extract_down_to_earth()
            >>> print(down_to_earth_reqs)
            ["5+ years of experience in software development", "Strong knowledge of Python"]
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
        Extract 'bare minimum' requirements from the job posting.

        Returns:
            list or None: A list of 'bare minimum' requirements, or None if not found.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> bare_minimum_reqs = parser.extract_bare_minimum()
            >>> print(bare_minimum_reqs)
            ["Bachelor's degree in Computer Science or equivalent"]
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
        Extract 'pie in the sky' requirements from the job posting.

        Returns:
            list or None: A list of 'pie in the sky' requirements, or None if not found.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> pie_in_the_sky_reqs = parser.extract_pie_in_the_sky()
            >>> print(pie_in_the_sky_reqs)
            ["PhD in Computer Science", "10+ years of experience in AI research"]
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
        Extract 'other' categories of job requirements from the job posting.

        Returns:
            list or None: A list of 'other' requirements, or None if not found.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> other_reqs = parser.extract_other()
            >>> print(other_reqs)
            ["English proficiency required", "Experience working with international teams"]
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

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> flattened_reqs = parser.extract_flatten_reqs()
            >>> print(flattened_reqs)
            {'req_1': "5+ years of experience in software development", 'req_2': "Strong knowledge of Python"}
        """
        try:
            pie_in_sky_reqs = self.extract_pie_in_the_sky()
            down_to_earth_reqs = self.extract_down_to_earth()
            other_reqs = self.extract_other()
        except Exception as e:
            logging.error(f"Error extracting requirements: {e}")
            return []

        pie_in_sky_reqs = pie_in_sky_reqs or []
        down_to_earth_reqs = down_to_earth_reqs or []
        other_reqs = other_reqs or []

        combined_reqs = pie_in_sky_reqs + down_to_earth_reqs + other_reqs

        flattened_reqs_dict = flatten_dict_and_list(combined_reqs)

        logging.info("Extracted and flattened job requirements.")
        return flattened_reqs_dict

    def extract_flatten_concat_reqs(self):
        """
        Extract, flatten, and concatenate all relevant job requirements into a single string.

        Returns:
            str: Concatenated string of all job requirements.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> concatenated_reqs = parser.extract_flatten_concat_reqs()
            >>> print(concatenated_reqs)
            "5+ years of experience in software development\nStrong knowledge of Python\nEnglish proficiency required"
        """
        merged_flat_dict = self.extract_flatten_reqs()

        if not merged_flat_dict:
            logging.warning("No job requirements found to concatenate.")
            return ""

        job_reqs_str = "\n".join(merged_flat_dict.values())
        logging.info("Extracted, flattened, and concatenated job requirements.")
        return job_reqs_str

    def extract_all(self):
        """
        Extract all job requirements from the job posting, excluding the first-level key.

        Returns:
            dict: A dictionary containing all job requirements.

        Example:
            >>> parser = JobRequirementsParser('job_requirements.json', 'https://example.com/job1')
            >>> all_reqs = parser.extract_all()
            >>> print(all_reqs)
            {
                "pie_in_the_sky": ["PhD in Computer Science"],
                "down_to_earth": ["5+ years of experience in software development"],
                "other": ["English proficiency required"]
            }
        """
        if self.job_reqs_dict:
            all_requirements = self.job_reqs_dict
            logging.info("Fetched all job requirements.")
            return all_requirements
        else:
            logging.error("Invalid job requirements data.")
            return None
