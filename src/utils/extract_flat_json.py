"""extract_flat_json.py"""

import logging
import loggings_config
from preprocessing.resume_preprocessor import ResumeParser
from preprocessing.requirements_preprocessor import JobRequirementsParser


logger = logging.getLogger(__name__)
