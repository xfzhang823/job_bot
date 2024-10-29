""" Temp for testing """

import pandas as pd
from pydantic import ValidationError
from models.resume_job_description_io_models import SimilarityMetrics
from utils.validation_utils import validate_dataframe_with_pydantic
from utils.llm_api_utils import call_claude_api


def main():
    PROMPT = "Tell me a joke about Python."

    response = call_claude_api(prompt=PROMPT)

    print(response)


if __name__ == "__main__":
    main()
