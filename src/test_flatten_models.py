import json
from pathlib import Path
from pprint import pprint
from IPython.display import display
from typing import Union
import pandas as pd

from models.resume_job_description_io_models import Requirements, NestedResponsibilities
from utils.flatten_pyd_models import (
    flatten_requirements_model,
    flatten_nested_responsibilities_model,
)


def test_flatten_models(
    requirements_path: Union[str, Path], responsibilities_path: Union[str, Path]
):
    """
    Loads JSON files, parses into Pydantic models, flattens them, and prints results.

    Args:
        requirements_path (str | Path): Path to the job requirements JSON file.
        responsibilities_path (str | Path): Path to the nested responsibilities JSON file.
    """

    requirements_path = Path(requirements_path)
    responsibilities_path = Path(responsibilities_path)

    print(f"\n📥 Loading: {requirements_path}")
    with requirements_path.open("r", encoding="utf-8") as f:
        raw_reqs = json.load(f)
        req_model = Requirements.model_validate(raw_reqs)
        req_flat = flatten_requirements_model(req_model)
        # print(f"\n🔎 Flattened Requirements ({len(req_flat)} rows):")
        # pprint(req_flat[:3])  # Preview first 3 rows
        req_df = pd.DataFrame(req_flat)
        display(req_df)

    print(f"\n📥 Loading: {responsibilities_path}")
    with responsibilities_path.open("r", encoding="utf-8") as f:
        raw_resps = json.load(f)
        resp_model = NestedResponsibilities.model_validate(raw_resps)
        resp_flat = flatten_nested_responsibilities_model(resp_model)
        # print(f"\n🔎 Flattened Responsibilities ({len(resp_flat)} rows):")
        # pprint(resp_flat[:3])  # Preview first 3 rows
        resp_df = pd.DataFrame(resp_flat)
        display(resp_df)
    return req_df, resp_df


pd.set_option("display.max_columns", None)  # Show all columns
pd.set_option("display.width", 200)  # Set wider display width
pd.set_option("display.max_colwidth", None)  # Don't truncate column values


# Example usage (update paths as needed):
file1 = r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_openai\iteration_1\requirements\Accenture_Enterprise_AI_Value_Strategy_Senior_Manager_reqs_flat_iter1.json"
file2 = r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_openai\iteration_1\responsibilities\S_P_Global_Ratings_Director_of_Data_Science___RAG__NLP__LLM_and_GenAI__Hybrid_or_Virtual__resps_nested_iter1.json"
df1, df2 = test_flatten_models(file1, file2)
df1.to_csv("temp_1.csv", index=True)
df2.to_csv("temp_2.csv", index=False)
