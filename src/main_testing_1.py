""" Temp for testing """

import pandas as pd
from pydantic import ValidationError
from models.resume_job_description_io_models import SimilarityMetrics
from utils.validation_utils import validate_dataframe_with_pydantic


def main():
    f_path = r"C:\github\job_bot\input_output\evaluation_optimization\iteration_1\similarity_metrics\Amazon_Product_Manager__Artificial_General_Intelligence_-_Data_Services_sim_metrics_iter1.csv"

    df = pd.read_csv(f_path)

    validated_rows = validate_dataframe_with_pydantic(df)
    print(validated_rows)


if __name__ == "__main__":
    main()
