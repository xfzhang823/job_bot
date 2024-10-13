""" Temp for testing """

import pandas as pd
from evaluation_optimization.resume_pruner import ResponsibilitiesPruner


def main():

    adjustment = 10
    file_path = r"C:\github\job_bot\data\testing_pruner_data.csv"
    # Create the DataFrame
    df = pd.read_csv(file_path)

    pruner = ResponsibilitiesPruner(df)
    results = pruner.run_pruning_process(max_k=7, S=6)

    print(f"S adjustment: {adjustment}")
    print(f"original: {len(df)}")
    print(f"after: {len(results)}")


if __name__ == "__main__":
    main()
