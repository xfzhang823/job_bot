"""
Filename: responsibility_pruner.py
Author: Xiao-Fei Zhang
Last Updated On: 
"""

import json
import pandas as pd
import numpy as np
import logging
import logging_config

logger = logging.getLogger(__name__)


class ResponsibilitiesPruner:
    """TBA"""

    def __init__(
        self,
        df,
        composite_col="composite_score",
        pca_col="pca_score",
        responsibility_key_cols=["responsibility_key", "responsibility"],
    ):
        """
        Initialize the pruner with a DataFrame containing responsibilities, requirements, and scores.

        Args:
        - df (pd.DataFrame): DataFrame containing responsibilities, requirements, and scores.
        - composite_col (str): Column name for the composite score.
        - pca_col (str): Column name for the PCA score.
        - responsibility_key_cols (list): List of possible column names for responsibility_key.
        """
        self.df = df
        self.composite_col = composite_col
        self.pca_col = pca_col

        # Determine which responsibility_key column exists
        for col in responsibility_key_cols:
            if col in df.columns:
                self.responsibility_key_col = col
                break
        else:
            raise ValueError(
                "No valid responsibility_key column found in the DataFrame."
            )

    def _normalize_column_names(self):
        """
        This function converts the column names of the class's DataFrame to lowercase
        and replaces spaces with underscores.

        Modifies:
        - self.df (pd.DataFrame): Updates the DataFrame's column names in place.

        Returns:
        - self.df (pd.DataFrame): The DataFrame with normalized column names.
        """
        # Normalize the column names of the class's DataFrame
        self.df.columns = self.df.columns.str.lower().str.replace(" ", "_")
        return self.df

    def filter_responsibilities_by_low_scores(self, fields):
        """
        Filters out responsibilities where all specified fields have 'Low' scores for all requirements.

        Parameters:
        - fields (list): A list of column names to check for 'Low' values.

        Returns:
        - responsibilities_to_optimize (np.array): An array of unique responsibilities
        that do not have 'Low' scores for all requirements in the specified fields.
        """
        aggregation_dict = {field: lambda x: (x == "Low").sum() for field in fields}
        aggregation_dict["requirement_key"] = (
            "count"  # Count the number of requirements
        )

        # Group by responsibility and calculate the count of 'Low' values per responsibility
        grouped = (
            self.df.groupby(self.responsibility_key_col)
            .agg(aggregation_dict)
            .reset_index()
        )

        # Filter responsibilities where all specified fields have "Low" scores for all requirements
        filter_condition = grouped[fields[0]] == grouped["requirement_key"]
        for field in fields[1:]:
            filter_condition &= grouped[field] == grouped["requirement_key"]

        filtered_responsibilities = grouped[filter_condition]

        # Get responsibilities that do not match the "all Low" condition
        responsibilities_to_optimize = self.df[
            ~self.df[self.responsibility_key_col].isin(
                filtered_responsibilities[self.responsibility_key_col]
            )
        ][self.responsibility_key_col].unique()

        return responsibilities_to_optimize

    def rank_responsibilities(self):
        """
        Rank the responsibilities based on composite and PCA scores.
        Returns the DataFrame with responsibilities ranked.
        """
        self.df["composite_rank"] = self.df[self.composite_col].rank(ascending=False)
        self.df["pca_rank"] = self.df[self.pca_col].rank(ascending=False)
        return self.df.sort_values(
            by=[self.composite_col, self.pca_col], ascending=[False, False]
        )

    def find_best_match(self):
        """
        For each responsibility, find the single best matching requirement based on the max composite score.
        Track the best matches and add them to the DataFrame.
        """
        best_match_indices = []
        best_match_scores = []

        for _, responsibility in self.df.iterrows():
            # Filter the matching requirements
            matching_requirements = self.df[
                self.df["responsibility"] == responsibility["responsibility"]
            ]

            if not matching_requirements.empty:
                best_match_idx = matching_requirements[self.composite_col].idxmax()
                best_match_score = matching_requirements[self.composite_col].max()

                best_match_indices.append(best_match_idx)
                best_match_scores.append(best_match_score)
            else:
                best_match_indices.append(None)
                best_match_scores.append(None)

        self.df["best_match_idx"] = best_match_indices
        self.df["best_match_score"] = best_match_scores
        return self.df

    def prune_responsibilities(self, composite_threshold=0.5, prune_percentage=0.2):
        """
        Retain a percentage of responsibilities based on the composite score.

        Args:
        - composite_threshold (float): The threshold score below which responsibilities are pruned.
        - prune_percentage (float): The percentage of responsibilities to prune (e.g., 0.2 = 20%).

        Returns:
        - pruned_df (pd.DataFrame): The DataFrame with the retained responsibilities.
        """
        # Filter based on max composite score threshold
        filtered_df = self.df[self.df["best_match_score"] >= composite_threshold]

        # Group by responsibility_key and calculate the number of unique responsibilities
        grouped_df = (
            filtered_df.groupby(self.responsibility_key_col).max().reset_index()
        )

        # Calculate the number of responsibilities to prune
        prune_count = int(prune_percentage * len(grouped_df))

        # Sort by best match score and retain the top entries (prune the bottom ones)
        pruned_df = grouped_df.sort_values(by="best_match_score", ascending=False).iloc[
            : len(grouped_df) - prune_count
        ]

        logger.info(
            f"Retaining {len(pruned_df)} responsibilities out of {len(grouped_df)}."
        )
        return pruned_df

    def calculate_requirement_coverage(self, pruned_df=None):
        """
        Calculate the percentage of unique requirements that have at least one responsibility matched.

        Args:
        - pruned_df (pd.DataFrame, optional): The DataFrame containing the pruned responsibilities.
        If None, it will calculate coverage based on the full DataFrame.

        Returns:
        - coverage_percentage (float): The percentage of requirements with matched responsibilities.
        """
        # Use the full DataFrame if pruned_df is not provided
        df_to_check = pruned_df if pruned_df is not None else self.df

        # Get unique requirements before and after pruning
        unique_requirements_before = self.df["requirement_key"].nunique()
        unique_requirements_after = df_to_check["requirement_key"].nunique()

        # Calculate the percentage of matched requirements
        coverage_fraction = unique_requirements_after / unique_requirements_before

        logger.info(f"Requirements before pruning: {unique_requirements_before}")
        logger.info(f"Requirements after pruning: {unique_requirements_after}")
        logger.info(f"Coverage percentage: {coverage_fraction:.2f}")

        return coverage_fraction

    def generate_json_output(self, pruned_df):
        """
        Generate a JSON output format where each responsibility is associated with its unique key.

        Args:
        - pruned_df (pd.DataFrame): The DataFrame containing the pruned responsibilities.

        Returns:
        - output_json (dict): A dictionary with the original JSON format (responsibility_key: responsibility).
        """
        if pruned_df.empty:
            logger.warning("Pruned DataFrame is empty, generating empty JSON.")
            return {}

        output_json = pruned_df.set_index(self.responsibility_key_col)[
            "responsibility"
        ].to_dict()
        return output_json

    def run_pruning_process(
        self, composite_threshold=0, prune_percentage=0.1, low_score_fields=None
    ):
        """
        Run the full pipeline: rank responsibilities, find best matches, and prune.

        Args:
        - composite_threshold (float): The threshold score for pruning.
        - prune_percentage (float): The percentage of responsibilities to prune.
        - low_score_fields (list): A list of fields to check for consistently 'Low' scores.

        Returns:
        - Pruned JSON object containing the final responsibilities with their keys.
        """

        logger.info("Ranking responsibilities...")
        ranked_df = self.rank_responsibilities()

        print("Finding best matches...")
        matched_df = self.find_best_match()

        if low_score_fields:
            logger.info(
                "Filtering responsibilities with low scores across all fields..."
            )
            filtered_responsibilities = self.filter_responsibilities_by_low_scores(
                low_score_fields
            )
            print(f"Responsibilities to optimize: {len(filtered_responsibilities)}")
            # Update DataFrame by keeping only responsibilities that passed the low score filter
            self.df = self.df[self.df["responsibility"].isin(filtered_responsibilities)]
            logger.info(f"Pruning {prune_percentage * 100}% of responsibilities...")

        pruned_df = self.prune_responsibilities(
            composite_threshold=composite_threshold, prune_percentage=prune_percentage
        )

        print("Generating JSON output...")
        output_json = self.generate_json_output(pruned_df)

        return output_json


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "responsibility_key": [
            "0.responsibilities.0",
            "0.responsibilities.1",
            "0.responsibilities.2",
        ],
        "responsibility": [
            "Provided strategic insights to a major global IT vendor.",
            "Assisted a U.S.-based international services provider.",
            "Co-authored an industry-recognized report on M&A.",
        ],
        "composite_score": [0.8, 0.6, 0.9],
        "pca_score": [0.85, 0.7, 0.95],
    }

    df = pd.DataFrame(data)
    print(df)

    pruner = ResponsibilitiesPruner(df)
    final_json_output = pruner.run_pruning_process(
        composite_threshold=0.6, prune_percentage=0.15
    )

    print("\nFinal Pruned JSON Output:")
    print(json.dumps(final_json_output, indent=4))
