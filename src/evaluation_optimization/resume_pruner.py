"""
Filename: responsibility_pruner.py
Author: Xiao-Fei Zhang
Last Updated On: 
"""

import pandas as pd
import numpy as np


class ResponsibilityMatcher:
    def __init__(
        self,
        df,
        composite_col="composite_score",
        pca_col="pca_score",
        metric_col="metric_score",
    ):
        """
        Initialize the matcher with a DataFrame containing responsibilities, requirements, and scores.

        Args:
        - df (pd.DataFrame): DataFrame containing responsibilities, requirements, and scores.
        - composite_col (str): Column name for the composite score.
        - pca_col (str): Column name for the PCA score.
        - metric_col (str): Column name for the metric score.
        """
        self.df = df
        self.composite_col = composite_col
        self.pca_col = pca_col
        self.metric_col = metric_col

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
        For each responsibility, find the single best matching requirement based on the metric score.
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
                best_match_idx = matching_requirements[self.metric_col].idxmax()
                best_match_score = matching_requirements[self.metric_col].max()

                best_match_indices.append(best_match_idx)
                best_match_scores.append(best_match_score)
            else:
                best_match_indices.append(None)
                best_match_scores.append(None)

        self.df["best_match_idx"] = best_match_indices
        self.df["best_match_score"] = best_match_scores
        return self.df

    def prune_responsibilities(self, threshold=0.5, prune_percentage=0.2):
        """
        Prune responsibilities that have no good matches or few good matches based on the threshold.
        Aim to knock out 10% to 20% of the responsibilities.

        Args:
        - threshold (float): The threshold score below which responsibilities are considered poorly matched.
        - prune_percentage (float): The percentage of responsibilities to prune.

        Returns:
        - Pruned DataFrame.
        """
        # Filter out responsibilities with poor matches
        filtered_df = self.df[self.df["best_match_score"] >= threshold]

        # Calculate the number of responsibilities to prune
        prune_count = int(prune_percentage * len(self.df))

        # Sort by best match score and prune the bottom entries
        pruned_df = filtered_df.sort_values(by="best_match_score", ascending=True).iloc[
            prune_count:
        ]
        return pruned_df

    def run(self, threshold=0.5, prune_percentage=0.2):
        """
        Run the full pipeline: rank responsibilities, find best matches, and prune.

        Args:
        - threshold (float): The threshold score for pruning.
        - prune_percentage (float): The percentage of responsibilities to prune.

        Returns:
        - Pruned DataFrame with ranked responsibilities.
        """
        print("Ranking responsibilities...")
        ranked_df = self.rank_responsibilities()

        print("Finding best matches...")
        matched_df = self.find_best_match()

        print(f"Pruning {prune_percentage * 100}% of responsibilities...")
        pruned_df = self.prune_responsibilities(
            threshold=threshold, prune_percentage=prune_percentage
        )

        return pruned_df


def filter_responsibilities_by_low_scores(df, fields):
    """
    Filters out responsibilities where all specified fields have 'Low' scores for all requirements.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing responsibility and requirement data.
    - fields (list): A list of column names to check for 'Low' values.

    Returns:
    - responsibilities_to_optimize (np.array): An array of unique responsibilities
    that do not have 'Low' scores for all requirements in the specified fields.
    """
    # Group the data by Responsibility and aggregate the counts of "Low" scores for each specified field
    aggregation_dict = {field: lambda x: (x == "Low").sum() for field in fields}
    aggregation_dict["Requirement Key"] = (
        "count"  # Count the number of requirements associated with each responsibility
    )

    grouped = df.groupby("Responsibility").agg(aggregation_dict).reset_index()

    # Filter responsibilities where all specified fields have "Low" scores for all requirements
    filter_condition = grouped[fields[0]] == grouped["Requirement Key"]
    for field in fields[1:]:
        filter_condition &= grouped[field] == grouped["Requirement Key"]

    filtered_responsibilities = grouped[filter_condition]

    # Get responsibilities that don't match the above criteria
    responsibilities_to_optimize = df[
        ~df["Responsibility"].isin(filtered_responsibilities["Responsibility"])
    ]["Responsibility"].unique()

    return responsibilities_to_optimize


# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        "responsibility": ["Resp 1", "Resp 2", "Resp 3"],
        "requirement": ["Req A", "Req B", "Req C"],
        "composite_score": [0.8, 0.6, 0.9],
        "pca_score": [0.85, 0.7, 0.95],
        "metric_score": [0.75, 0.65, 0.85],
    }

    df = pd.DataFrame(data)

    matcher = ResponsibilityMatcher(df)
    pruned_df = matcher.run(threshold=0.6, prune_percentage=0.15)

    print(pruned_df)
