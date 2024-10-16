"""
Filename: responsibility_pruner.py
Author: Xiao-Fei Zhang
Last Updated On: 
"""

import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Union
from scipy.spatial.distance import cdist
from kneed import KneeLocator
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

logger = logging.getLogger(__name__)

JSON = Union[Dict[str, Any], List[Any], str, int, float, bool, None]


# Class to prune resume's responsibilityies
class ResponsibilitiesPruner:
    """
    A class for pruning responsibilities based on various scoring metrics and the elbow method.

    This class provides functionality to analyze, rank, and prune responsibilities
    using composite scores and PCA scores. It implements the elbow method for determining
    the optimal number of responsibilities to retain.

    What to optimize:
    - Composite score is the primary factor used to sort the responsibilities.
    - PCA score is used as a secondary factor for ranking, especially in cases
      where responsibilities have similar composite scores.

    Attributes:
        df (pd.DataFrame): The input DataFrame containing responsibility and requirement data.
        composite_col (str): The name of the column containing composite scores.
        pca_col (str): The name of the column containing PCA scores.
        responsibility_key_col (str): The name of the column containing unique responsibility keys.
        requirement_key_col (str): The name of the column containing unique requirement keys.
    """

    def __init__(self, df):
        """
        Initialize the ResponsibilitiesPruner with a DataFrame.

        Args:
            df (pd.DataFrame): The input DataFrame containing responsibility and requirement data.

        Raises:
            ValueError: If required columns are missing from the DataFrame.
        """
        # Constructor to initialize key variables from DataFrame
        self.df = df
        self.composite_col = "composite_score"
        self.pca_col = "pca_score"
        self.responsibility_key_col = "responsibility_key"
        self.requirement_key_col = "requirement_key"

        # Validate that the DataFrame contains all the required columns
        self._validate_dataframe()
        logger.info(
            f"ResponsibilitiesPruner initialized with DataFrame of shape {self.df.shape}"
        )

    def _validate_dataframe(self):
        """
        Validate that the input DataFrame contains all required columns.

        Raises:
            ValueError: If any required column is missing from the DataFrame.
        """
        required_cols = [
            self.composite_col,
            self.pca_col,
            self.responsibility_key_col,
            self.requirement_key_col,
        ]
        # Identify missing columns if any
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in DataFrame: {', '.join(missing_cols)}"
            )

    def rank_responsibilities(self):
        """
        Rank the responsibilities based on composite and PCA scores.

        Returns:
            pd.DataFrame: The DataFrame with responsibilities ranked by composite and PCA scores.
        """
        # Rank based on composite and PCA scores
        self.df["composite_rank"] = self.df[self.composite_col].rank(ascending=False)
        self.df["pca_rank"] = self.df[self.pca_col].rank(ascending=False)
        ranked_df = self.df.sort_values(
            by=[self.composite_col, self.pca_col], ascending=[False, False]
        )

        logger.info(
            f"Responsibilities ranked. Top composite score: {ranked_df[self.composite_col].max():.4f}"
        )
        return ranked_df

    def find_best_match(self, group_by_responsibility: bool = True):
        """
        Find the single best matching requirement(s) based on the max composite score.

        Args:
            group_by_responsibility (bool): If True, groups by responsibility_key and finds the best match for each.
                                            If False, considers the entire dataset without grouping.

        Returns:
            pd.DataFrame: The DataFrame with best matches identified and scored.
        """
        if group_by_responsibility:
            # Group by responsibility key and find the best matching requirement for each
            grouped = self.df.groupby(self.responsibility_key_col)
            best_matches = grouped.apply(
                lambda x: x.loc[x[self.composite_col].idxmax()]
            )
            self.df = best_matches.reset_index(drop=True)
            logger.info(
                f"Best matches found. Number of unique responsibilities: {len(self.df)}"
            )
        else:
            # Use the entire DataFrame without grouping
            self.df = self.df.sort_values(
                by=self.composite_col, ascending=False
            ).reset_index(drop=True)
            logger.info(
                f"Data sorted without grouping. Total responsibilities: {len(self.df)}"
            )
        return self.df

    def elbow_method_pruning(
        self,
        score_column: str = "composite_score",
        max_k: int = 10,
        S: float = 1.0,
        elbow_curve_plot_file: str = None,
    ):
        """
        Apply the elbow method to determine the optimal number of responsibilities to retain.

        Args:
            score_column (str): The name of the column containing the scores to be used for pruning.
            max_k (int): The maximum number of clusters to test.
            S (float): Sensitivity parameter for KneeLocator.

        Returns:
            pd.DataFrame: The pruned DataFrame containing the optimal number of responsibilities.
        """
        # Sort the DataFrame by the specified score column and reshape for KMeans
        sorted_df = self.df.sort_values(by=score_column, ascending=False).reset_index(
            drop=True
        )
        scores = sorted_df[score_column].values.reshape(-1, 1)

        sum_of_squared_distances = []
        K = range(1, min(max_k, len(scores)) + 1)
        for k in K:
            km = KMeans(n_clusters=k, random_state=42)
            km = km.fit(scores)
            sum_of_squared_distances.append(km.inertia_)

        # Detect the elbow point (optimal number of clusters to retain)
        kneedle = KneeLocator(
            K, sum_of_squared_distances, S=S, curve="convex", direction="decreasing"
        )
        optimal_num_responsibilities = (
            kneedle.knee or max_k
        )  # Default to max_k if knee not found

        pruned_df = sorted_df.iloc[:optimal_num_responsibilities]

        # Plot the SSD vs. K curve (elbow curve) and save (Optional)
        if elbow_curve_plot_file:
            plt.figure(figsize=(8, 6))
            plt.plot(K, sum_of_squared_distances, "bx-")
            plt.xlabel("Number of Clusters (K)")
            plt.ylabel("Sum of Squared Distances (Inertia)")
            plt.title("Elbow Method For Optimal K")
            plt.savefig(elbow_curve_plot_file)

        logger.info(
            f"Elbow method pruning complete. Optimal number of responsibilities: {optimal_num_responsibilities}"
        )
        return pruned_df

    def manual_selection_pruning(self, prune_fraction: float = 0.1):
        """
        Manually select the top N responsibilities based on composite score.

        Args:
            prune_fraction: The fractuib if responsibilities to be pruned.

        Returns:
            pd.DataFrame: The pruned DataFrame containing the top N responsibilities.
        """
        if not 0 < prune_fraction <= 1:
            raise ValueError("coverage_fraction must be between 0 and 1.")

        sorted_df = self.df.sort_values(
            by=self.composite_col, ascending=False
        ).reset_index(drop=True)
        total_responsibilities = len(sorted_df)
        num_responsibilities = int(total_responsibilities * (1 - prune_fraction))
        num_responsibilities = max(
            1, num_responsibilities
        )  # Ensure at least one responsibility is selected

        pruned_df = sorted_df.iloc[:num_responsibilities]
        logger.info(
            f"Manual selection pruning complete. Number of responsibilities retained: {len(pruned_df)} "
            f"out of {total_responsibilities} {(1-prune_fraction)*100:.1f}% are prun)."
        )
        return pruned_df

    def threshold_based_pruning(
        self, score_column: str = "composite_score", threshold: float = 0.5
    ):
        """
        Prune responsibilities based on a score threshold.

        Args:
            score_column (str): The name of the score column to use for thresholding.
            threshold (float): The score threshold.

        Returns:
            pd.DataFrame: The pruned DataFrame containing responsibilities above the threshold.
        """
        pruned_df = self.df[self.df[score_column] >= threshold].reset_index(drop=True)
        logger.info(
            f"Threshold pruning complete. Number of responsibilities retained: {len(pruned_df)}"
        )
        return pruned_df

    def generate_json_output(self, pruned_df):
        """
        Generate a JSON output format where each responsibility is associated with its unique key.

        Args:
            pruned_df (pd.DataFrame): The DataFrame containing the pruned responsibilities.

        Returns:
            dict: A dictionary with responsibility_key as keys and responsibility as values.
        """
        if pruned_df.empty:
            logger.warning("Pruned DataFrame is empty, generating empty JSON.")
            return {}

        # Check if responsibility_key_col exists and has valid entries
        if (
            self.responsibility_key_col not in pruned_df.columns
            or pruned_df[self.responsibility_key_col].isnull().all()
        ):
            logger.error("Responsibility key column is missing or has invalid values.")
            return {}

        try:
            # Converst to JSON using the respoonsibility_key as the index first
            pruned_df = pruned_df.set_index(self.responsibility_key_col)
            output_json = pruned_df["responsibility"].to_dict()
            logger.info(
                f"JSON output generated with {len(output_json)} responsibilities"
            )
            return output_json
        except Exception as e:
            logger.error(f"Error generating JSON output: {e}")
            return {}

    def run_pruning_process(
        self, method: str = "elbow", group_by_responsibility: bool = True, **kwargs
    ) -> JSON:
        """
        Run the full pruning process: rank responsibilities, find best matches, and prune.

        Args:
            method (str): The pruning method to use. Options are 'elbow', 'manual', 'threshold'.
            group_by_responsibility (bool): Whether to group by responsibility_key when finding best matches.
            **kwargs: Additional arguments for the pruning method.

        Returns:
            dict: A dictionary containing the pruned responsibilities.
        """
        logger.info(f"Starting pruning process using {method} method")

        try:
            ranked_df = self.rank_responsibilities()
            matched_df = self.find_best_match(
                group_by_responsibility=group_by_responsibility
            )

            if method == "elbow":
                pruned_df = self.elbow_method_pruning(**kwargs)
            elif method == "manual":
                pruned_df = self.manual_selection_pruning(**kwargs)
            elif method == "threshold":
                pruned_df = self.threshold_based_pruning(**kwargs)
            else:
                raise ValueError(f"Unsupported pruning method: {method}")

            output_json = self.generate_json_output(pruned_df)
            logger.info("Pruning process completed successfully")
            return output_json

        except Exception as e:
            logger.error(f"Error during pruning process: {str(e)}")
            raise

    def get_pruning_stats(self, original_df, pruned_df):
        """
        Generate statistics about the pruning process.

        Args:
            original_df (pd.DataFrame): The original DataFrame before pruning.
            pruned_df (pd.DataFrame): The pruned DataFrame.

        Returns:
            dict: A dictionary containing various statistics about the pruning process.
        """
        stats = {
            "original_count": len(original_df),
            "pruned_count": len(pruned_df),
            "reduction_percentage": (1 - len(pruned_df) / len(original_df)) * 100,
            "original_score_mean": original_df[self.composite_col].mean(),
            "pruned_score_mean": pruned_df[self.composite_col].mean(),
            "original_score_median": original_df[self.composite_col].median(),
            "pruned_score_median": pruned_df[self.composite_col].median(),
        }
        logger.info(f"Pruning stats: {stats}")
        return stats

    def plot_score_distribution(
        self, score_column="composite_score", save_path="score_distribution.png"
    ):
        plt.figure(figsize=(8, 6))
        self.df[score_column].hist(bins=20)
        plt.title(f"Distribution of {score_column}")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Score distribution plot saved to {save_path}")


# Revised Class by Claude
class RevisedResponsibilityPruner:
    def __init__(self, df):
        self.df = df
        self.composite_col = "composite_score"
        self.pca_col = "pca_score"
        self.responsibility_key_col = "responsibility_key"
        self.requirement_key_col = "requirement_key"

    def rank_and_score(self):
        # Rank based on composite and PCA scores
        self.df["composite_rank"] = self.df[self.composite_col].rank(ascending=False)
        self.df["pca_rank"] = self.df[self.pca_col].rank(ascending=False)

        # Calculate a combined score
        self.df["combined_score"] = (
            self.df["composite_rank"] + self.df["pca_rank"]
        ) / 2
        return self.df.sort_values("combined_score")

    def adaptive_pruning(self, target_percentage=0.8, step=0.05):
        ranked_df = self.rank_and_score()
        total_responsibilities = len(ranked_df[self.responsibility_key_col].unique())
        target_count = int(total_responsibilities * target_percentage)

        current_percentage = 1.0
        pruned_df = ranked_df

        while len(pruned_df[self.responsibility_key_col].unique()) > target_count:
            current_percentage -= step
            threshold = ranked_df["combined_score"].quantile(current_percentage)
            pruned_df = ranked_df[ranked_df["combined_score"] <= threshold]

        return pruned_df

    def plot_score_distribution(self):
        plt.figure(figsize=(10, 6))
        self.df[self.composite_col].hist(bins=30, alpha=0.5, label="Composite Score")
        self.df[self.pca_col].hist(bins=30, alpha=0.5, label="PCA Score")
        plt.title("Distribution of Scores")
        plt.xlabel("Score")
        plt.ylabel("Frequency")
        plt.legend()
        plt.show()

    def run_pruning_process(self, target_percentage=0.8):
        self.plot_score_distribution()
        pruned_df = self.adaptive_pruning(target_percentage)
        return self.generate_json_output(pruned_df)

    def generate_json_output(self, pruned_df):
        return pruned_df.set_index(self.responsibility_key_col)[
            "responsibility"
        ].to_dict()
