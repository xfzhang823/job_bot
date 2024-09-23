"""
Class to compare metrics for each iteration of responsibilities to requirements modification.
"""

import logging
import logging_config
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

# Set up logger
logger = logging.getLogger(__name__)

DEFAULT_COMPOSITE_WEIGHT = {
    "deberta_entailment": 0.45,
    "soft_similarity": 0.35,
    "word_movers": 0.15,
    "alberta_score": 0.05,
}

DEFAULT_COMPOSITE_METRICS = [
    "bert_score_precision",
    "deberta_entailment_score",
    "soft_similarity",
    "word_movers_distance",
]


def get_df_dim(df):
    rows, cols = df.shape
    print(f"Number of rows: {rows}\nNumber of columns: {cols}")


class MetricsCalculator:
    """TBA"""

    def __init__(
        self,
        df: pd.DataFrame,
        resp_key: str = "Responsibility_Key",
        req_key: str = "Requirement_Key",
        metrics: list = None,
        max_word_movers: float = 1.0,
        composite_weights: dict = None,
        scaler_type: str = "minmax",
    ):
        """
        Initialize the MetricsCalculator class with a single DataFrame.

        Parameters:
        - df (pd.DataFrame): DataFrame for a single iteration.
        - resp_key (str): Column name representing the unique key for responsibilities.
        - req_key (str): Column name representing the unique key for requirements.
        - metrics (list): List of metrics to analyze. If not provided, default metrics will be used.
        - max_word_movers (float): Maximum value to normalize word movers.
        - composite_weights (dict): Weights for calculating composite score for metrics.
        """
        self.df = df
        self.resp_key = resp_key
        self.req_key = req_key
        self.max_word_movers = max_word_movers
        self.composite_weights = composite_weights or DEFAULT_COMPOSITE_WEIGHT
        self.metrics = metrics or DEFAULT_COMPOSITE_METRICS
        self.scaler = MinMaxScaler() if scaler_type == "minmax" else StandardScaler()

    def scale_metrics(self):
        """
        Scale the metrics in the DataFrame using the specified scaler.

        Returns:
        - Scaled DataFrame
        """
        # Select only the columns to be scaled
        metrics_df = self.df[self.metrics]

        # Fit and transform the metrics DataFrame
        scaled_values = self.scaler.fit_transform(metrics_df)

        # Create a new DataFrame with the scaled values
        scaled_df = pd.DataFrame(
            scaled_values, columns=[f"scaled_{col}" for col in self.metrics]
        )

        # Reverse polarity for word movers distance (lower score -> higher similarity)
        scaled_df["scaled_word_movers_distance"] = (
            1 - scaled_df["scaled_word_movers_distance"]
        )

        # Append scaled columns back to the original DataFrame
        self.df = pd.concat([self.df, scaled_df], axis=1)

    def calculate_composite_score(self, row):
        """
        Calculate the composite score based on weighted contributions from scaled metrics.

        Parameters:
        - row: Row of the DataFrame containing the necessary scaled metrics.

        Returns:
        - Composite score (float)
        """
        deberta_entailment = row["scaled_deberta_entailment_score"]
        soft_similarity = row["scaled_soft_similarity"]
        word_movers = row["scaled_word_movers_distance"]
        alberta_score = row["scaled_bert_score_precision"]

        # Reverse polarity for word movers distance (lower score -> higher similarity)
        normalized_word_movers = 1 - word_movers

        # Calculate the composite score using scaled and weighted values
        composite_score = (
            deberta_entailment * self.composite_weights["deberta_entailment"]
            + soft_similarity * self.composite_weights["soft_similarity"]
            + normalized_word_movers * self.composite_weights["word_movers"]
            + alberta_score * self.composite_weights["alberta_score"]
        )

        return composite_score

    def add_composite_score_to_df(self):
        """
        Calculate and insert the composite score into the DataFrame.

        This method adds a new column 'Composite_Score' to the DataFrame.
        """
        # Scale the metrics first
        self.scale_metrics()

        # Apply composite score calculation row by row
        self.df["Composite_Score"] = self.df.apply(
            self.calculate_composite_score, axis=1
        )

        return self.df

    def add_pca_score_to_df(self):
        """
        Apply PCA on the scaled metrics to reduce them to a single composite score.

        Returns:
        - Updated DataFrame with PCA composite score.
        """
        # Select scaled metrics
        scaled_metrics = self.df[[f"scaled_{metric}" for metric in self.metrics]]

        # Apply PCA
        pca = PCA(n_components=1)
        self.df["PCA_Score"] = pca.fit_transform(scaled_metrics)

    def calculate_pca_score(self):
        """
        Calculate the composite score using PCA based on the provided metrics.

        Returns:
        - DataFrame with the PCA score added.
        """
        scaler = StandardScaler()
        pca = PCA(n_components=1)

        # Normalize word movers score to be between 0 and 1
        metrics_to_normalize = ["word_movers_distance"]
        self.df[metrics_to_normalize] = MinMaxScaler().fit_transform(
            self.df[metrics_to_normalize]
        )

        # Apply PCA to get a single component
        self.df["PCA_Score"] = pca.fit_transform(
            scaler.fit_transform(self.df[self.metrics])
        )

    def processed_df(self):
        """
        Return the DataFrame after applying metric calculations.

        Returns:
        - DataFrame with composite score and PCA score added.
        """
        self.add_composite_score_to_df()
        self.add_pca_score_to_df()
        return self.df


class MetricsComparison:
    """
    Class to compare similarity-related metrics from multiple iterations.
    """

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        resp_key: str = "Responsibility_Key",
        req_key: str = "Requirement_Key",
        metrics: list = None,
    ):
        """
        Initialize the MetricsComparison class with multiple processed DataFrames.

        Parameters:
        - dfs (List[pd.DataFrame]): List of DataFrames from different iterations.
        - resp_key (str): Column name representing the unique key for responsibilities.
        - req_key (str): Column name representing the unique key for requirements.
        - metrics (list): List of metrics to analyze. If not provided, default metrics will be used.
        """
        self.dfs = dfs
        self.resp_key = resp_key
        self.req_key = req_key
        self.metrics = metrics or [
            "bert_score_precision",
            "deberta_entailment_score",
            "soft_similarity",
            "word_movers_distance",
            "Composite_Score",
            "PCA_Score",
        ]
        self.df_merged = None

    # Internal method to group by Responsibility_Key
    def _group_by_responsibility(self) -> pd.DataFrame:
        """Group DataFrame by Responsibility_Key."""
        numeric_cols = self.df_merged.select_dtypes(
            include=["int64", "float64"]
        ).columns
        return self.df_merged.groupby(self.resp_key)[numeric_cols].mean().reset_index()

    # Internal method to group by Requirement_Key
    def _group_by_requirement(self) -> pd.DataFrame:
        """Group DataFrame by Requirement_Key."""
        numeric_cols = self.df_merged.select_dtypes(
            include=["int64", "float64"]
        ).columns
        return self.df_merged.groupby(self.req_key)[numeric_cols].mean().reset_index()

    # Internal method to calculate absolute changes
    def _calculate_absolute_change(self, df_grouped: pd.DataFrame) -> pd.DataFrame:
        """Calculate absolute changes for each metric."""
        changes = {}
        for metric in self.metrics:
            change_col = f"{metric}_change"
            df_grouped[change_col] = (
                df_grouped[f"{metric}_1"] - df_grouped[f"{metric}_0"]
            )
            changes[metric] = change_col
        return df_grouped, changes

    # Internal method to calculate percentage changes
    def _calculate_percentage_change(self, df_grouped: pd.DataFrame) -> pd.DataFrame:
        """Calculate percentage changes for each metric, handling division by zero."""
        pct_changes = {}
        for metric in self.metrics:
            pct_change_col = f"{metric}_pct_change"
            df_grouped[pct_change_col] = (
                df_grouped[f"{metric}_1"] - df_grouped[f"{metric}_0"]
            ) / df_grouped[f"{metric}_0"].replace(
                0, np.nan
            )  # Prevent division by zero
            pct_changes[metric] = pct_change_col
        return df_grouped, pct_changes

    def _merge_dataframes(self):
        """Merge multiple DataFrames on Responsibility_Key and Requirement_Key."""
        df_merged = self.dfs[0]
        for i, df in enumerate(self.dfs[1:], start=1):
            df_merged = pd.merge(
                df_merged,
                df,
                on=[self.resp_key, self.req_key],
                suffixes=(f"_{i-1}", f"_{i}"),
                how="inner",
            )
        self.df_merged = df_merged

    def calculate_changes(
        self, by_pct: bool = False, group_by: str = None
    ) -> pd.DataFrame:
        """
        Calculate changes between previous and current metrics, with options for absolute or percentage change.

        Parameters:
        - by_pct (bool): If True, return percentage changes; if False, return absolute value changes.
        - group_by (str): If 'responsibility', group by Responsibility_Key; if 'requirement', group by Requirement_Key.

        Returns:
        pd.DataFrame with either absolute changes or percentage changes, optionally grouped.
        """
        if group_by == "responsibility":
            df_grouped = self._group_by_responsibility()
            key_col = self.resp_key
        elif group_by == "requirement":
            df_grouped = self._group_by_requirement()
            key_col = self.req_key
        else:
            df_grouped = self.df_merged
            key_col = [self.resp_key, self.req_key]

        if by_pct:
            df_grouped, changes = self._calculate_percentage_change(df_grouped)
        else:
            df_grouped, changes = self._calculate_absolute_change(df_grouped)

        if isinstance(key_col, list):
            df_change = df_grouped[key_col + list(changes.values())]
        else:
            df_change = df_grouped[[key_col] + list(changes.values())]

        total_col = "total_pct_change" if by_pct else "total_change"
        df_change.loc[:, total_col] = (
            df_change[list(changes.values())].abs().sum(axis=1)
        )

        return df_change.sort_values(by=total_col, ascending=False)

    def execute_comparison(self):
        """
        Execute the comparison process: merge dataframes and calculate changes.
        """
        # Step 1: Merge DataFrames
        self._merge_dataframes()

        # Step 2: Calculate and return changes
        return self.calculate_changes()

    # Methods for summary_stats, most improved requirements, etc., remain the same.
    def summary_stats(self, group_by: str = None) -> pd.DataFrame:
        """
        Generate summary statistics (mean, min, max, var) for each metric.

        Parameters:
        - group_by (str): If 'responsibility', group by Responsibility_Key; if 'requirement', group by Requirement_Key.

        Returns:
        pd.DataFrame with summary statistics.
        """
        # Optionally group the data
        if group_by == "responsibility":
            df_grouped = self._group_by_responsibility()
        elif group_by == "requirement":
            df_grouped = self._group_by_requirement()
        else:
            df_grouped = self.df_merged

        stats_df = df_grouped.groupby(
            self.resp_key if group_by == "responsibility" else self.req_key
        )[
            [f"{metric}_0" for metric in self.metrics]
            + [f"{metric}_1" for metric in self.metrics]
        ].agg(
            ["mean", "min", "max", "var"]
        )

        # Flatten MultiIndex columns
        stats_df.columns = ["_".join(col) for col in stats_df.columns]
        return stats_df

    def aggregate_metric_analysis(self, group_by: str = None) -> pd.DataFrame:
        """
        Perform aggregate analysis of metrics (mean and standard deviation).

        Parameters:
        - group_by (str): If 'responsibility', group by Responsibility_Key; if 'requirement', group by Requirement_Key.

        Returns:
        pd.DataFrame with aggregate analysis of metrics (mean, std).
        """
        # Optionally group the data
        if group_by == "responsibility":
            df_grouped = self._group_by_responsibility()
        elif group_by == "requirement":
            df_grouped = self._group_by_requirement()
        else:
            df_grouped = self.df_merged

        # Aggregate analysis
        return df_grouped[
            [f"{metric}_0" for metric in self.metrics]
            + [f"{metric}_1" for metric in self.metrics]
        ].agg(["mean", "std"])

    def metric_change_analysis(self, group_by: str = None) -> pd.DataFrame:
        """
        Analyze which metrics represent the most changes between the two iterations, using average absolute percentage change.

        Parameters:
        - group_by (str): If 'responsibility', group by Responsibility_Key; if 'requirement', group by Requirement_Key.

        Returns:
        pd.DataFrame showing the average absolute percentage change for each metric.
        """
        # Optionally group the data
        if group_by == "responsibility":
            df_grouped = self._group_by_responsibility()
        elif group_by == "requirement":
            df_grouped = self._group_by_requirement()
        else:
            df_grouped = self.df_merged

        # Calculate the average absolute percentage change for each metric
        metric_change_summary = {}
        for metric in self.metrics:
            metric_change_summary[f"{metric}_avg_abs_change"] = (
                df_grouped[f"{metric}_pct_change"]
                .abs()
                .mean()  # Average of absolute changes
            )

        # Convert the summary into a DataFrame
        metric_change_df = pd.DataFrame.from_dict(
            metric_change_summary, orient="index", columns=["avg_abs_pct_change"]
        )

        # Sort the metrics by the average absolute percentage change
        return metric_change_df.sort_values(by="avg_abs_pct_change", ascending=False)

    def most_improved_requirements(self) -> pd.DataFrame:
        """
        Return the requirements that improved their scores the most (by average percentage change).

        Returns:
        pd.DataFrame showing the most improved requirements.
        """
        # Check if percentage change columns are available; if not, calculate them
        pct_change_cols = [f"{metric}_pct_change" for metric in self.metrics]

        if not all(col in self.df_merged.columns for col in pct_change_cols):
            # Calculate percentage changes if they haven't been calculated
            self.calculate_changes(by_pct=True)

        # Calculate average percentage change across all metrics
        self.df_merged["avg_pct_change"] = self.df_merged[pct_change_cols].mean(axis=1)

        # Return the rows with the most improvement (sorted by average percentage change)
        return self.df_merged[[self.req_key, "avg_pct_change"]].sort_values(
            by="avg_pct_change", ascending=False
        )

    def most_improved_responsibilities(self) -> pd.DataFrame:
        """
        Return the responsibilities that improved their scores the most (by average percentage change).

        Returns:
        pd.DataFrame showing the most improved responsibilities.
        """
        # Check if percentage change columns are available; if not, calculate them
        pct_change_cols = [f"{metric}_pct_change" for metric in self.metrics]

        if not all(col in self.df_merged.columns for col in pct_change_cols):
            # Calculate percentage changes if they haven't been calculated
            self.calculate_changes(by_pct=True)

        # Calculate average percentage change across all metrics
        self.df_merged["avg_pct_change"] = self.df_merged[pct_change_cols].mean(axis=1)

        # Return the rows with the most improvement (sorted by average percentage change)
        return self.df_merged[[self.resp_key, "avg_pct_change"]].sort_values(
            by="avg_pct_change", ascending=False
        )

    def show_descripitive_stats(self):
        """
        Example method that returns summary statistics including the composite score.
        """
        return self.df_merged.describe()

    def get_top_responsibilities_by_composite_score(
        self, previous_or_current="current", n=5
    ):
        """
        Method that returns the top N responsibilities by composite score.

        Parameters:
        - previous_or_current (str): "current" metrics or "previous" metrics
        - n (int): Number of top responsibilities to return (default is 5).
        """
        if previous_or_current == "current":
            return self.df_merged.nlargest(n, "Composite_Score_1")[
                [self.resp_key, "Composite_Score_1", "Responsibility_1"]
            ]
        elif previous_or_current == "previous":
            return self.df_merged.nlargest(n, "Composite_Score_0")[
                [self.resp_key, "Composite_Score_0", "Responsibility_0"]
            ]
        else:
            raise ValueError(
                "Invalid option for 'previous_or_current'. Use 'current' or 'previous'."
            )

    def get_full_data_with_composite_score(self):
        """
        Method to return the full DataFrame, including the composite score, for any output method.
        """
        return self.df_merged
