"""
Class to compare metrics for each iteration of responsibilities to requirements modification.
"""

import logging
import logging_config
from typing import List
import pandas as pd
import numpy as np
from evaluation_optimization.evaluation_optimization_utils import DataMerger


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


def get_dataframe_dim(df):

    rows, cols = df.shape
    logger.info(f"Number of rows: {rows}\nNumber of columns: {cols}")


class Metrics_Comparator:
    """
    Class responsible for comparing metrics across different iterations or groups.
    """

    def __init__(self, dfs: list, metrics: list, resp_key: str, req_key: str):
        """
        Initialize Comparator class with multiple DataFrames.

        Parameters:
        - dfs (List[pd.DataFrame]): List of DataFrames from different iterations.
        - metrics (list): List of metrics to analyze. If not provided, default metrics will be used.
        - resp_key (str): Column name representing the unique key for responsibilities.
        - req_key (str): Column name representing the unique key for requirements.
        """
        self.dfs = dfs
        self.metrics = metrics
        self.resp_key = resp_key
        self.req_key = req_key
        self.df_merged = None

    def merge_and_compare(self):
        """
        Merge DataFrames and compare metrics.
        """
        # Call the DataMerger class to merge the DataFrames
        data_merger = DataMerger(self.dfs, self.resp_key, self.req_key)
        self.df_merged = data_merger.merge_dataframes()

        # Perform metric comparisons
        for metric in self.metrics:
            comparison_col = f"{metric}_comparison"
            self.df_merged[comparison_col] = (
                self.df_merged[f"{metric}_1"] - self.df_merged[f"{metric}_0"]
            )
        return self.df_merged


class MetricsComparator_old_version:
    """
    Class to compare similarity-related metrics from multiple iterations.
    """

    def __init__(
        self,
        dfs: List[pd.DataFrame],
        resp_key: str = "responsibility_Key",
        req_key: str = "requirement_Key",
        metrics: list = None,
    ):
        """
        Initialize the MetricsComparison class with multiple processed DataFrames.

        Parameters:

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

    # Internal method to group by responsibility_key
    def _group_by_responsibility(self) -> pd.DataFrame:
        """Group DataFrame by responsibility_key."""
        numeric_cols = self.df_merged.select_dtypes(
            include=["int64", "float64"]
        ).columns
        return self.df_merged.groupby(self.resp_key)[numeric_cols].mean().reset_index()

    # Internal method to group by requirement_key
    def _group_by_requirement(self) -> pd.DataFrame:
        """Group DataFrame by requirement_key."""
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

    def get_full_data_with_composite_score(self):
        """
        Method to return the full DataFrame, including the composite score, for any output method.
        """
        return self.df_merged


class ChangeCalculator:
    """
    Class responsible for calculating absolute and percentage changes between metrics.
    """

    def __init__(self, df: pd.DataFrame, metrics: list):
        """
        Initialize ChangeCalculator class.

        Parameters:
        - df: The DataFrame containing the data.
        - metrics: The list of metrics to calculate changes for.
        """
        self.df = df
        self.metrics = metrics

    def calculate_absolute_change(self):
        """
        Calculate absolute changes for each metric.

        Returns:
        - DataFrame with absolute change columns.
        """
        changes = {}
        for metric in self.metrics:
            change_col = f"{metric}_change"
            self.df[change_col] = self.df[f"{metric}_1"] - self.df[f"{metric}_0"]
            changes[metric] = change_col
        return self.df, changes

    def calculate_percentage_change(self):
        """
        Calculate percentage changes for each metric, handling division by zero.

        Returns:
        - DataFrame with percentage change columns.
        """
        pct_changes = {}
        for metric in self.metrics:
            pct_change_col = f"{metric}_pct_change"
            self.df[pct_change_col] = (
                self.df[f"{metric}_1"] - self.df[f"{metric}_0"]
            ) / self.df[f"{metric}_0"].replace(0, np.nan)
            pct_changes[metric] = pct_change_col
        return self.df, pct_changes
