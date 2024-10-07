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
        resp_key: str = "Responsibility_Key",
        req_key: str = "Requirement_Key",
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

    # def _merge_dataframes(self):
    #     """Merge multiple DataFrames on Responsibility_Key and Requirement_Key."""
    #     df_merged = self.dfs[0]
    #     for i, df in enumerate(self.dfs[1:], start=1):
    #         df_merged = pd.merge(
    #             df_merged,
    #             df,
    #             on=[self.resp_key, self.req_key],
    #             suffixes=(f"_{i-1}", f"_{i}"),
    #             how="inner",
    #         )
    #     self.df_merged = df_merged

    # def calculate_changes(
    #     self, by_pct: bool = False, group_by: str = None
    # ) -> pd.DataFrame:
    #     """
    #     Calculate changes between previous and current metrics, with options for absolute or percentage change.

    #     Args:
    #         -by_pct (bool): If True, return percentage changes; if False, return absolute value changes.
    #         -group_by (str): If 'responsibility', group by Responsibility_Key; if 'requirement', \
    #             group by Requirement_Key.

    #     Returns:
    #         pd.DataFrame with either absolute changes or percentage changes, optionally grouped.
    #     """
    #     if group_by == "responsibility":
    #         df_grouped = self._group_by_responsibility()
    #         key_col = self.resp_key
    #     elif group_by == "requirement":
    #         df_grouped = self._group_by_requirement()
    #         key_col = self.req_key
    #     else:
    #         df_grouped = self.df_merged
    #         key_col = [self.resp_key, self.req_key]

    #     if by_pct:
    #         df_grouped, changes = self._calculate_percentage_change(df_grouped)
    #     else:
    #         df_grouped, changes = self._calculate_absolute_change(df_grouped)

    #     if isinstance(key_col, list):
    #         df_change = df_grouped[key_col + list(changes.values())]
    #     else:
    #         df_change = df_grouped[[key_col] + list(changes.values())]

    #     total_col = "total_pct_change" if by_pct else "total_change"
    #     df_change.loc[:, total_col] = (
    #         df_change[list(changes.values())].abs().sum(axis=1)
    #     )

    #     return df_change.sort_values(by=total_col, ascending=False)

    # def execute_comparison(self):
    #     """
    #     Execute the comparison process: merge dataframes and calculate changes.
    #     """
    #     # Step 1: Merge DataFrames
    #     self._merge_dataframes()

    #     # Step 2: Calculate and return changes
    #     return self.calculate_changes()

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
