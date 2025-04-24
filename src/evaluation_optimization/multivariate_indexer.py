"""
Filename: multivariate_indexer.py
Author: Xiao-Fei Zhang
Last updated on:
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from typing import List, Dict, Optional
import logging
import logging_config

# Set logger
logger = logging.getLogger(__name__)

DEFAULT_COMPOSITE_WEIGHT = {
    "roberta_entailment_score": 0.30,  # Added to simulate ATS system (more conservative)
    "deberta_entailment_score": 0.15,  # Match with the metric name
    "soft_similarity": 0.40,  # Match with the metric name
    "word_movers_distance": 0.10,  # Match with the metric name
    "bert_score_precision": 0.05,  # Match with the metric name
}

DEFAULT_COMPOSITE_METRICS = [
    "roberta_entailment_score",
    "deberta_entailment_score",
    "soft_similarity",
    "word_movers_distance",
    "bert_score_precision",
]

DEFAULT_COLUMN_ORDER = [
    "url",
    "responsibility_key",
    "responsibility",
    "requirement_key",
    "requirement",
    "bert_score_precision",
    "soft_similarity",
    "word_movers_distance",
    "deberta_entailment_score",
    "roberta_entailment_score",
    "bert_score_precision_cat",
    "soft_similarity_cat",
    "word_movers_distance_cat",
    "deberta_entailment_score_cat",
    "roberta_entailment_score_cat",
    "scaled_bert_score_precision",
    "scaled_soft_similarity",
    "scaled_word_movers_distance",
    "scaled_deberta_entailment_score",
    "scaled_roberta_entailment_score",
    "composite_score",
    "pca_score",
]


class MultivariateIndexer:
    """
    A class for calculating and analyzing composite scores and principal components
    from multivariate data, designed for use in various index construction tasks.

    This class handles scaling, normalizing, and aggregating multiple metrics into
    composite scores, including weighted composites and dimensionality reduction via
    Principal Component Analysis (PCA). It is useful for reducing the complexity of
    multivariate datasets while maintaining interpretability through key indices.

    Attributes:
    -----------
    df: pd.DataFrame
        The input DataFrame containing data for a single iteration of analysis.
    resp_key : str
        Column name representing the unique key for responsibilities 
        (default is "responsibility_key").
    req_key : str
        Column name representing the unique key for requirements 
        (default is "requirement_key").
    metrics : list
        List of metric column names to analyze. If not provided, default metrics are used.
    max_word_movers : float
        Maximum value for normalizing the word movers distance metric (default is 1.0).
    composite_weights : dict
        Dictionary specifying the weights to be used for calculating a weighted 
        composite score.
    scaler : object
        The scaler (e.g., MinMaxScaler or StandardScaler) used for normalizing metrics 
        before PCA or composite calculations.

    Methods:
    --------
    scale_metrics():
        Scales the metrics in the DataFrame using the specified scaler \
            (MinMaxScaler or StandardScaler).

    calculate_composite_score(row):
        Calculates a weighted composite score for a given row of scaled metrics.

    add_composite_score_to_df():
        Adds the weighted composite score for all rows in the DataFrame.

    calculate_pca_score():
        Scales the metrics, applies PCA, and adds the PCA-derived score to the DataFrame.

    processed_df():
        Returns the DataFrame after both composite and PCA scores have been calculated and added.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        resp_key: str = "responsibility_key",
        req_key: str = "requirement_key",
        metrics: Optional[List[str]] = None,
        max_word_movers: float = 1.0,
        composite_weights: Optional[Dict[str, float]] = None,
        scaler_type: str = "minmax",
    ):
        """
        Initialize the MultivariateIndexer class with a single DataFrame.

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

    def validate_metrics(self):
        """
        Validate that the required metrics are present in the DataFrame.

        Returns:
        - True if all required metrics are present, raises ValueError otherwise.
        """
        missing_metrics = [
            metric for metric in self.metrics if metric not in self.df.columns
        ]
        if missing_metrics:
            raise ValueError(
                f"The following metrics are missing from the DataFrame: {missing_metrics}"
            )
        return True

    def scale_and_add_metrics_to_df(self):
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

        # Reindex just to ensure
        scaled_df.index = self.df.index

        # Add or update each scaled column
        # (it avoids duplicates columns if the headers already exist)
        for col in scaled_df.columns:
            self.df[col] = scaled_df[col]

        logger.info("Metrics scaled.")

        logger.info(
            f"scaled columns added: {scaled_df.columns}"
        )  # todo: debug; delete later

    def calculate_composite_score(self, row):
        """
        Calculate the composite score based on weighted contributions from scaled metrics.

        Parameters:
        - row: Row of the DataFrame containing the necessary scaled metrics.

        Returns:
        - Composite score (float)
        """
        roberta_entailment = row["scaled_roberta_entailment_score"]
        deberta_entailment = row["scaled_deberta_entailment_score"]
        soft_similarity = row["scaled_soft_similarity"]
        word_movers = row["scaled_word_movers_distance"]
        bert_score = row["scaled_bert_score_precision"]

        # Calculate the composite score using scaled and weighted values
        composite_score = (
            roberta_entailment * self.composite_weights["roberta_entailment_score"]
            + deberta_entailment * self.composite_weights["deberta_entailment_score"]
            + soft_similarity * self.composite_weights["soft_similarity"]
            + word_movers
            * self.composite_weights[
                "word_movers_distance"
            ]  # Already reversed in scaling (do not reverse here again!)
            + bert_score * self.composite_weights["bert_score_precision"]
        )

        return composite_score

    def calculate_add_composite_scores_to_df(self):
        """
        Calculate and insert the composite score into the DataFrame.

        This method adds a new column 'Composite_Score' to the DataFrame.
        """
        # Scale the metrics first
        self.scale_and_add_metrics_to_df()

        # Apply composite score calculation row by row
        self.df["composite_score"] = self.df.apply(
            self.calculate_composite_score, axis=1
        )

        logger.info("composite_score calculated and added/updated.")

        return self.df

    def calculate_add_pca_scores_to_df(self):
        """
        Calculate the composite score using PCA based on the provided metrics.

        Returns:
        - DataFrame with the PCA score added.
        """
        # Ensure all metrics are scaled using StandardScaler
        # scaler = StandardScaler()

        # Reverse polarity for word movers distance (lower score -> higher similarity)
        if "word_movers_distance" in self.df.columns:
            self.df["word_movers_distance"] = 1 - MinMaxScaler().fit_transform(
                self.df[["word_movers_distance"]]
            )

        # Apply scaling to the rest of the metrics
        scaled_values = self.scaler.fit_transform(self.df[self.metrics])

        # Apply PCA to the scaled metrics
        pca = PCA(n_components=1)
        self.df["pca_score"] = pca.fit_transform(scaled_values)

        logger.info("pca_score calculated and added/updated.")

        return self.df

    def clean_and_reorder_columns(
        self, desired_col_order: Optional[list[str]] = None
    ) -> None:
        """
        Drops 'Unnamed' columns and reorders columns based on the desired order list.
        Modifies self.df in place.

        Args:
            desired_col_order (list[str], optional):
                - A list of column names in the exact order you want them.
                - Columns not in self.df are ignored, and columns not in desired_order
                remain at the end.
                - If None, no reordering is performed.
        """
        # 1. Drop columns that start with 'Unnamed'
        self.df = self.df.loc[:, ~self.df.columns.str.startswith("Unnamed")]

        # 2. Drop columns that are entirely empty (all NaN)
        self.df.dropna(axis=1, how="all", inplace=True)

        # 3. Reorder columns if a desired order is provided
        if desired_col_order is not None:
            valid_order = [col for col in desired_col_order if col in self.df.columns]
            self.df = self.df.reindex(
                columns=valid_order
                + [c for c in self.df.columns if c not in valid_order]
            )

        # 4. Log final column order for debugging
        logging.info("After cleaning & reordering, columns are:")
        logging.info(self.df.columns.tolist())

    def add_multivariate_indices_to_df(
        self, desired_col_order: Optional[List[str]] = None
    ):
        """
        Return the DataFrame after applying metric calculations.

        Returns:
            DataFrame with composite score and PCA score added.
        """
        self.calculate_add_composite_scores_to_df()
        self.calculate_add_pca_scores_to_df()

        # Reorder and clean columns as specified by desired_col_order
        if desired_col_order is None:
            desired_col_order = DEFAULT_COLUMN_ORDER

        self.clean_and_reorder_columns(desired_col_order)

        return self.df
