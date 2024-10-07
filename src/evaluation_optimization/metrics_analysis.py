"""
TBA
"""

import pandas as pd

class MetricsAnalyer:
    """
    Class responsible for analyzing metrics, generating summary statistics, and aggregating data.
    """

    def __init__(self, df: pd.DataFrame, metrics: list, resp_key: str, req_key: str):
        self.df = df
        self.metrics = metrics
        self.resp_key = resp_key
        self.req_key = req_key

    def summary_stats(self, group_by: str = None):
        """
        Generate summary statistics (mean, min, max, var) for each metric.
        """
        df_grouped = self.df
        if group_by == "responsibility":
            df_grouped = self.df.groupby(self.resp_key).mean().reset_index()
        elif group_by == "requirement":
            df_grouped = self.df.groupby(self.req_key).mean().reset_index()

        stats_df = df_grouped.agg(["mean", "min", "max", "var"])
        return stats_df

    def aggregate_metric_analysis(self):
        """
        Perform aggregate analysis of metrics (mean and standard deviation).
        """
        return self.df[self.metrics].agg(["mean", "std"])

    def metric_change_analysis(self):
        """
        Analyze which metrics represent the most changes between the two iterations.
        """
        metric_change_summary = {}
        for metric in self.metrics:
            metric_change_summary[f"{metric}_avg_abs_change"] = (
                self.df[f"{metric}_pct_change"].abs().mean()
            )
        return pd.DataFrame.from_dict(
            metric_change_summary, orient="index", columns=["avg_abs_pct_change"]
        )
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