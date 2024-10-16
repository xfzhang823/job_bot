import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


def plot_histograms(self, bins=20):
    """
    Plot histograms of the original metrics, scaled metrics, composite score, and PCA score.

    Parameters:
    - bins (int): Number of bins to use in the histograms (default is 20).
    """
    # Define columns to plot
    metrics_to_plot = self.metrics
    scaled_metrics_to_plot = [
        f"scaled_{metric}"
        for metric in self.metrics
        if f"scaled_{metric}" in self.df.columns
    ]

    # Check if composite score and PCA score columns are in the DataFrame
    extra_metrics = []
    if "Composite_Score" in self.df.columns:
        extra_metrics.append("Composite_Score")
    if "PCA_Score" in self.df.columns:
        extra_metrics.append("PCA_Score")

    # Combine all columns to plot
    columns_to_plot = metrics_to_plot + scaled_metrics_to_plot + extra_metrics

    # Plot histograms
    self.df[columns_to_plot].hist(bins=bins, figsize=(10, 8))
    plt.suptitle("Histograms of Metrics, Composite, and PCA Scores")
    plt.show()
