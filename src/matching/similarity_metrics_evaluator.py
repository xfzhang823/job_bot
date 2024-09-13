# metrics_evaluator.py

# Conventional metric criteria suggested by GPT
STANDARD_METRIC_CRITERIA = {
    # "BERTScore Precision"
    "bert_score_precision": {
        "range": (0, 1),
        "high_threshold": 0.85,
        "low_threshold": 0.70,
    },
    # "Soft Similarity (SBERT)"
    "soft_similarity": {
        "range": (-1, 1),
        "high_threshold": 0.7,
        "low_threshold": 0.4,
    },
    # "Word Mover's Distance"
    "word_movers_distance": {
        "range": (0, float("inf")),
        "high_threshold": 5,  # High score is considered "Low" for this metric
        "low_threshold": 15,  # Low score is considered "High" for this metric
        "reverse": True,  # Indicates smaller scores are better
    },
    # "NLI Entailment Score"
    "nli_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.7,
        "low_threshold": 0.3,
    },
    # "Jaccard Similarity"
    "jaccard_similarity": {
        "range": (0, 1),
        "high_threshold": 0.5,
        "low_threshold": 0.2,
    },
    # "DeBERTa Entailment Score"
    "deberta_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.8,
        "low_threshold": 0.4,
    },
}

# Define a more custom metric criteria in a dictionary
METRIC_CRITERIA = {
    # "BERTScore Precision"
    "bert_score_precision": {
        "range": (0, 1),
        "high_threshold": 0.85,
        "low_threshold": 0.75,
    },
    # "Soft Similarity (SBERT)"
    "soft_similarity": {
        "range": (-1, 1),
        "high_threshold": 0.5,
        "low_threshold": 0.2,
    },
    # "Word Mover's Distance"
    "word_movers_distance": {
        "range": (0, float("inf")),
        "high_threshold": 3,  # High score is considered "Low" for this metric
        "low_threshold": 8,  # Low score is considered "High" for this metric
        "reverse": True,  # Indicates smaller scores are better
    },
    # "DeBERTa Entailment Score"
    "deberta_entailment_score": {
        "range": (0, 1),
        "high_threshold": 0.75,
        "low_threshold": 0.25,
    },
    # # "Jaccard Similarity"
    # "jaccard_similarity": {
    #     "range": (0, 1),
    #     "high_threshold": 0.25,
    #     "low_threshold": 0.05,
    # },
}


def evaluate_score(metric_name, score):
    """
    Evaluate the score for a given metric name.

    Args:
        metric_name (str): The name of the metric.
        score (float): The score to evaluate.

    Returns:
        str: The category of the score ("High", "Medium", "Low").
    """
    if metric_name not in METRIC_CRITERIA:
        raise ValueError(f"Metric '{metric_name}' is not defined in the criteria.")

    criteria = METRIC_CRITERIA[metric_name]
    high_threshold = criteria["high_threshold"]
    low_threshold = criteria["low_threshold"]
    reverse = criteria.get("reverse", False)

    # For metrics where a lower score is better (like Word Mover's Distance)
    if reverse:
        if score <= high_threshold:
            return "High"
        elif score >= low_threshold:
            return "Low"
    else:
        if score >= high_threshold:
            return "High"
        elif score <= low_threshold:
            return "Low"

    return "Medium"


def categorize_scores(scores):
    """
    Categorize a dictionary of scores based on their metric names.

    Args:
        scores (dict): A dictionary where keys are metric names and values are scores.

    Returns:
        dict: A dictionary with the same keys but with "High", "Medium", or "Low" as values.
    """
    categorized_scores = {}
    for metric_name, score in scores.items():
        try:
            categorized_scores[metric_name] = evaluate_score(metric_name, score)
        except ValueError as e:
            print(e)
            categorized_scores[metric_name] = "Unknown"
    return categorized_scores


def categorize_scores_of_row(row):
    # List of metric names and corresponding categories
    metrics = [
        "bert_score_precision",
        "soft_similarity",
        "word_movers_distance",
        "deberta_entailment_score",
    ]

    # Loop over each metric and apply the evaluate_score function
    for metric in metrics:
        # Create a new column name for the category
        category_col = f"{metric}_cat"
        # Apply evaluate_score to get 'High', 'Medium', 'Low' for each metric
        row[category_col] = evaluate_score(metric, row[metric])

    return row


if __name__ == "__main__":
    # Example usage
    example_scores = {
        "bert_score_precision": 0.88,
        "soft_similarity": 0.45,
        "word_movers_distance": 4.2,
        "nli_entailment_score": 0.2,
        "jaccard_similarity": 0.6,
        "deberta_entailment_score": 0.75,
    }

    categorized = categorize_scores(example_scores)
    print("Categorized Scores:", categorized)
