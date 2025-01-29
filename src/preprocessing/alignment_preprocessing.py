import numpy as np
import pandas as pd
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def load_data(
    composite_scores_file: str, positions_file: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load composite scores and position data from CSV files.

    Args:
        composite_scores_file (str): Path to the composite scores CSV file.
        positions_file (str): Path to the positions CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames containing composite scores and positions data.
    """
    try:
        composite_scores = pd.read_csv(composite_scores_file)
        positions = pd.read_csv(positions_file)
        return composite_scores, positions
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise ValueError("Failed to load data from CSV files.") from e


def scale_positions(positions: pd.Series, max_position_value: int = 20000) -> pd.Series:
    """
    Scale position values using logarithmic scaling and reverse the polarity.

    Args:
        positions (pd.Series): The raw position values.
        max_position_value (int): Maximum possible position value to normalize against.

    Returns:
        pd.Series: Scaled position values between 0 and 1 with reversed polarity.
    """
    try:
        scaled_positions = np.log(positions + 1) / np.log(max_position_value + 1)
        return 1 - scaled_positions  # Reverse the polarity
    except Exception as e:
        logger.error(f"Error scaling positions: {e}")
        raise ValueError("Failed to scale positions.") from e
