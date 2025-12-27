"""
responsibility_filters.py

Tiny helpers for excluding specific resume responsibilities.

Current scope (intentionally small):
    - drop_responsibility_keys(df, keys_to_drop)

The function operates on DataFrames that have a 'responsibility_key' column.
No fancy patterns, no URL logic – just "kill this key" behavior.
"""

from __future__ import annotations

from typing import Sequence
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def drop_responsibility_keys(
    df: pd.DataFrame,
    keys_to_drop: Sequence[str] | None = None,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Remove rows whose `responsibility_key` is in `keys_to_drop`.

    If the required column is missing, or `keys_to_drop` is empty/None,
    the original DataFrame is returned unchanged.

    Args
    ----
    df :
        Input DataFrame. Expected to have a 'responsibility_key' column.
    keys_to_drop :
        Iterable of responsibility keys (e.g. "4.responsibilities.5")
        that should be removed.
    inplace :
        If True, mutate `df` in place and return it.
        If False (default), return a filtered copy.

    Returns
    -------
    pd.DataFrame
        The filtered DataFrame (or the original if no changes were made).

    Examples
    --------
    >>> KEYS_TO_SKIP = ["4.responsibilities.5"]

    # 1) Non-destructive usage
    >>> from job_bot.utils.responsibility_filters import drop_responsibility_keys
    >>> cleaned = drop_responsibility_keys(df, KEYS_TO_SKIP)

    # 2) In-place usage
    >>> drop_responsibility_keys(df, KEYS_TO_SKIP, inplace=True)
    >>> # df is now filtered
    """
    if not keys_to_drop:
        # Nothing to do
        return df

    if "responsibility_key" not in df.columns:
        logger.warning(
            "drop_responsibility_keys: 'responsibility_key' column missing; "
            "returning DataFrame unchanged."
        )
        return df

    keys_set = set(keys_to_drop)
    mask = ~df["responsibility_key"].isin(keys_set)

    if inplace:
        df.drop(df.index[~mask], inplace=True)
        return df

    return df.loc[mask].copy()
