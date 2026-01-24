"""utils/resp_key_sorter.py"""

import re
import pandas as pd

_RESP_KEY_RE = re.compile(r"^(?:(?P<prefix>.*\D))?(?P<num>\d+)$")


def resp_key_sorter(s: pd.Series) -> pd.Series:
    """
    Natural sort for keys like:
      '1.responsibilities.2'  -> num=2
      '1.responsibilities.10' -> num=10

    Sort primarily by the non-numeric prefix, then by the trailing integer.
    """
    s = s.astype(str)

    # Split into [prefix, num] where num is trailing digits
    extracted = s.str.extract(_RESP_KEY_RE)
    prefix = extracted["prefix"].fillna("")  # e.g. '1.responsibilities.'
    num = pd.to_numeric(extracted["num"], errors="coerce")

    # Build a sortable composite:
    # - prefix as-is
    # - numeric part as an integer (NaNs pushed to end)
    num_filled = num.fillna(10**12).astype("int64")
    return prefix + num_filled.astype(str).str.zfill(12)
