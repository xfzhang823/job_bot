"""
fill_with_originals.py

Materialize rows for edited_responsibilities by overlaying LLM edits onto originals.

This module is intentionally pipeline-callable and pure:
- no DuckDB reads/writes
- no mutation of input dicts
- returns rows/DF ready for insert_df_with_config()

You decide upstream what "pairs_to_materialize" means:
- common choice: ALL pairs that will exist downstream (e.g., your filtered/top-k cohort)
- conservative choice: only eligible_pairs (smaller table, less downstream surface area)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

import pandas as pd

Pair = Tuple[str, str]


@dataclass(frozen=True)
class MaterializeEditedResponsibilitiesConfig:
    """
    Column names for edited_responsibilities table.
    """

    resp_key_col: str = "responsibility_key"
    req_key_col: str = "requirement_key"
    resp_text_col: str = "responsibility"


def materialize_edited_responsibilities_rows(
    *,
    pairs_to_materialize: Iterable[Pair],
    original_responsibilities: Mapping[str, str],
    edited_responsibilities: Mapping[str, str],
    cfg: MaterializeEditedResponsibilitiesConfig = MaterializeEditedResponsibilitiesConfig(),
) -> list[dict[str, Any]]:
    """
    Build row dicts for edited_responsibilities:
      responsibility_key, requirement_key, responsibility

    For each (resp_key, req_key) pair:
      - if resp_key exists in edited_responsibilities -> use edited text
      - else -> use original_responsibilities text

    Raises:
      KeyError if a resp_key is missing from BOTH edited and original mappings.
    """
    rows: list[dict[str, Any]] = []

    for resp_key, req_key in pairs_to_materialize:
        r = str(resp_key)
        q = str(req_key)
        if not r or not q:
            continue

        if r in edited_responsibilities:
            text = edited_responsibilities[r]
        elif r in original_responsibilities:
            text = original_responsibilities[r]
        else:
            raise KeyError(
                f"materialize: responsibility_key '{r}' not found in edited or original mappings."
            )

        rows.append(
            {
                cfg.resp_key_col: r,
                cfg.req_key_col: q,
                cfg.resp_text_col: text,
            }
        )

    return rows


def materialize_edited_responsibilities_df(
    *,
    pairs_to_materialize: Iterable[Pair],
    original_responsibilities: Mapping[str, str],
    edited_responsibilities: Mapping[str, str],
    cfg: MaterializeEditedResponsibilitiesConfig = MaterializeEditedResponsibilitiesConfig(),
) -> pd.DataFrame:
    """
    Convenience wrapper: returns a DataFrame.

    Note: This DF intentionally does NOT include url/iteration/llm_provider/model_id/source_file.
    Stamp those via insert_df_with_config(...).
    """
    rows = materialize_edited_responsibilities_rows(
        pairs_to_materialize=pairs_to_materialize,
        original_responsibilities=original_responsibilities,
        edited_responsibilities=edited_responsibilities,
        cfg=cfg,
    )
    return pd.DataFrame(rows)
