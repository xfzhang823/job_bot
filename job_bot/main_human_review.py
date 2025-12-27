"""
Entry point for the Underlines → Selection pipeline.

This script:
    • Builds a list of XLSX alignment-review files
    • Calls run_underlines_selection_pipeline()
    • Prints a clean summary

You can:
    • Hardcode a list of XLSX files (as below)
    • OR read the list from your database / FSM
    • OR use globbing by passing xlsx_paths=None
"""

from __future__ import annotations

import logging
from pathlib import Path

from job_bot.pipelines_with_fsm.underlines_to_selection_pipeline_fsm import (
    run_underlines_selection_pipeline,
)
from job_bot.config.project_config import EXCEL_DIR


def build_worklist() -> list[Path]:
    """
    Build the list of XLSX files to process.

    Here we use a hardcoded list (as requested), but you can replace this with:
        • A database query to pipeline_control
        • A glob inside EXCEL_DIR
        • Any other dynamic logic

    Returns
    -------
    list[Path]
        Paths to XLSX alignment review workbooks.
    """

    # EXAMPLE HARDCODED LIST — EDIT AS NEEDED
    return [
        EXCEL_DIR / "brown-brown_director-of-business-intelligence_0000003602.xlsx",
        # EXCEL_DIR / "coreweave_sr-manager-market-research-intelligence_4613475006.xlsx",
        # EXCEL_DIR / "mediabrands_director-intelligence-solutions_4800265007.xlsx",
        # EXCEL_DIR / "offerfit_senior-ai-success-manager_4472546005.xlsx",
        # EXCEL_DIR / "research-partnership_director-market-research_7306002.xlsx",
        # EXCEL_DIR / "wattswater_manager-strategic-marketing-research-bus_10015633.xlsx",
        # EXCEL_DIR / "aws_principal-product-manager-technical-agi-_3017572.xlsx",
        # EXCEL_DIR / "aws_sr-am-genai-startups-genai-startup-team_3052771.xlsx",
    ]


def main() -> int:
    """
    Main entrypoint for running the Underlines → Selection batch pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, non-zero on failure.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ----------------------------------------------------------------------
    # 1. Build worklist
    # ----------------------------------------------------------------------
    xlsx_files = build_worklist()

    if not xlsx_files:
        print("No XLSX files found to process.")
        return 0

    print("\n📄 Files to process:")
    for f in xlsx_files:
        print(f"   - {f}")

    # ----------------------------------------------------------------------
    # 2. Run pipeline
    # ----------------------------------------------------------------------
    print("\n🚀 Running Underlines → Selection pipeline...\n")

    results = run_underlines_selection_pipeline(
        xlsx_paths=xlsx_files,
        sheet_name="review_grid",
        log_level="INFO",
        stop_on_error=False,
    )

    # ----------------------------------------------------------------------
    # 3. Summary
    # ----------------------------------------------------------------------
    total_rows = sum(len(df) for (df, _, _) in results.values())

    print("\n✅ Pipeline completed.")
    print(f"   Processed files: {len(results)}")
    print(f"   Total extracted rows: {total_rows}")

    for xlsx_path, (df, csv_p, json_p) in results.items():
        print(f"\n[{xlsx_path}]")
        print(f"   rows : {len(df)}")
        print(f"   CSV  : {csv_p}")
        print(f"   JSON : {json_p}")

    print("\nDone.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
