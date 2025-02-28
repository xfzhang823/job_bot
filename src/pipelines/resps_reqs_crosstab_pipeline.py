"""resps_reqs_crosstab_file_creation_pipeline"""

import logging
from pathlib import Path
import pandas as pd

from human_review_and_editing.create_resp_req_crosstab import (
    create_resp_req_crosstab,
    save_crosstab,
)
from project_config import RESPS_REQS_MATCHING_DIR

logger = logging.getLogger(__name__)

# ✅ List of input files (scalable)
INPUT_FILES = {
    "AI_Data_Specialist": Path(
        r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_anthropic\iteration_0\similarity_metrics\Deloitte_AI_Data_Specialist_sim_metrics_iter0.csv"
    ),
    "Strategy_Manager": Path(
        r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_anthropic\iteration_0\similarity_metrics\Deloitte_Global_Business_Services__GBS__Strategy_Manager_sim_metrics_iter0.csv"
    ),
    "Research_Manager": Path(
        r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_anthropic\iteration_0\similarity_metrics\Deloitte_Market_Research_Sr_Manager_sim_metrics_iter0.csv"
    ),
}


def process_file(
    input_file: Path, output_dir: Path, output_filename: str, score_threshold: float = 0
) -> None:
    """
    Processes a single input file, creates a responsibility-requirements crosstab,
    and saves it to an output file.

    Args:
        input_file (Path): Path to the input similarity metrics CSV.
        output_dir (Path): Path to the output directory.
        output_filename (str): Name of the output file.
        score_threshold (float): Score threshold for showing responsibility text.

    Returns:
        None
    """
    output_file = output_dir / output_filename

    try:
        logger.info(f"Processing: {input_file.name}")

        # ✅ Generate cross-tabulation
        result_df = create_resp_req_crosstab(
            file_path=input_file, score_threshold=score_threshold
        )

        # ✅ Save to Excel
        save_crosstab(result_df, output_file, file_format="excel")
        logger.info(f"✅ Cross-tabulation saved to: {output_file}")

    except Exception as e:
        logger.error(f"❌ Error processing {input_file.name}: {str(e)}")


def run_resps_reqs_crosstab_pipeline(score_threshold: float = 0) -> None:
    """
    Runs the pipeline to process multiple input files and generate responsibility-requirement crosstabs.

    Args:
        score_threshold (float): Score threshold for showing responsibility text.

    Returns:
        None
    """
    output_dir = Path(RESPS_REQS_MATCHING_DIR)

    # ✅ Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"📂 Output Directory: {output_dir.resolve()}")

    # ✅ Process each input file dynamically
    for file_key, file_path in INPUT_FILES.items():
        output_filename = (
            f"{file_key.lower().replace(' ', '_')}_resps_reqs_crosstab.xlsx"
        )
        process_file(file_path, output_dir, output_filename, score_threshold)
