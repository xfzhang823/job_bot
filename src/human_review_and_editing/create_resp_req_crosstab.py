"""resps_vs_reqs_viewer.py"""

from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd


# Setup logger
logger = logging.getLogger(__name__)


def create_resp_req_crosstab(file_path: Path | str, score_threshold: float = 0.0):
    """
    Creates a cross-tabulation where:
    - Shows full text for scores >= threshold
    - Shows only the score for scores < threshold

    Args:
        file_path: Path to the CSV file
        score_threshold: Score threshold for showing full text
    """
    # Read necessary columns
    needed_cols = [
        "responsibility_key",
        "responsibility",
        "requirement_key",
        "requirement",
        "composite_score",
    ]
    df = pd.read_csv(file_path, usecols=needed_cols)

    # Get unique keys
    resp_keys = df["responsibility_key"].unique()
    req_keys = df["requirement_key"].unique()

    # Create requirement text dictionary for the header row
    req_text_dict = (
        df.drop_duplicates("requirement_key")
        .set_index("requirement_key")["requirement"]
        .to_dict()
    )

    # Create responsibility text and score matrix
    resp_matrix = {}
    for resp_key in resp_keys:
        resp_matrix[resp_key] = {}
        resp_df = df[df["responsibility_key"] == resp_key]
        for _, row in resp_df.iterrows():
            resp_matrix[resp_key][row["requirement_key"]] = {
                "responsibility": row["responsibility"],
                "score": row["composite_score"],
            }

    # Create the output DataFrame
    result_data = {"Resp Key / Req Key": ["Requirements"]}

    # Generate alternating column names (Requirement 1, Score 1, Requirement 2, Score 2, ...)
    alternating_columns = []
    for req_key in req_keys:
        alternating_columns.append(f"{req_text_dict[req_key]}")
        alternating_columns.append(f"{req_text_dict[req_key]} (Score)")

    # Add header row
    result_data.update({col: [""] for col in alternating_columns})

    # Add responsibility rows
    for resp_key in resp_keys:
        row_data = [resp_key]  # First column is the responsibility key
        for req_key in req_keys:
            entry = resp_matrix[resp_key].get(req_key, {})
            responsibility = entry.get("responsibility", "")
            score = entry.get("score", "")

            # Determine what to display based on score
            if score and score >= score_threshold:
                display_text_responsibility = responsibility
            else:
                display_text_responsibility = ""

            display_text_score = f"{score:.3f}" if score != "" else ""

            # Append both responsibility and score to interleave
            row_data.append(display_text_responsibility)
            row_data.append(display_text_score)

        # Add the row to result_data
        for col, value in zip(["Resp Key / Req Key"] + alternating_columns, row_data):
            result_data[col].append(value)

    # Create final DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df


import pandas as pd
from pathlib import Path


def save_crosstab(
    df: pd.DataFrame, output_path: Path | str, file_format: str = "csv"
) -> None:
    """
    Saves the cross-tabulation DataFrame to a file in either CSV or Excel format.

    This function ensures:
    - The output directory exists.
    - Write permissions are checked before saving.
    - Numeric values remain numbers in Excel.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        output_path (Path | str): The full file path (CSV or Excel).
        file_format (str): File format ("csv" or "excel"). Defaults to "csv".

    Raises:
        PermissionError: If the directory cannot be written to.
        ValueError: If an unsupported file format is provided.
    """
    # Ensure `output_path` is a `Path` object
    if isinstance(output_path, str):
        output_path = Path(output_path)

    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # Check write permission using `Path`
    try:
        test_file = output_dir / "write_test.tmp"
        test_file.touch(exist_ok=False)  # ✅ Try creating a temp file
        test_file.unlink()  # ✅ Remove test file immediately
    except Exception as e:
        raise PermissionError(f"❌ No write permission for {output_dir}: {e}")

    # ✅ Convert numeric values correctly (but keep text as text)
    df = df.infer_objects()  # Converts obvious numeric types
    df = df.apply(pd.to_numeric, errors="ignore")  # Keeps text intact

    # ✅ Save to file
    if file_format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif file_format.lower() == "excel":
        df.to_excel(output_path, index=False, sheet_name="Crosstab", engine="openpyxl")
    else:
        raise ValueError(
            f"❌ Unsupported file format: {file_format}. Use 'csv' or 'excel'."
        )

    print(f"File saved successfully: {output_path}")

    # todo: comment out for now.
    # # Configure Excel writer to adjust column widths
    # with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
    #     df.to_excel(writer, index=False, sheet_name="Crosstab")
    #     worksheet = writer.sheets["Crosstab"]

    #     # Adjust column widths
    #     for idx, col in enumerate(df.columns):
    #         max_length = max(df[col].astype(str).apply(len).max(), len(str(col)))
    #         # Limit max width to avoid extremely wide columns
    #         adjusted_width = min(max_length + 2, 100)
    #         worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width
