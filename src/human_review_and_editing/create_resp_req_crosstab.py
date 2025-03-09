"""resps_vs_reqs_viewer.py"""

from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import uuid  # Generate unique test file names

# Setup logger
logger = logging.getLogger(__name__)


def create_resp_req_crosstab(
    file_path: Path | str,
    score_threshold: float = 0.0,
    original_resps_dict: dict[str, str] | None = None,
):
    """
    Creates a cross-tabulation where:
    - Shows full text for scores >= threshold
    - Shows only the score for scores < threshold
    - Optionally, includes an extra column with the original responsibility text
      if an original_resp_mapping is provided.

    Args:
        - file_path: Path to the CSV file.
        - score_threshold: Score threshold for showing full responsibility text.
        - original_resp_mapping: Optional dictionary mapping responsibility keys to
        their original responsibility text.

    Returns:
        A pandas DataFrame containing the cross-tabulation.
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
    # df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce")

    # ✅ Log shape and column details
    logger.info(f"✅ DataFrame Loaded: Shape {df.shape}")
    logger.info(f"📝 Column Names: {df.columns.tolist()}")

    # ✅ Log row count for each column
    row_counts = df.count()
    logger.info(f"📊 Row Counts per Column:\n{row_counts}")

    # ✅ Check if any column has missing values
    missing_counts = df.isna().sum()
    logger.info(f"🚨 Missing Value Counts per Column:\n{missing_counts}")

    # 🚨 If row counts are inconsistent, we have an issue
    if len(set(row_counts.values)) > 1:
        raise ValueError(f"🚨 Inconsistent row counts detected in {file_path}")

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
    # Start with a column for responsibility key
    header = ["Resp Key / Req Key"]

    # Optionally include a header for original responsibility
    if original_resps_dict is not None:
        header.append("Original Responsibility")

    # result_data = {"Resp Key / Req Key": ["Requirements"]}

    # Generate alternating column names for each requirement: text and score columns
    # (Requirement 1, Score 1, Requirement 2, Score 2, ...)
    alternating_columns = []
    for req_key in req_keys:
        alternating_columns.append(f"{req_text_dict[req_key]}")
        alternating_columns.append(f"{req_text_dict[req_key]} (Score)")
    header.extend(alternating_columns)

    # Initialize result data with header row
    result_data = {col: [] for col in header}  # Initialize result data with header row

    # result_data.update({col: [""] for col in alternating_columns})
    # Add the header row (for the top of the table, you might decide to leave it blank or descriptive)

    # Add an empty header row for subsequent rows to align properly.
    for col in header:
        result_data[col].append(
            col if col != "Resp Key / Req Key" else "Responsibilities"
        )

    # Add responsibility rows
    for resp_key in resp_keys:
        row_data = [resp_key]  # First column is the responsibility key

        # If original mapping provided, add original responsibility text for the key
        if original_resps_dict is not None:
            orig_text = original_resps_dict.get(resp_key, "")
            row_data.append(orig_text)

        # Process each requirement column pair
        for req_key in req_keys:
            entry = resp_matrix[resp_key].get(req_key, {})
            responsibility = entry.get("responsibility", "")
            score = entry.get("score", "")

            # Determine what to display based on score
            if score != "" and score >= score_threshold:
                display_text_responsibility = responsibility
            else:
                display_text_responsibility = ""

            # Formatting (set type to float)
            try:
                numeric_score = float(score)
            except (TypeError, ValueError):
                numeric_score = ""
            display_text_score = f"{numeric_score:.3f}" if numeric_score != "" else ""

            # display_text_score = f"{score:.3f}" if score != "" else ""

            # Append both responsibility and score to interleave
            row_data.append(display_text_responsibility)
            row_data.append(display_text_score)

        # Append the row data into result_data for each corresponding column
        for col, value in zip(header, row_data):
            result_data[col].append(value)

        # for col, value in zip(["Resp Key / Req Key"] + alternating_columns, row_data):
        #     result_data[col].append(value)

    # Create final DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df


def save_crosstab(
    df: pd.DataFrame, output_path: Path | str, file_format: str = "csv"
) -> None:
    """
    Saves the cross-tabulation DataFrame to a file in either CSV or Excel format.

    Ensures:
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
    # ✅ Check if the file already exists and skip
    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    if output_path.exists():
        logger.info(f"⏭️ File already exists, skipping: {output_path}")
        return

    output_path = Path(output_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)  # Ensure directory exists

    # ✅ Use a unique test file to avoid conflicts in parallel processes
    test_file = output_dir / f"write_test_{uuid.uuid4().hex}.tmp"

    try:
        test_file.touch(exist_ok=False)  # ✅ Try creating a temp file
        test_file.unlink()  # ✅ Remove test file immediately
    except Exception as e:
        raise PermissionError(f"❌ No write permission for {output_dir}: {e}")

    # ✅ Convert numeric values correctly (but keep text as text)

    # Identify numeric columns (those containing scores)
    score_columns = [col for col in df.columns if "(Score)" in col]

    # Convert score columns to float explicitly
    for col in score_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # df = df.infer_objects()  # Converts obvious numeric types
    # df = df.apply(pd.to_numeric, errors="ignore")  # Keeps text intact

    # ✅ Save to file
    # Save to file
    if file_format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif file_format.lower() == "excel":
        # Use ExcelWriter with openpyxl for more control
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Crosstab", float_format="%.3f")
            # Get the workbook and worksheet objects
            workbook = writer.book
            worksheet = writer.sheets["Crosstab"]

            # Set number format for score columns
            from openpyxl.styles import numbers

            for col_idx, col_name in enumerate(
                df.columns, start=1
            ):  # start=1 for header offset
                if "(Score)" in col_name:
                    for row_idx in range(2, len(df) + 2):  # start=2 for header row
                        cell = worksheet.cell(row=row_idx, column=col_idx)
                        cell.number_format = "0.000"  # 3 decimal places
    else:
        raise ValueError(
            f"❌ Unsupported file format: {file_format}. Use 'csv' or 'excel'."
        )

    logger.info(f"✅ Crosstab saved: {output_path}")
    # if file_format.lower() == "csv":
    #     df.to_csv(output_path, index=False)
    # elif file_format.lower() == "excel":
    #     df.to_excel(output_path, index=False, sheet_name="Crosstab", engine="openpyxl")
    # else:
    #     raise ValueError(
    #         f"❌ Unsupported file format: {file_format}. Use 'csv' or 'excel'."
    #     )

    # logger.info(f"✅ Crosstab saved: {output_path}")
