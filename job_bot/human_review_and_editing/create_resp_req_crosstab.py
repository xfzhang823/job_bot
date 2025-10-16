"""resps_vs_reqs_viewer.py"""

from pathlib import Path
import logging
from typing import Dict, List
import pandas as pd
import uuid  # Generate unique test file names

# Setup logger
logger = logging.getLogger(__name__)


def create_core_resp_req_crosstab(
    file_path: Path | str,
    score_threshold: float = 0.0,
) -> pd.DataFrame:
    """
    Creates a responsibility-requirement crosstab DataFrame without
    the 'Original Responsibility' column.

    This function reads a CSV file containing responsibility and requirement data, constructs
    a crosstab where rows represent responsibilities and columns represent requirements, and
    includes similarity scores between them. The first row of the DataFrame contains
    the `requirement_key` values, and subsequent rows contain the responsibility data and scores.
    The column headers are the requirement texts and their corresponding score columns.

    Parameters
    ----------
    file_path : Path | str
        The path to the input CSV file. The CSV must contain the following columns:
        - 'responsibility_key': Unique identifier for each responsibility.
        - 'responsibility': The responsibility text.
        - 'requirement_key': Unique identifier for each requirement.
        - 'requirement': The requirement text.
        - 'composite_score': The similarity score between the responsibility and requirement.
    score_threshold : float, optional (default=0.0)
        The minimum composite score threshold for displaying responsibility text in the crosstab.
        If the score is below this threshold, the responsibility text is not shown in the cell.

    Returns
        -------
        pd.DataFrame
            A DataFrame where:
            - The column headers are 'Resp Key / Req Key' followed by pairs of requirement texts
            and their '(Score)' columns (e.g., 'Requirement Text', 'Requirement Text (Score)').
            - The first row contains the `requirement_key` values (e.g., '0.pie_in_the_sky.0') with
            an empty string under 'Resp Key / Req Key'.
            - Subsequent rows correspond to each `responsibility_key`, with the responsibility text
            and score for each requirement.
            - If duplicate requirement texts exist, the column names are made unique by appending the
            `requirement_key` in brackets (e.g., 'Requirement Text [req_key]').

    Raises
        ValueError
            - If there is a mismatch between the number of columns in the header and
            the number of values in a row.
            - If the final column lengths in the `result_data` dictionary are not consistent
            (i.e., not all columns have the same number of rows).
        FileNotFoundError
            If the specified `file_path` does not exist.
        KeyError
            If the required columns ('responsibility_key', 'responsibility', 'requirement_key',
            'requirement', 'composite_score') are not present in the CSV file.
    """
    # Load and clean data
    needed_cols = [
        "responsibility_key",
        "responsibility",
        "requirement_key",
        "requirement",
        "composite_score",
    ]
    df = pd.read_csv(file_path, usecols=needed_cols)
    logger.debug(f"Loaded CSV: shape={df.shape}, columns={df.columns}")
    df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce")
    df = df.dropna(
        subset=["responsibility_key", "requirement_key"]
    )  # Ensure valid keys

    # Get unique keys
    resp_keys = sorted(df["responsibility_key"].unique())
    req_keys = sorted(df["requirement_key"].unique())
    logger.debug(f"resp_keys ({len(resp_keys)}): {resp_keys}")
    logger.debug(f"req_keys ({len(req_keys)}): {req_keys}")

    # Map requirement_key to requirement text
    req_text_dict = (
        df.drop_duplicates("requirement_key")
        .set_index("requirement_key")["requirement"]
        .to_dict()
    )
    logger.debug(f"req_text_dict: {req_text_dict}")

    # Check for duplicate requirement texts
    req_texts = list(req_text_dict.values())
    if len(req_texts) != len(set(req_texts)):
        logger.warning(
            "⚠️ Duplicate requirement texts found. This may cause column conflicts."
        )
        from collections import Counter

        duplicates = {
            text: count for text, count in Counter(req_texts).items() if count > 1
        }
        logger.debug(f"Duplicate requirement texts: {duplicates}")

    # Define column headers using the full requirement text
    header = ["Resp Key / Req Key"]
    req_key_to_column = {}
    used_column_names = set(header)
    for req_key in req_keys:
        base_text = req_text_dict[req_key]
        # Ensure the column name is unique by appending the req_key if necessary
        text_column = base_text
        score_column = f"{base_text} (Score)"
        while text_column in used_column_names:
            text_column = f"{base_text} [{req_key}]"
        while score_column in used_column_names:
            score_column = f"{base_text} (Score) [{req_key}]"
        header.extend([text_column, score_column])
        used_column_names.add(text_column)
        used_column_names.add(score_column)
        req_key_to_column[req_key] = (text_column, score_column)

    logger.debug(f"Header ({len(header)} columns): {header}")

    # Initialize result_data with the header as column names
    result_data = {col: [] for col in header}

    # First row: requirement_key values
    result_data["Resp Key / Req Key"].append("")
    for req_key in req_keys:
        text_column, score_column = req_key_to_column[req_key]
        result_data[text_column].append(req_key)  # Use the full requirement_key
        result_data[score_column].append("")
    logger.debug(
        f"After first row (req_keys), column lengths: { {col: len(vals) for col, vals in result_data.items()} }"
    )

    # Data rows
    for resp_key in resp_keys:
        row_data = [resp_key]
        for req_key in req_keys:
            entry = df[
                (df["responsibility_key"] == resp_key)
                & (df["requirement_key"] == req_key)
            ]
            responsibility = entry["responsibility"].iloc[0] if not entry.empty else ""
            score = (
                entry["composite_score"].iloc[0] if not entry.empty else float("nan")
            )
            display_text_responsibility = (
                responsibility if pd.notna(score) and score >= score_threshold else ""
            )
            display_text_score = f"{score:.3f}" if pd.notna(score) else ""
            row_data.extend([display_text_responsibility, display_text_score])

        if len(row_data) != len(header):
            logger.error(
                f"❌ Length mismatch: resp_key={resp_key}, header={len(header)}, row_data={len(row_data)}"
            )
            logger.debug(f"row_data: {row_data}")
            raise ValueError("❌ Mismatch between row_data and header length")

        for col, value in zip(header, row_data):
            result_data[col].append(value)
        logger.debug(
            f"After adding row for {resp_key}, column lengths: { {col: len(vals) for col, vals in result_data.items()} }"
        )

    # Final validation
    column_lengths = {col: len(vals) for col, vals in result_data.items()}
    if len(set(column_lengths.values())) != 1:
        logger.error(f"❌ Final column length mismatch: {column_lengths}")
        logger.debug(
            f"result_data sample: { {col: vals[:5] for col, vals in result_data.items()} }"
        )
        raise ValueError("❌ Final column length mismatch in result_data")

    # Create the DataFrame with the header as column names
    df_result = pd.DataFrame(result_data)
    logger.debug(f"DataFrame shape: {df_result.shape}")
    logger.debug(
        f"First few rows of 'Resp Key / Req Key': {df_result['Resp Key / Req Key'].head().tolist()}"
    )

    return df_result


def insert_original_responsibility(
    df: pd.DataFrame,
    original_resps_dict: dict[str, str] | None,
) -> pd.DataFrame:
    """
    Inserts the 'Original Responsibility' column into an existing responsibility-requirement
    crosstab DataFrame.

    This function adds a new column named 'Original Responsibility' immediately after
    the 'Resp Key / Req Key' column.

    The values in this column are sourced from the `original_resps_dict`, which maps
    `responsibility_key` values to their original responsibility texts. The first row
    (corresponding to the `requirement_key` row) is left empty in this column.

    Parameters
        - df: pd.DataFrame
            The input DataFrame, typically the output of `create_core_resp_req_crosstab`.
            It must have a 'Resp Key / Req Key' column containing `responsibility_key` values
            starting  from the second row. The first row is assumed to contain `requirement_key`
            values.
        - original_resps_dict : dict[str, str] | None
            A dictionary mapping `responsibility_key` values to their original responsibility texts.
            If None, the DataFrame is returned unchanged.

    Returns
        pd.DataFrame
            The modified DataFrame with the 'Original Responsibility' column inserted after
            'Resp Key / Req Key'.
            The new column has an empty string in the first row
            (corresponding to the `requirement_key` row),
            followed by the original responsibility texts for each `responsibility_key`.

    Raises
        ValueError
            - If the length of the `original_resp_values` list does not match the number of rows
            in the DataFrame.
            - If the column lengths in the DataFrame are not consistent after inserting the new column.
        KeyError
            If the 'Resp Key / Req Key' column is not present in the DataFrame.
    """
    if original_resps_dict is None:
        logger.debug("No original_resps_dict provided, returning DataFrame unchanged.")
        return df

    # Create a copy of the DataFrame
    result_df = df.copy()
    logger.debug(f"DataFrame shape before insertion: {result_df.shape}")

    # Prepare the 'Original Responsibility' column values
    original_resp_values = [""]  # First row (req_key row)
    # For data rows, map responsibility_key to original responsibility
    data_rows = result_df["Resp Key / Req Key"].iloc[1:]  # Skip the req_key row
    for resp_key in data_rows:
        original_resp = original_resps_dict.get(resp_key, "")
        original_resp_values.append(original_resp)

    # Validate the length of original_resp_values matches the DataFrame's number of rows
    if len(original_resp_values) != len(result_df):
        logger.error(
            f"❌ Length mismatch: DataFrame has {len(result_df)} rows, but original_resp_values has {len(original_resp_values)} values"
        )
        logger.debug(f"original_resp_values: {original_resp_values}")
        raise ValueError(
            "❌ Length mismatch between DataFrame rows and original_resp_values"
        )

    # Insert the 'Original Responsibility' column after 'Resp Key / Req Key'
    result_df.insert(1, "Original Responsibility", original_resp_values)
    logger.debug(
        f"Inserted 'Original Responsibility' column. New columns: {result_df.columns.tolist()}"
    )
    logger.debug(f"DataFrame shape after insertion: {result_df.shape}")

    # Validate column lengths
    column_lengths = {col: len(result_df[col]) for col in result_df.columns}
    if len(set(column_lengths.values())) != 1:
        logger.error(
            f"❌ Column length mismatch after inserting 'Original Responsibility': {column_lengths}"
        )
        raise ValueError(
            "❌ Column length mismatch after inserting 'Original Responsibility'"
        )

    return result_df


def create_resp_req_crosstab(
    file_path: Path | str,
    score_threshold: float = 0.0,
    original_resps_dict: dict[str, str] | None = None,
) -> pd.DataFrame:
    """
    Creates a responsibility-requirement crosstab DataFrame, optionally including
    the 'Original Responsibility' column.

    This function is a wrapper that first builds a core crosstab using
    `create_core_resp_req_crosstab` and then optionally inserts the
    'Original Responsibility' column using `insert_original_responsibility`.

    The resulting DataFrame has responsibilities as rows and requirements as columns,
    with similarity scores between them. The first row contains the `requirement_key` values,
    and subsequent rows contain the responsibility data and scores.

    Parameters
    ----------
    file_path : Path | str
        The path to the input CSV file. The CSV must contain the following columns:
        - 'responsibility_key': Unique identifier for each responsibility.
        - 'responsibility': The responsibility text.
        - 'requirement_key': Unique identifier for each requirement.
        - 'requirement': The requirement text.
        - 'composite_score': The similarity score between the responsibility and requirement.
    score_threshold : float, optional (default=0.0)
        The minimum composite score threshold for displaying responsibility text in the crosstab.
        If the score is below this threshold, the responsibility text is not shown in the cell.
    original_resps_dict : dict[str, str] | None, optional (default=None)
        A dictionary mapping `responsibility_key` values to their original responsibility texts.
        If provided, an 'Original Responsibility' column is inserted after 'Resp Key / Req Key'.
        If None, the column is not added.

    Returns
        pd.DataFrame
            A DataFrame where:
            - The column headers are 'Resp Key / Req Key', optionally 'Original Responsibility',
            followed by pairs of requirement texts and their '(Score)' columns (e.g., 'Requirement Text',
            'Requirement Text (Score)').
            - The first row contains the `requirement_key` values (e.g., '0.pie_in_the_sky.0') with
            an empty string under 'Resp Key / Req Key' and 'Original Responsibility' (if present).
            - Subsequent rows correspond to each `responsibility_key`, with the responsibility text
            and score for each requirement.
            - If duplicate requirement texts exist, the column names are made unique by appending
            the `requirement_key` in brackets (e.g., 'Requirement Text [req_key]').

    Raises
        ValueError
            - If there is a mismatch between the number of columns in the header and the number of values
            in a row (from `create_core_resp_req_crosstab`).
            - If the final column lengths in the `result_data` dictionary are not consistent (from
            `create_core_resp_req_crosstab` or `insert_original_responsibility`).
            - If the length of the `original_resp_values` list does not match the number of rows in
            the DataFrame
            (from `insert_original_responsibility`).
        FileNotFoundError
            If the specified `file_path` does not exist (from `create_core_resp_req_crosstab`).
        KeyError
            - If the required columns ('responsibility_key', 'responsibility', 'requirement_key',
            'requirement', 'composite_score') are not present in the CSV file
            (from `create_core_resp_req_crosstab`).
            - If the 'Resp Key / Req Key' column is not present in the DataFrame
            (from `insert_original_responsibility`).
    """
    # Step 1: Build the core crosstab
    df = create_core_resp_req_crosstab(file_path, score_threshold)
    logger.debug(f"Core crosstab created with shape: {df.shape}")

    # Step 2: Insert the 'Original Responsibility' column if provided
    df = insert_original_responsibility(df, original_resps_dict)
    logger.debug(f"Final crosstab shape: {df.shape}")

    return df


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


# def create_resp_req_crosstab(
#     file_path: Path | str,
#     score_threshold: float = 0.0,
#     original_resps_dict: dict[str, str] | None = None,
# ):
#     """
#     Creates a cross-tabulation where:
#     - Shows full text for scores >= threshold
#     - Shows only the score for scores < threshold
#     - Optionally, includes an extra column with the original responsibility text
#       if an original_resp_mapping is provided.

#     Args:
#         - file_path: Path to the CSV file.
#         - score_threshold: Score threshold for showing full responsibility text.
#         - original_resp_mapping: Optional dictionary mapping responsibility keys to
#         their original responsibility text.

#     Returns:
#         A pandas DataFrame containing the cross-tabulation.
#     """
#     # Read necessary columns
#     needed_cols = [
#         "responsibility_key",
#         "responsibility",
#         "requirement_key",
#         "requirement",
#         "composite_score",
#     ]
#     df = pd.read_csv(file_path, usecols=needed_cols)
#     # df["composite_score"] = pd.to_numeric(df["composite_score"], errors="coerce")

#     # ✅ Log shape and column details
#     logger.info(f"✅ DataFrame Loaded: Shape {df.shape}")
#     logger.info(f"📝 Column Names: {df.columns.tolist()}")

#     # ✅ Log row count for each column
#     row_counts = df.count()
#     logger.info(f"📊 Row Counts per Column:\n{row_counts}")

#     # ✅ Check if any column has missing values
#     missing_counts = df.isna().sum()
#     logger.info(f"🚨 Missing Value Counts per Column:\n{missing_counts}")

#     # 🚨 If row counts are inconsistent, we have an issue
#     if len(set(row_counts.values)) > 1:
#         raise ValueError(f"🚨 Inconsistent row counts detected in {file_path}")

#     # Get unique keys
#     resp_keys = df["responsibility_key"].unique()
#     req_keys = df["requirement_key"].unique()

#     # Create requirement text dictionary for the header row
#     req_text_dict = (
#         df.drop_duplicates("requirement_key")
#         .set_index("requirement_key")["requirement"]
#         .to_dict()
#     )

#     # Create responsibility text and score matrix
#     resp_matrix = {}
#     for resp_key in resp_keys:
#         resp_matrix[resp_key] = {}
#         resp_df = df[df["responsibility_key"] == resp_key]
#         for _, row in resp_df.iterrows():
#             resp_matrix[resp_key][row["requirement_key"]] = {
#                 "responsibility": row["responsibility"],
#                 "score": row["composite_score"],
#             }

#     # Create the output DataFrame
#     # Start with a column for responsibility key
#     header = ["Resp Key / Req Key"]

#     # Optionally include a header for original responsibility
#     if original_resps_dict is not None:
#         header.append("Original Responsibility")

#     # result_data = {"Resp Key / Req Key": ["Requirements"]}

#     # Generate alternating column names for each requirement: text and score columns
#     # (Requirement 1, Score 1, Requirement 2, Score 2, ...)
#     alternating_columns = []
#     for req_key in req_keys:
#         alternating_columns.append(f"{req_text_dict[req_key]}")
#         alternating_columns.append(f"{req_text_dict[req_key]} (Score)")
#     header.extend(alternating_columns)

#     # Initialize result data with header row
#     result_data = {col: [] for col in header}  # Initialize result data with header row

#     # result_data.update({col: [""] for col in alternating_columns})
#     # Add the header row (for the top of the table, you might decide to leave it blank or
#     # descriptive)

#     # Add an empty header row for subsequent rows to align properly.
#     for col in header:
#         result_data[col].append(
#             col if col != "Resp Key / Req Key" else "Responsibilities"
#         )

#     # Add responsibility rows
#     for resp_key in resp_keys:
#         row_data = [resp_key]  # First column is the responsibility key

#         # If original mapping provided, add original responsibility text for the key
#         if original_resps_dict is not None:
#             orig_text = original_resps_dict.get(resp_key, "")
#             row_data.append(orig_text)

#         # Process each requirement column pair
#         for req_key in req_keys:
#             entry = resp_matrix[resp_key].get(req_key, {})
#             responsibility = entry.get("responsibility", "")
#             score = entry.get("score", "")

#             # Determine what to display based on score
#             if score != "" and score >= score_threshold:
#                 display_text_responsibility = responsibility
#             else:
#                 display_text_responsibility = ""

#             # Formatting (set type to float)
#             try:
#                 numeric_score = float(score)
#             except (TypeError, ValueError):
#                 numeric_score = ""
#             display_text_score = f"{numeric_score:.3f}" if numeric_score != "" else ""

#             # display_text_score = f"{score:.3f}" if score != "" else ""

#             # Append both responsibility and score to interleave
#             row_data.append(display_text_responsibility)
#             row_data.append(display_text_score)

#         # Debug logging: lengths of header vs. row_data
#         if len(row_data) != len(header):
#             logger.error(
#                 f"❌ Length mismatch: resp_key={resp_key} | header={len(header)}, row_data={len(row_data)}"
#             )
#             logger.debug(f"Header: {header}")
#             logger.debug(f"Row data: {row_data}")
#             raise ValueError("❌ Mismatch between row_data and header length")

#         for col, value in zip(header, row_data):
#             result_data[col].append(value)

#         # Append the row data into result_data for each corresponding column
#         for col, value in zip(header, row_data):
#             result_data[col].append(value)

#         # for col, value in zip(["Resp Key / Req Key"] + alternating_columns, row_data):
#         #     result_data[col].append(value)

#     # Final consistency check (log column lengths explicitly)
#     column_lengths = {col: len(vals) for col, vals in result_data.items()}
#     if len(set(column_lengths.values())) != 1:
#         logger.error(f"🚨 Final Column length mismatch: {column_lengths}")
#         raise ValueError("❌ Final column length mismatch in result_data")

#     # Create final DataFrame
#     result_df = pd.DataFrame(result_data)

#     return result_df
