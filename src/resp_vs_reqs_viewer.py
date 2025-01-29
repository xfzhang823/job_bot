import pandas as pd
import numpy as np


def create_resp_req_crosstab(file_path, score_threshold=0.3):
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
    # First, create the requirements row
    result_data = {"Resp Key / Req Key": ["Requirements"]}
    for req_key in req_keys:
        result_data[req_key] = [req_text_dict[req_key]]

    # Add responsibility rows
    for resp_key in resp_keys:
        row_data = [resp_key]  # First column is the responsibility key
        for req_key in req_keys:
            entry = resp_matrix[resp_key].get(req_key, {})
            score = entry.get("score", "")

            # Determine what to display based on score
            if score and score >= score_threshold:
                # Show full text for high scores
                display_text = f"{entry.get('responsibility', '')} (Score: {score:.3f})"
            else:
                # Show only score for low scores
                display_text = f"Score: {score:.3f}" if score != "" else ""

            row_data.append(display_text)

        # Add the row to result_data
        for col, value in zip(["Resp Key / Req Key"] + list(req_keys), row_data):
            result_data[col].append(value)

    # Create final DataFrame
    result_df = pd.DataFrame(result_data)

    return result_df


def save_crosstab(df, output_path, format="csv"):
    """
    Saves the cross-tabulation to a file.
    Supports CSV and Excel formats.
    """
    if format.lower() == "csv":
        df.to_csv(output_path, index=False)
    elif format.lower() == "excel":
        # Configure Excel writer to adjust column widths
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Crosstab")
            worksheet = writer.sheets["Crosstab"]

            # Adjust column widths
            for idx, col in enumerate(df.columns):
                max_length = max(df[col].astype(str).apply(len).max(), len(str(col)))
                # Limit max width to avoid extremely wide columns
                adjusted_width = min(max_length + 2, 100)
                worksheet.column_dimensions[chr(65 + idx)].width = adjusted_width


# Example usage
def main():
    input_file = r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_claude\iteration_1\similarity_metrics\Microsoft_Head_of_Partner_Intelligence_and_Strategy_sim_metrics_iter1.csv"
    output_file = r"C:\github\job_bot\data\resp_vs_reqs_crosstab_output_filtered.csv"  # Output file (can be .csv or .xlsx)
    score_threshold = 0.3  # Set score threshold

    try:
        # Create the cross-tabulation
        result_df = create_resp_req_crosstab(
            input_file, score_threshold=score_threshold
        )

        # Save to CSV by default
        save_crosstab(result_df, output_file, format="csv")
        print(f"Cross-tabulation saved successfully to {output_file}")
        print(
            f"Showing full text for scores >= {score_threshold}, only scores for others"
        )

        # Display first few rows in console
        print("\nPreview of the cross-tabulation:")
        pd.set_option("display.max_columns", 4)  # Limit columns for display
        pd.set_option("display.width", 1000)
        print(result_df.iloc[:2, :4])  # Show first 2 rows and 4 columns

    except Exception as e:
        print(f"An error occurred: {str(e)}")


# Example usage
if __name__ == "__main__":
    main()
