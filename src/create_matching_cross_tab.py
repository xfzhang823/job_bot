import pandas as pd
import xlwings as xw


def create_pivot_table(sim_metrics_csv, output_excel):
    """
    Reads the CSV and creates a pivot table:
      - Index: responsibility_key
      - Columns: (requirement_key, requirement)
      - Values: responsibility, composite_score
    Then saves it as an Excel file and applies conditional formatting using xlwings.
    """
    # Load CSV file
    df = pd.read_csv(sim_metrics_csv)

    # Format responsibility text based on composite_score
    def format_responsibility(row):
        if pd.isna(row["composite_score"]):  # Handle NaN values
            return row["responsibility"]
        elif row["composite_score"] >= 0.75:
            return f"{row['responsibility']}"  # Highlight important ones
        elif row["composite_score"] < 0.3:
            return f"{row['responsibility']}"  # Mark low ones
        return row["responsibility"]

    df["formatted_responsibility"] = df.apply(format_responsibility, axis=1)

    # Create pivot table
    pivot_table = df.pivot_table(
        index="responsibility_key",
        columns=["requirement_key", "requirement"],
        values=["formatted_responsibility", "composite_score"],
        aggfunc="first",
    )

    pivot_table = pivot_table.fillna("")
    pivot_table.to_excel(output_excel)

    # Apply Conditional Formatting with xlwings
    apply_xlwings_formatting(output_excel)
    print(f"Pivot table saved and formatted at: {output_excel}")


def apply_xlwings_formatting(excel_file):
    """Applies conditional formatting to value cells (not headers) based on their composite_score."""
    app = xw.App(visible=True)  # Keep Excel open for debugging
    wb = xw.Book(excel_file)
    ws = wb.sheets[0]

    # Detect last row and last column
    last_row = ws.range("A1").expand("down").last_cell.row
    last_col = ws.range("A1").expand("right").last_cell.column

    # Iterate through all data cells (excluding headers)
    for row in range(2, last_row + 1):  # Start from row 2 to avoid header
        for col in range(2, last_col + 1):  # Start from col 2 to avoid row labels
            cell = ws.cells(row, col)
            try:
                value = float(cell.value)  # Convert value to float
                if value >= 0.75:
                    cell.api.Interior.Color = xw.utils.rgb_to_int(
                        (0, 255, 0)
                    )  # Green for high scores
                elif value < 0.3:
                    cell.api.Interior.Color = xw.utils.rgb_to_int(
                        (255, 0, 0)
                    )  # Red for low scores
                else:
                    cell.api.Interior.Color = xw.utils.rgb_to_int(
                        (255, 255, 0)
                    )  # Yellow for mid-range scores
            except (ValueError, TypeError):
                pass  # Ignore non-numeric values

    wb.save()
    wb.close()
    app.quit()
    print("✅ Conditional formatting applied to value cells!")


def main():
    input_csv = r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_openai\iteration_1\similarity_metrics\Microsoft_Head_of_Partner_Intelligence_and_Strategy_sim_metrics_iter1.csv"
    output_excel = (
        r"C:\github\job_bot\data\matching_examples\resp_vs_reqs_pivot_output_1.xlsx"
    )

    create_pivot_table(input_csv, output_excel)


if __name__ == "__main__":
    main()
