from transformers.utils import logging
from transformers import AutoModel, AutoTokenizer
import os

# Enable debug logging
logging.get_logger("transformers").setLevel("DEBUG")

# ✅ Explicitly print current Hugging Face cache settings
print("HF_HOME:", os.environ.get("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
print("HF_HUB_CACHE:", os.environ.get("HF_HUB_CACHE"))

# ✅ Load a model and check where it's caching
model_name = "bert-base-uncased"

model = AutoModel.from_pretrained(
    model_name, cache_dir=os.environ.get("TRANSFORMERS_CACHE")
)
tokenizer = AutoTokenizer.from_pretrained(
    model_name, cache_dir=os.environ.get("TRANSFORMERS_CACHE")
)

print("✅ Model loaded successfully.")

import os
from transformers.utils import is_offline_mode

print(f"TRANSFORMERS_OFFLINE: {os.environ.get('TRANSFORMERS_OFFLINE')}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print(f"Offline mode status: {is_offline_mode()}")


def get_underlined_cells(sheet):
    """Extract values from underlined cells while keeping the first two columns as is."""
    table_range = sheet.range("A1").expand("table")

    underlined_cells = []

    for i in range(1, table_range.rows.count + 1):

        underlined_row = []

        for j in range(1, table_range.columns.count + 1):

            cell = table_range.api.Cells(i, j)

            if j <= 2:  # Keep the first 2 columns as is

                underlined_row.append(cell.Value)

            elif cell.Font.Underline != -4142:  # -4142 represents no underline

                underlined_row.append(cell.Value)

            else:

                underlined_row.append(None)  # or "empty"

        underlined_cells.append(underlined_row)

    return pd.DataFrame(underlined_cells).set_index(0)


def clean_and_remove_rows(df):
    """Remove None values and reorganize the dataframe."""

    max_length = df.apply(lambda row: row.dropna().shape[0], axis=1).max()

    new_df = pd.DataFrame(index=df.index)

    for i in range(max_length):

        new_df[i] = df.apply(lambda row: get_ith_non_nan(row, i), axis=1)

    return new_df.iloc[2:]  # Remove the first two rows


def get_ith_non_nan(row, i):
    """Retrieve the ith non-NaN value from a row."""

    non_nan = row.dropna()

    return non_nan.iloc[i] if i < len(non_nan) else np.nan


def rename_dataframe_columns(df):
    """Rename index and columns dynamically based on the number of columns."""

    df.index.name = "responsibility_key"

    column_names = ["original_responsibility"] + [
        f"edited_responsibility_{i}" for i in range(1, df.shape[1])
    ]

    df.columns = column_names[: df.shape[1]]
    return df


def json_to_docx(json_data: dict, output_file: Path | str):

    doc = Document()

    if isinstance(json_data, str):

        data = json.loads(json_data)

    else:

        data = json_data

    for main_key, sub_dict in data.items():

        for key, value in sub_dict.items():

            doc.add_paragraph(f"{key}:")

            doc.add_paragraph(value)

            doc.add_paragraph("")  # Add blank line

    if isinstance(output_file, Path):

        doc.save(str(output_file))

    else:

        doc.save(output_file)
