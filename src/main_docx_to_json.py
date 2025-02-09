from pathlib import Path
import json
from resume_docx_json_conversion.doxc_to_json_converter import DocxJsonProcessor
from utils.generic_utils import (
    pretty_print_json,
    save_to_json_file,
    read_from_json_file,
)
from project_config import RESUME_DOCX_FILE, resume_json_file_temp


def run_pipe_line():
    doc_path = Path(RESUME_DOCX_FILE)
    processor = DocxJsonProcessor(doc_path)

    # Convert DOCX to JSON
    json_content = processor.extract_to_json()

    pretty_print_json(json_content)

    # Save JSON content to a file (optional)
    with open(resume_json_file_temp, "w") as json_file:
        json.dump(json_content, json_file, indent=4)

    # # Optionally modify the JSON content...

    # Convert JSON back to DOCX
    testing_file = "C:\github\job_bot\input_output\input\Resume Xiao-Fei Zhang 2024_Mkt_Intel_testing.docx"
    processor.json_to_docx(json_content, testing_file)


if __name__ == "__main__":
    run_pipe_line()
