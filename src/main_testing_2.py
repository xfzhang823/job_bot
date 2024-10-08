from evaluation_optimization.create_mapping_file import (
    load_existing_or_create_new_mapping,
)
from utils.generic_utils import read_from_json_file
from config import (
    ITERATE_0_DIR,
    mapping_file_name,
    job_descriptions_json_file,
    FLAT_REQS_FILES_ITERATE_0_DIR,
    FLAT_RESPS_FILES_ITERATE_0_DIR,
    SIMILARITY_METRICS_ITERATE_0_DIR,
)
from pathlib import Path

# Allow nested event loops (needed for Jupyter/IPython environments)
mapping_file = ITERATE_0_DIR / mapping_file_name
print(mapping_file)

job_descriptions = read_from_json_file(job_descriptions_json_file)
file_mapping = load_existing_or_create_new_mapping(
    mapping_file_path=mapping_file,
    job_descriptions=job_descriptions,
    reqs_output_dir=FLAT_REQS_FILES_ITERATE_0_DIR,
    resps_output_dir=FLAT_REQS_FILES_ITERATE_0_DIR,
    metrics_output_dir=SIMILARITY_METRICS_ITERATE_0_DIR,
)
