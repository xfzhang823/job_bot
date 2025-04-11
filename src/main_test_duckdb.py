from pathlib import Path
import json
import pandas as pd
from db_io.duckdb_adapter import insert_df_to_table  # assume this exists


def flatten_responsibilities_json(json_path: Path) -> pd.DataFrame:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    url = data["url"]
    responsibilities = data.get("responsibilities", {})

    rows = []
    for key1, section in responsibilities.items():
        opt_by_reqs = section.get("optimized_by_requirements", {})
        for key2, detail in opt_by_reqs.items():
            rows.append(
                {
                    "url": url,
                    "group_key": key2,
                    "optimized_text": detail.get("optimized_text", ""),
                }
            )
    return pd.DataFrame(rows)


df = flatten_responsibilities_json(
    Path(r"C:\github\job_bot\pipeline_data\json\test\sample.json")
)
insert_df_to_table(df, table="test_responsibilities")
