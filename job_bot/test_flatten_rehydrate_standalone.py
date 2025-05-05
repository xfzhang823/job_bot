"""
Test: flatten → table → rehydrate for JobPostings and ExtractedRequirements

Run this as a standalone script:
    python test_flatten_rehydrate_standalone.py
"""

import json
import pandas as pd
from pathlib import Path

from db_io.flatten_and_rehydrate import (
    flatten_job_postings_to_table,
    rehydrate_job_postings_from_table,
    flatten_extracted_requirements_to_table,
    rehydrate_extracted_requirements_from_table,
)

# === INPUT FILES (adjust if needed) ===
job_postings_path = Path(
    "C:/github/job_bot/input_output/preprocessing/jobpostings.json"
)
requirements_path = Path(
    "C:/github/job_bot/input_output/preprocessing/extracted_job_requirements.json"
)


# === LOAD JSON ===
def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


# === TEST: Job Postings ===
def test_job_postings():
    print("\n🧪 Testing job postings...")
    original = load_json(job_postings_path)
    df = flatten_job_postings_to_table(original)
    print(f"✅ Flattened job postings: {len(df)} rows")
    df.to_csv("flattened_job_postings.csv", index=False)

    roundtrip = rehydrate_job_postings_from_table(df)
    print(f"🔁 Rehydrated job postings: {len(roundtrip)} entries")

    missing_keys = set(original.keys()) - set(roundtrip.keys())
    if missing_keys:
        print(
            f"⚠️ {len(missing_keys)} postings dropped during round-trip (likely invalid entries)"
        )
    else:
        print("🎉 All job postings rehydrated successfully")


# === TEST: Extracted Requirements ===
def test_extracted_requirements():
    print("\n🧪 Testing extracted requirements...")
    original = load_json(requirements_path)
    df = flatten_extracted_requirements_to_table(original)
    print(f"✅ Flattened requirements: {len(df)} rows")
    df.to_csv("flattened_extracted_requirements.csv", index=False)

    roundtrip = rehydrate_extracted_requirements_from_table(df)
    print(f"🔁 Rehydrated requirements: {len(roundtrip)} entries")

    missing_keys = set(original.keys()) - set(roundtrip.keys())
    if missing_keys:
        print(
            f"⚠️ {len(missing_keys)} requirement entries dropped (invalid or unparseable)"
        )
    else:
        print("🎉 All requirements rehydrated successfully")


# === MAIN ===
if __name__ == "__main__":
    test_job_postings()
    test_extracted_requirements()
