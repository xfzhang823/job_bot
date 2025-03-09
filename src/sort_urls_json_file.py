from pathlib import Path
import json
from project_config import JOB_POSTING_URLS_TO_EXCLUDE_FILE


def sort_json_by_company(json_file_path):
    # Step 1: Read the JSON file
    with open(json_file_path, "r") as file:
        data = json.load(file)

    print(f"JSON data: {data}")

    # Step 2: Convert the dictionary to a list of (key, value) pairs
    items = list(data.items())

    print(f"List data: {items}")

    # Step 3: Sort the list by the "company" field in the value
    # Note: Using .lower() to ensure case-insensitive sorting
    items.sort(key=lambda x: x[1]["company"].lower())

    print(f"items sorted: {items}")

    # Step 4: Create a new dictionary in sorted order
    sorted_data = {key: value for key, value in items}

    print(f"items data: {sorted_data}")

    # Step 5: Write the sorted dictionary back to the same file
    with open(json_file_path, "w") as file:
        json.dump(sorted_data, file, indent=2)


if __name__ == "__main__":
    # Replace 'jobs.json' with the path to your JSON file
    sort_json_by_company(JOB_POSTING_URLS_TO_EXCLUDE_FILE)
