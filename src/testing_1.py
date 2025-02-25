import json
from pathlib import Path


def reorder_json_files(directory: str | Path):
    """
    Reorders all JSON files in the specified directory to ensure 'url' appears first.

    Args:
        directory (str or Path): Directory containing JSON files.

    Returns:
        None
    """
    directory = Path(directory)

    for file_path in directory.glob("*.json"):
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)

        # Ensure the order: 'url' first, then 'responsibilities'
        if "url" in data and "responsibilities" in data:
            reordered_data = {
                "url": data["url"],
                "responsibilities": data["responsibilities"],
            }
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(reordered_data, file, indent=4, ensure_ascii=False)
            print(f"Reordered {file_path}")

    print("All files processed.")


# Example usage
responsibilities_dir = r"C:\github\job_bot\input_output\evaluation_optimization\evaluation_optimization_by_anthropic\iteration_1\responsibilities"  # Replace with actual path
reorder_json_files(responsibilities_dir)  # Replace with actual directory path
