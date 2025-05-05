"""Utils file to find root directory"""

from pathlib import Path


def find_project_root(starting_path=None, marker=".git") -> Path | None:
    """
    Recursively find the root directory of the project by looking for a specific marker.

    Args:
        - starting_path (str or Path): The starting path to begin the search.
        Defaults to the current script's directory.
        - marker (str): The marker to look for (e.g., '.git', 'setup.py', 'README.md').

    Returns:
        Path: The Path object pointing to the root directory of the project,
        or None if not found.
    """
    # Start from the directory of the current file if not specified
    if starting_path is None:
        starting_path = Path(__file__).resolve().parent

    # Convert starting_path to a Path object if it's not already
    starting_path = Path(starting_path)

    # Traverse up the directory tree
    for parent in starting_path.parents:
        # Check if the marker exists in the current directory
        if (parent / marker).exists():
            return parent

    return None  # Return None if the marker is not found
