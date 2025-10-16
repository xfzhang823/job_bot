"""
This module provides utility functions for file operations.

Functions:
    get_file_names: Collects and returns file names in a given directory filtered by file types.
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Union, Optional
import hashlib
from urllib.parse import urlparse, parse_qs


logger = logging.getLogger(__name__)


def get_file_names(
    directory_path: Union[str, Path],
    full_path: bool = False,
    recursive: bool = False,
    file_types: Optional[Union[str, List[str]]] = None,
    file_type_inclusive: bool = True,
) -> List[str]:
    """
    Collects and returns file names in a given directory filtered by file types.

    Args:
        directory_path (str or Path): The target directory to list files from.
        full_path (bool): Whether to return the full path of the files.
        file_types (str or list): File extension or list of extensions to filter by
        (e.g., '.txt' or ['.txt', '.csv']).
        recursive (bool): Whether to search directories recursively. Default is False.
        file_type_inclusive (bool): Whether to include or exclude the specified file types.
                                    Default is True (inclusive).

    Returns:
        list: File names or full paths to the files in the directory that match the file types,
        or an empty list if none.

    Example:
        >>> get_file_names("C:/example_dir", full_path=True, file_types=[".txt", ".csv"], recursive=True)
        ['C:/example_dir/file1.txt', 'C:/example_dir/subdir/file2.csv']

        >>> get_file_names("C:/example_dir", file_types=".py", recursive=False, file_type_inclusive=False)
        ['file1.txt', 'file2.csv', 'file3.md']
    """
    if not os.path.isdir(directory_path):
        return []

    if isinstance(file_types, str):
        file_types = [file_types]

    def match_file_type(file_name: str) -> bool:
        if file_types is None:
            return True
        if file_type_inclusive:
            return any(file_name.endswith(ext) for ext in file_types)
        else:
            return not any(file_name.endswith(ext) for ext in file_types)

    matched_files = []
    for root, dirs, files in os.walk(directory_path):
        matched_files.extend(
            [
                (
                    os.path.join(root, f)
                    if full_path
                    else os.path.relpath(os.path.join(root, f), directory_path)
                )
                for f in files
                if match_file_type(f)
            ]
        )
        if not recursive:
            break

    return matched_files


# --- tiny helpers ---


def _slug(s: str, max_len: int) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)  # non-alnum -> -
    s = re.sub(r"-{2,}", "-", s).strip("-")  # collapse dashes
    return s[:max_len] if max_len > 0 else s


def _extract_job_id(url: str) -> str | None:
    """
    Prefer explicit query param like job_id=123..., else any 6+ digit run.
    """
    qs = parse_qs(urlparse(url).query)
    if "job_id" in qs and qs["job_id"]:
        return re.sub(r"\D+", "", qs["job_id"][0]) or None
    m = re.search(r"(?<!\d)(\d{6,})(?!\d)", url)  # 6+ consecutive digits
    return m.group(1) if m else None


def _short_hash(url: str, n: int = 6) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()[:n]
