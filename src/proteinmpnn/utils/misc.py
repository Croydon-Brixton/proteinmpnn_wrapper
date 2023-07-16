import os
from typing import List

from .log_utils import get_logger

logger = get_logger(__name__)

__all__ = ["find_files"]


def find_files(path: str, ext: str, depth: int = 3) -> List[str]:
    """Find files up to `depth` levels down that match the given file extension.

    Args:
        path (str): The starting directory path.
        ext (str): The file extension to match.
        depth (int, optional): Maximum number of subdirectory levels to search. Defaults to 3.

    Returns:
        list: A list of file paths that match the file extension.
    """
    path = str(path)
    if depth < 0:
        logger.error("Depth cannot be negative.")
        return []
    elif not os.path.isdir(path):
        logger.error(f"Path {path} does not exist.")
        return []

    files = []
    root_depth = path.rstrip(os.path.sep).count(os.path.sep)
    for dirpath, dirs, filenames in os.walk(path):
        current_depth = dirpath.count(os.path.sep)
        if current_depth - root_depth <= depth:
            for filename in filenames:
                if filename.endswith(ext):
                    files.append(os.path.join(dirpath, filename))
        if current_depth >= root_depth + depth:
            # Modify dirs in-place to limit os.walk's recursion
            dirs.clear()

    logger.info(f"Found {len(files)} files with extension {ext} in {path}.")
    return sorted(files)
