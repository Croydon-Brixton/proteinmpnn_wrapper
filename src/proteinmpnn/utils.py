import os
from logging import getLogger
from typing import List

import biotite.structure as bs
import numpy as np
import logging
import logging.handlers
import os
import sys

from .constants import DEFAULT_LOG_FORMATTER, DEFAULT_LOG_LEVEL

logger = getLogger(__name__)


def atom_array_from_numpy(coords: np.ndarray, ca_only: bool = True) -> bs.AtomArray:
    """Convert a numpy array of coordinates to a biotite AtomArray, inserting ALA residues as dummies.
    Useful for visualising pure coordinate backbones in PyMOL.

    Args:
        coords (np.ndarray): A numpy array of shape (n_atoms, 3) containing the coordinates.
        ca_only (bool, optional): Whether each coordinate is a CA atom. If False, it is assumed that
            the coordinates correspond to the backbone atoms N, CA, C, O in this order. Defaults to True.
    """
    n_atoms = coords.shape[0]
    atom_array = bs.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.chain_id = ["A"] * n_atoms
    atom_array.res_name = ["ALA"] * n_atoms
    if ca_only:
        atom_array.atom_name = ["CA"] * n_atoms
        atom_array.element = ["C"] * n_atoms
        atom_array.res_id = np.arange(1, n_atoms + 1)
    else:
        assert n_atoms % 4 == 0, "Number of atoms must be a multiple of 4 if ca_only is False."
        atom_array.atom_name = ["N", "CA", "C", "O"] * int(n_atoms / 4)
        atom_array.element = ["N", "C", "C", "O"] * int(n_atoms / 4)
        atom_array.res_id = np.repeat(np.arange(1, int(n_atoms / 4) + 1), 4)
    return atom_array


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
    return files


def _get_console_handler(
    stream=sys.stdout,
    formatter: logging.Formatter = DEFAULT_LOG_FORMATTER,
) -> logging.StreamHandler:
    """Returns Handler that prints to stdout."""
    console_handler = logging.StreamHandler(stream)
    console_handler.setFormatter(formatter)
    return console_handler

def get_logger(
    logger_name: str,
    level: int = DEFAULT_LOG_LEVEL,
    propagate: bool = False,
    log_to_console: bool = True,
    **handler_kwargs,
) -> logging.Logger:
    """Returns logger with console and timed file handler."""

    logger = logging.getLogger(logger_name)

    # if logger already has handlers attached to it, skip the configuration
    if logger.hasHandlers():
        logger.debug("Logger %s already set up.", logger.name)
        return logger

    logger.setLevel(level)

    if log_to_console:
        logger.addHandler(_get_console_handler(**handler_kwargs))
    
    # with this pattern, it's rarely necessary to propagate the error up to parent
    logger.propagate = propagate

    return logger
