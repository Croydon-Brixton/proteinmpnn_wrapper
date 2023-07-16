import gzip
import io
import os
import pathlib
from typing import Union

import biotite.structure as bs
import numpy as np
from biotite.structure.io import mmtf, pdb, pdbx

from ..constants import AA_1_TO_3
from .log_utils import get_logger
from .misc import filter_kwargs

logger = get_logger(__name__)

__all__ = ["atom_array_from_numpy", "load_structure"]


def atom_array_from_numpy(
    coords: np.ndarray,
    ca_only: bool = True,
    chain_id: Union[np.ndarray, str] = "A",
    res_name: Union[np.ndarray, str] = "ALA",
) -> bs.AtomArray:
    """Convert a numpy array of coordinates to a biotite AtomArray, inserting ALA residues as dummies.
    Useful for visualising pure coordinate backbones in PyMOL.

    Args:
        coords (np.ndarray): A numpy array of shape (n_atoms, 3) containing the coordinates.
        ca_only (bool, optional): Whether each coordinate is a CA atom. If False, it is assumed that
            the coordinates correspond to the backbone atoms N, CA, C, O in this order. Defaults to True.
        chain_id (Union[np.ndarray, str], optional): The chain ID to assign to the atom array. If a string,
            the same chain ID is assigned to all atoms. If an array, it must be of length n_atoms. Defaults to "A".
        res_name (Union[np.ndarray, str], optional): The residue name to assign to the atom array. If a string,
            the same residue name is assigned to all atoms. If an array, it must be of length n_atoms. Defaults to "ALA".

    Returns:
        atom_array (bs.AtomArray): The atom array representing the backbone protein structure.
    """
    n_atoms = coords.shape[0]
    if ca_only:
        n_residues = n_atoms
    else:
        assert n_atoms % 4 == 0, "Number of atoms must be a multiple of 4 if ca_only is False."
        n_residues = int(n_atoms / 4)

    # Insert chain IDs
    if isinstance(chain_id, str):
        chain_id = np.repeat(chain_id, n_atoms)
    else:
        assert len(chain_id) == n_atoms, "chain_id must be a string or an array of length n_atoms."
        assert isinstance(chain_id[0], str), "chain_id must be a string or an array of strings."

    # Insert AA residue identities
    if isinstance(res_name, str):
        # .. insert the given residue name if it is a string
        if len(res_name) != 3:
            res_name = AA_1_TO_3[res_name]
        res_name = np.repeat(res_name, n_atoms)
    else:
        # ... otherwise, assume it is an array of length n_atoms
        assert len(res_name) == n_atoms, "res_name must be a string or an array of length n_atoms."
        if len(res_name[0]) != 3:
            res_name = np.array([AA_1_TO_3[res] for res in res_name])

    atom_array = bs.AtomArray(n_atoms)
    atom_array.coord = coords
    atom_array.chain_id = chain_id
    atom_array.res_name = res_name
    if ca_only:
        atom_array.atom_name = ["CA"] * n_atoms
        atom_array.element = ["C"] * n_atoms
        atom_array.res_id = np.arange(1, n_atoms + 1)
    else:
        atom_array.atom_name = ["N", "CA", "C", "O"] * n_residues
        atom_array.element = ["N", "C", "C", "O"] * n_residues
        atom_array.res_id = np.repeat(np.arange(1, n_residues + 1), 4)

    return atom_array


def structure_from_buffer(buffer: Union[io.StringIO, io.BytesIO], file_type: str, **load_kwargs) -> bs.AtomArray:
    buffer.seek(0)
    if file_type in ("cif", "mmcif", "pdbx"):
        file = pdbx.PDBxFile()
        file.read(buffer)
        if "assembly_id" in load_kwargs:
            load_kwargs = filter_kwargs(load_kwargs, pdbx.get_assembly)
            return pdbx.get_assembly(file, **load_kwargs)
        load_kwargs = filter_kwargs(load_kwargs, pdbx.get_structure)
        return pdbx.get_structure(file, **load_kwargs)
    elif file_type in ("pdb", "pdb1"):
        file = pdb.PDBFile()
        file.read(buffer)
        load_kwargs = filter_kwargs(load_kwargs, pdb.get_structure)
        return pdb.get_structure(file, **load_kwargs)
    elif file_type == "mmtf":
        file = mmtf.MMTFFile()
        file.read(buffer)
        load_kwargs = filter_kwargs(load_kwargs, mmtf.get_structure)
        return mmtf.get_structure(file, **load_kwargs)
    else:
        raise ValueError(f"Unknown type: {file_type}")


def load_structure(path_or_buffer: Union[io.StringIO, io.BytesIO, pathlib.Path, str], **load_kwargs) -> bs.AtomArray:
    if isinstance(path_or_buffer, io.StringIO):
        # ... if loading from string buffer
        assert "file_type" in load_kwargs, "Type must be specified when loading from buffer"
        return structure_from_buffer(path_or_buffer, **load_kwargs)
    elif isinstance(path_or_buffer, io.BytesIO):
        # ... if loading from byts buffer
        load_kwargs["file_type"] = "mmtf"
        return structure_from_buffer(path_or_buffer, **load_kwargs)
    else:
        # ... if loading from path
        path = pathlib.Path(path_or_buffer)
        assert path.exists(), f"File does not exist: {path}"

        if path.suffix in (".gz", ".gzip"):
            open_func = gzip.open
            file_type = os.path.splitext(path.stem)[-1].split(".")[-1]
        else:
            open_func = open
            file_type = path.suffix.split(".")[-1]

        buffer = io.BytesIO() if file_type == "mmtf" else io.StringIO()
        mode = "rb" if file_type == "mmtf" else "rt"
        with open_func(path, mode) as file:
            buffer.write(file.read())
        load_kwargs["file_type"] = file_type
        return structure_from_buffer(buffer, **load_kwargs)
