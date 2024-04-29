from __future__ import annotations

import io
import pathlib
from typing import Literal, Optional, Union

import biotite.structure as bs
import numpy as np
import torch

from .constants import AA_1_TO_3, AA_3_TO_1, PROTEIN_MPNN_ALPHABET
from .utils.biotite_utils import load_structure
from .utils.misc import filter_kwargs

_REVERSE_ALPHABET = {v: k for k, v in PROTEIN_MPNN_ALPHABET.items()}


def as_numpy(x: torch.Tensor | np.ndarray | list) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):
        return x
    elif isinstance(x, list):
        return np.asarray(x)
    else:
        raise ValueError("Cannot convert to numpy array.")


def tokenise_sequence(seq: str, device: str) -> torch.Tensor:
    return torch.tensor([PROTEIN_MPNN_ALPHABET[aa] for aa in seq], dtype=torch.long, device=device)


def untokenise_sequence(x: torch.Tensor) -> str:
    assert x.dtype == torch.long
    if x.ndim == 2 and x.shape[0] > 1:
        return ["".join([_REVERSE_ALPHABET[res.item()] for res in seq]) for seq in x]
    return "".join([_REVERSE_ALPHABET[i.item()] for i in x.squeeze()])


class BackboneSample:
    def __init__(
        self,
        bb_coords: np.ndarray,
        *,
        ca_only: bool = True,
        res_name: Union[str, np.ndarray] = "X",
        res_mask: Optional[np.ndarray] = None,
        res_bias: Optional[np.ndarray] = None,
        res_disallowed_mask: Optional[np.ndarray] = None,
        chain_id: Union[str, np.ndarray] = "A",
        chain_mask: Optional[np.ndarray] = None,
    ):
        # Check inputs
        n_atoms, n_dim = bb_coords.shape
        assert n_dim == 3, "bb_coords must be a numpy array of shape (n_atoms, 3)."
        if ca_only:
            n_residues = n_atoms
        else:
            assert n_atoms % 4 == 0, "Number of atoms must be a multiple of 4 if ca_only is False."
            n_residues = int(n_atoms / 4)

        # Coordinate information
        self.bb_coords = bb_coords
        self.ca_only = ca_only

        # Residue information
        # ... insert AA residue identities
        if isinstance(res_name, str):
            # .. insert the given residue name if it is a string
            if len(res_name) == n_residues:
                res_name = np.array(list(res_name))
            elif len(res_name) == 3:
                res_name = AA_3_TO_1[res_name]
                res_name = np.repeat(res_name, n_residues)
            else:
                assert len(res_name) == 1, "res_name must be a string or an array of length n_residues."
                res_name = np.repeat(res_name, n_residues)
        else:
            # ... otherwise, assume it is an array of length n_atoms
            assert len(res_name) == n_residues, "res_name must be a string or an array of length n_residues."
            if len(res_name[0]) != 1:
                res_name = np.array([AA_3_TO_1[res] for res in res_name])
        # ... create residue mask
        if res_mask is None:
            res_mask = np.ones(n_residues).astype(bool)
        else:
            assert len(res_mask) == n_residues, "res_mask must be an array of length n_residues."
        # ... create residue bias
        if res_bias is None:
            res_bias = np.zeros((n_residues, len(PROTEIN_MPNN_ALPHABET)), dtype=bool)
        else:
            assert res_bias.shape == (
                n_residues,
                len(PROTEIN_MPNN_ALPHABET),
            ), "res_bias must be an array of shape (n_residues, 20)."
        # ... create mask indicating which residues are disallowed
        if res_disallowed_mask is None:
            res_disallowed_mask = np.zeros((n_residues, len(PROTEIN_MPNN_ALPHABET)), dtype=bool)
        else:
            assert res_disallowed_mask.shape == (
                n_residues,
                len(PROTEIN_MPNN_ALPHABET),
            ), "res_disallowed_mask must be an array of shape (n_residues, 20)."
        # ... assign to object
        self.res_name = res_name
        self.res_mask = res_mask
        self.res_bias = res_bias
        self.res_disallowed_mask = res_disallowed_mask
        self.res_id = np.arange(1, n_residues + 1)

        # Chain information
        # ... insert chain IDs
        if isinstance(chain_id, str):
            chain_id = np.repeat(chain_id, n_atoms)
        else:
            assert len(chain_id) == n_atoms, "chain_id must be a string or an array of length n_atoms."
            assert isinstance(chain_id[0], str), "chain_id must be a string or an array of strings."
        # ... create chain mask
        if chain_mask is None:
            chain_mask = np.ones(n_atoms).astype(bool)
        else:
            assert len(chain_mask) == n_atoms, "chain_mask must be an array of length n_atoms."
        # ... assign to object
        self.chain_id = chain_id
        self.chain_mask = chain_mask

    @property
    def n_atoms(self):
        return self.bb_coords.shape[0]

    @property
    def n_residues(self):
        return self.res_name.shape[0]

    def __repr__(self):
        return f"BackboneSample(n_atoms={self.n_atoms}, n_residues={self.n_residues}, ca_only={self.ca_only})"

    @classmethod
    def load_any(
        cls,
        file_or_buffer_or_obj: Union[
            str, pathlib.Path, bs.AtomArray, io.StringIO, io.BytesIO, np.ndarray, torch.Tensor
        ],
        **kwargs,
    ) -> "BackboneSample":
        """Load a protein structure or backbone from a file, buffer, numpy array or PyTorch tensor.

        Args:
            file_or_buffer_or_obj (Union[str, pathlib.Path, bs.AtomArray, io.StringIO, io.BytesIO, np.ndarray, torch.Tensor]):
                The file, buffer, numpy array or PyTorch tensor to load the structure from.
            **kwargs: Keyword arguments
                - `model` (int): If loading from a pdb-type file, the model number to load. Defaults to 1.
                - `ca_only` (bool): If loading from a numpy array/torch Tensor, whether to load only the CA atoms.
                    Defaults to True.
                - any other keyword arguments are passed to `biotite.structure.io.load_structure` or
                    `proteinmpnn.utils.biotite_utils.atom_array_from_numpy`.

        Returns:
            BackboneSample: The loaded backbone sample.
        """
        if isinstance(file_or_buffer_or_obj, (str, pathlib.Path, io.StringIO, io.BytesIO)):
            # Check if `model` in kwargs
            if "model" not in kwargs:
                kwargs["model"] = 1
            return cls.from_biotite(
                load_structure(file_or_buffer_or_obj, **kwargs), **filter_kwargs(kwargs, BackboneSample.from_biotite)
            )
        elif isinstance(file_or_buffer_or_obj, bs.AtomArray):
            return cls.from_biotite(file_or_buffer_or_obj, **kwargs)
        elif isinstance(file_or_buffer_or_obj, np.ndarray):
            assert "ca_only" in kwargs, "Must specify `ca_only` if passing a numpy array."
            return cls(file_or_buffer_or_obj, **kwargs)
        elif isinstance(file_or_buffer_or_obj, torch.Tensor):
            return load_protein_structure(file_or_buffer_or_obj.detach().numpy(), **kwargs)

    @classmethod
    def from_biotite(
        self,
        structure: bs.AtomArray,
        ca_only: bool = False,
        chains: Optional[Union[str, np.ndarray]] = None,
        **init_kwargs,
    ) -> "BackboneSample":
        structure = structure[bs.filter_amino_acids(structure)]
        if isinstance(chains, str):
            structure = structure[structure.chain_id == chains]
        elif isinstance(chains, np.ndarray):
            structure = structure[np.isin(structure.chain_id, chains)]
        backbone = structure[np.isin(structure.atom_name, ["N", "CA", "C", "O"])]

        if ca_only:
            backbone = backbone[backbone.atom_name == "CA"]

        seq = bs.residues.get_residues(backbone)[1]
        seq = "".join([AA_3_TO_1[res] for res in seq])

        return BackboneSample(backbone.coord, res_name=seq, ca_only=ca_only, chain_id=backbone.chain_id, **init_kwargs)

    def to_biotite(self) -> bs.AtomArray:
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
        n_atoms = self.bb_coords.shape[0]
        if self.ca_only:
            n_residues = n_atoms
        else:
            assert n_atoms % 4 == 0, "Number of atoms must be a multiple of 4 if ca_only is False."
            n_residues = int(n_atoms / 4)

        atom_array = bs.AtomArray(n_atoms)
        atom_array.coord = as_numpy(self.bb_coords)
        atom_array.chain_id = as_numpy(self.chain_id)
        atom_array.res_name = [AA_1_TO_3[res] for res in as_numpy(self.res_name)]
        if self.ca_only:
            atom_array.atom_name = ["CA"] * n_atoms
            atom_array.element = ["C"] * n_atoms
        else:
            atom_array.atom_name = ["N", "CA", "C", "O"] * n_residues
            atom_array.element = ["N", "C", "C", "O"] * n_residues
        atom_array.res_id = as_numpy(self.res_id)

        return atom_array

    def to_protein_mpnn_input(self, mode: str = Literal["sampling", "scoring"], device: str = "cuda"):
        if mode == "sampling":
            return self._to_protein_mpnn_sampling_input(device=device)
        elif mode == "scoring":
            return self._to_protein_mpnn_scoring_input(device=device)
        else:
            raise ValueError(f"Invalid mode {mode}. Must be 'sampling' or 'scoring'.")

    @classmethod
    def to_protein_mpnn_batch(
        cls, samples: list[BackboneSample], mode: str = Literal["sampling", "scoring"], device: str = "cuda"
    ):
        assert isinstance(samples[0], cls), "All samples must be instances of BackboneSample."
        samples = [sample.to_protein_mpnn_input(mode=mode) for sample in samples]
        if mode == "sampling":
            return cls._collate_protein_mpnn_sampling_input(samples, device=device)
        elif mode == "scoring":
            return cls._collate_protein_mpnn_scoring_input(samples, device=device)
        else:
            raise ValueError(f"Invalid mode {mode}. Must be 'sampling' or 'scoring'.")

    def _to_protein_mpnn_sampling_input(self, device: str) -> dict:
        chain_names = np.unique(self.chain_id)
        chain_names.sort()
        chain_labels = np.arange(len(chain_names))
        chain_encoding = dict(zip(chain_names, chain_labels))

        # Create remaining inputs with default values (can be overwritten in output dict if desired)
        #  but are needed to run the model
        omit_AAs_np = np.zeros(21)
        omit_AAs_np[-1] = 1.0  # disallow the `mask` token
        bias_AAs_np = np.zeros(21)
        # PSSM (position-specific scoring matrix) bias
        pssm_log_odds_mask = torch.ones(
            (self.n_residues, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device
        )
        pssm_bias = torch.zeros((self.n_residues, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device)
        pssm_coef = torch.zeros((self.n_residues), dtype=torch.float32, device=device)

        return {
            "X": torch.as_tensor(self.bb_coords, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # Backbone coordinates
            "S_true": tokenise_sequence(self.res_name, device=device).unsqueeze(
                0
            ),  # Amino acid sequence as tokenised integers
            "mask": torch.as_tensor(self.res_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if residue is allowed to be sampled
            "omit_AA_mask": torch.as_tensor(self.res_disallowed_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if residue is disallowed
            "chain_mask": torch.as_tensor(self.chain_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if atoms in chain are allowed to be sampled
            "chain_encoding_all": torch.as_tensor(
                [chain_encoding[chain] + 1 for chain in self.chain_id], dtype=torch.long, device=device
            ).unsqueeze(
                0
            ),  # Integer encoding of chain ID for each atom
            "residue_idx": torch.as_tensor(self.res_id - 1, dtype=torch.long, device=device).unsqueeze(
                0
            ),  # Residue indices
            "chain_M_pos": torch.as_tensor(self.chain_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if atoms in chain are allowed to be sampled
            "omit_AAs_np": omit_AAs_np,  # Amino acids that are not allowed to be sampled at a given position
            "bias_AAs_np": bias_AAs_np,  # Amino acids that are preferred to be sampled
            "bias_by_res": torch.as_tensor(self.res_bias, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # Amino acids that are preferred to be sampled (by residue
            "pssm_log_odds_mask": pssm_log_odds_mask.unsqueeze(0),  # PSSM (position-specific scoring matrix) mask
            "pssm_bias": pssm_bias.unsqueeze(0),  # PSSM (position-specific scoring matrix) bias
            "pssm_coef": pssm_coef.unsqueeze(0),  # PSSM (position-specific scoring matrix) coefficient
        }

    def _to_protein_mpnn_scoring_input(self, device: str) -> dict:
        chain_names = np.unique(self.chain_id)
        chain_names.sort()
        chain_labels = np.arange(len(chain_names))
        chain_encoding = dict(zip(chain_names, chain_labels))

        return {
            "X": torch.as_tensor(self.bb_coords, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # Backbone coordinates
            "S": tokenise_sequence(self.res_name, device=device).unsqueeze(
                0
            ),  # Amino acid sequence as tokenised integers
            "mask": torch.as_tensor(self.res_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if residue is allowed to be sampled
            "chain_M": torch.as_tensor(self.chain_mask, dtype=torch.float32, device=device).unsqueeze(
                0
            ),  # True if atoms in chain are allowed to be sampled
            "chain_encoding_all": torch.as_tensor(
                [chain_encoding[chain] + 1 for chain in self.chain_id], dtype=torch.long, device=device
            ).unsqueeze(
                0
            ),  # Integer encoding of chain ID for each atom
            "residue_idx": torch.as_tensor(self.res_id - 1, dtype=torch.long, device=device).unsqueeze(
                0
            ),  # Residue indices
        }

    @classmethod
    def _collate_protein_mpnn_sampling_input(cls, samples: list, device: str = "cuda"):
        max_seq_len = max([sample["S_true"].shape[1] for sample in samples])
        max_n_atoms = max([sample["X"].shape[1] for sample in samples])
        n_samples = len(samples)

        # Padded default values
        batch = dict(
            X=torch.zeros((n_samples, max_n_atoms, 3), dtype=torch.float32, device=device),
            S_true=torch.ones((n_samples, max_seq_len), dtype=torch.long, device=device)
            * (len(PROTEIN_MPNN_ALPHABET) - 1),
            mask=torch.zeros((n_samples, max_seq_len), dtype=torch.float32, device=device),
            omit_AA_mask=torch.zeros(
                (n_samples, max_seq_len, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device
            ),
            chain_mask=torch.zeros((n_samples, max_n_atoms), dtype=torch.float32, device=device),
            chain_encoding_all=torch.zeros((n_samples, max_n_atoms), dtype=torch.long, device=device),
            residue_idx=torch.zeros((n_samples, max_n_atoms), dtype=torch.long, device=device) - 1,
            chain_M_pos=torch.zeros((n_samples, max_n_atoms), dtype=torch.float32, device=device),
            bias_by_res=torch.zeros(
                (n_samples, max_seq_len, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device
            ),
            pssm_log_odds_mask=torch.zeros(
                (n_samples, max_seq_len, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device
            ),
            pssm_bias=torch.zeros(
                (n_samples, max_seq_len, len(PROTEIN_MPNN_ALPHABET)), dtype=torch.float32, device=device
            ),
            pssm_coef=torch.zeros((n_samples, max_seq_len), dtype=torch.float32, device=device),
            omit_AAs_np=samples[0]["omit_AAs_np"],
            bias_AAs_np=samples[0]["bias_AAs_np"],
        )

        # Fill in values
        for i, sample in enumerate(samples):
            batch["X"][i, : sample["X"].shape[1], :] = sample["X"]
            batch["S_true"][i, : sample["S_true"].shape[1]] = sample["S_true"]
            batch["mask"][i, : sample["mask"].shape[1]] = sample["mask"]
            batch["omit_AA_mask"][i, : sample["omit_AA_mask"].shape[1], :] = sample["omit_AA_mask"]
            batch["chain_mask"][i, : sample["chain_mask"].shape[1]] = sample["chain_mask"]
            batch["chain_encoding_all"][i, : sample["chain_encoding_all"].shape[1]] = sample["chain_encoding_all"]
            batch["residue_idx"][i, : sample["residue_idx"].shape[1]] = sample["residue_idx"]
            batch["chain_M_pos"][i, : sample["chain_M_pos"].shape[1]] = sample["chain_M_pos"]
            batch["bias_by_res"][i, : sample["bias_by_res"].shape[1], :] = sample["bias_by_res"]
            batch["pssm_log_odds_mask"][i, : sample["pssm_log_odds_mask"].shape[1], :] = sample["pssm_log_odds_mask"]
            batch["pssm_bias"][i, : sample["pssm_bias"].shape[1], :] = sample["pssm_bias"]
            batch["pssm_coef"][i, : sample["pssm_coef"].shape[1]] = sample["pssm_coef"]

        return batch

    @classmethod
    def _collate_protein_mpnn_scoring_input(cls, samples: list, device: str = "cuda"):
        max_seq_len = max([sample["S"].shape[1] for sample in samples])
        max_n_atoms = max([sample["X"].shape[1] for sample in samples])
        n_samples = len(samples)

        # Padded default values
        batch = dict(
            X=torch.zeros((n_samples, max_n_atoms, 3), dtype=torch.float32, device=device),
            S=torch.ones((n_samples, max_seq_len), dtype=torch.long, device=device) * (len(PROTEIN_MPNN_ALPHABET) - 1),
            mask=torch.zeros((n_samples, max_seq_len), dtype=torch.float32, device=device),
            chain_M=torch.zeros((n_samples, max_n_atoms), dtype=torch.float32, device=device),
            chain_encoding_all=torch.zeros((n_samples, max_n_atoms), dtype=torch.long, device=device),
            residue_idx=torch.zeros((n_samples, max_n_atoms), dtype=torch.long, device=device) - 1,
        )

        # Fill in values
        for i, sample in enumerate(samples):
            batch["X"][i, : sample["X"].shape[1], :] = sample["X"]
            batch["S"][i, : sample["S"].shape[1]] = sample["S"]
            batch["mask"][i, : sample["mask"].shape[1]] = sample["mask"]
            batch["chain_M"][i, : sample["chain_M"].shape[1]] = sample["chain_M"]
            batch["chain_encoding_all"][i, : sample["chain_encoding_all"].shape[1]] = sample["chain_encoding_all"]
            batch["residue_idx"][i, : sample["residue_idx"].shape[1]] = sample["residue_idx"]

        return batch
