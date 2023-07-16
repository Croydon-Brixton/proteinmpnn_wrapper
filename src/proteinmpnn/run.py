import io
import pathlib
import random
from typing import Union

import biotite.structure as bs
import numpy as np
import torch

from .constants import AA_3_TO_1, PROTEIN_MPNN_ALPHABET, PROTEIN_MPNN_MODELS
from .utils.biotite_utils import atom_array_from_numpy, load_structure
from .utils.log_utils import get_logger

logger = get_logger(__name__)

__all__ = ["load_protein_structure", "load_protein_mpnn_model", "design_sequences"]

_REVERSE_ALPHABET = {v: k for k, v in PROTEIN_MPNN_ALPHABET.items()}


def tokenise_sequence(seq: str) -> torch.Tensor:
    return torch.tensor([PROTEIN_MPNN_ALPHABET[aa] for aa in seq], dtype=torch.long)


def untokenise_sequence(x: torch.Tensor) -> str:
    assert x.dtype == torch.long
    return "".join([_REVERSE_ALPHABET[i.item()] for i in x.squeeze()])


def _set_seed(seed: int):
    if seed is None:
        seed = int(np.random.randint(0, high=int(1e5), size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Set seed to %d" % seed)


def load_protein_structure(
    file_or_buffer_or_obj: Union[str, pathlib.Path, bs.AtomArray, io.StringIO, io.BytesIO, np.ndarray, torch.Tensor],
    **kwargs,
) -> bs.AtomArray:
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
        atom_array (bs.AtomArray): The loaded atom array.
    """
    if isinstance(file_or_buffer_or_obj, (str, pathlib.Path, io.StringIO, io.BytesIO)):
        # Check if `model` in kwargs
        if "model" not in kwargs:
            kwargs["model"] = 1
        return load_structure(file_or_buffer_or_obj, **kwargs)
    elif isinstance(file_or_buffer_or_obj, bs.AtomArray):
        return file_or_buffer_or_obj
    elif isinstance(file_or_buffer_or_obj, np.ndarray):
        assert "ca_only" in kwargs, "Must specify `ca_only` if passing a numpy array."
        return atom_array_from_numpy(file_or_buffer_or_obj, **kwargs)
    elif isinstance(file_or_buffer_or_obj, torch.Tensor):
        return load_protein_structure(file_or_buffer_or_obj.detach().numpy(), **kwargs)


def load_protein_mpnn_model(
    model_type="vanilla", model_name="v_48_020", backbone_noise: float = 0.0, device: str = "cpu"
):
    assert (
        model_type in PROTEIN_MPNN_MODELS.keys()
    ), f"Model type `{model_type}` not found. Must be one of {list(PROTEIN_MPNN_MODELS.keys())}"
    assert (
        model_name in PROTEIN_MPNN_MODELS[model_type]
    ), f"Model name `{model_name}` not found. Must be one of {list(PROTEIN_MPNN_MODELS[model_type].keys())}"

    checkpoint_path = PROTEIN_MPNN_MODELS[model_type][model_name]
    hidden_dim = 128
    num_layers = 3

    checkpoint = torch.load(checkpoint_path, map_location=device)
    logger.info("Number of edges: %s", str(checkpoint["num_edges"]))
    noise_level_print = checkpoint["noise_level"]
    logger.info(f"Training noise level: {noise_level_print}A")

    from .ProteinMPNN.protein_mpnn_utils import ProteinMPNN

    model = ProteinMPNN(
        ca_only=True if model_type == "ca" else False,
        num_letters=21,
        node_features=hidden_dim,
        edge_features=hidden_dim,
        hidden_dim=hidden_dim,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        augment_eps=backbone_noise,
        k_neighbors=checkpoint["num_edges"],
    )
    model.to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Set flag to mark whether the model is for CA atoms only
    model.ca_only = True if model_type == "ca" else False
    model.device = device

    return model


def prepare_structure_for_mpnn(protein: bs.AtomArray, chain: str, ca_only: bool = True):
    protein = protein[bs.filter_amino_acids(protein) & (protein.chain_id == chain)]
    backbone = protein[np.isin(protein.atom_name, ["N", "CA", "C", "O"])]

    if ca_only:
        backbone = backbone[backbone.atom_name == "CA"]
        coords = torch.tensor(backbone.coord.reshape(1, -1, 3), dtype=torch.float32)
    else:
        coords = torch.tensor(backbone.coord.reshape(1, -1, 4, 3), dtype=torch.float32)

    seq = bs.residues.get_residues(backbone[backbone.chain_id == chain])[1]
    seq = "".join([AA_3_TO_1[res] for res in seq])
    tokenised_seq = tokenise_sequence(seq).unsqueeze(0)

    mask = torch.ones_like(tokenised_seq, dtype=torch.float32)
    chain_M = torch.ones_like(tokenised_seq, dtype=torch.float32)
    residue_idx = torch.arange(tokenised_seq.shape[1], dtype=torch.long).unsqueeze(0)
    chain_encoding_all = torch.ones_like(tokenised_seq, dtype=torch.long)
    randn = torch.randn_like(tokenised_seq, dtype=torch.float32)

    return coords, tokenised_seq, mask, chain_M, residue_idx, chain_encoding_all, randn


class BackboneDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list, ca_only: bool = True):
        self.samples = samples
        self.ca_only = ca_only

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        structure = load_protein_structure(self.samples[idx], ca_only=self.ca_only)
        return prepare_structure_for_mpnn(structure, "A", ca_only=self.ca_only)


def design_sequences(
    inputs,
    model: "ProteinMPNN",
    sampling_temp: float = 0.1,
    num_seq_per_target: int = 1,
    batch_size: int = 1,
    seed: int = 38,
) -> list:
    if batch_size > 1:
        # TODO: implement batched sampling
        raise NotImplementedError("Batched sampling not implemented yet.")

    _set_seed(seed)

    model.device
    ca_only = model.ca_only

    samples = []
    for input in inputs:
        # Sample loading (turn into torch data loader)
        structure = load_protein_structure(input, ca_only=ca_only)

        X, S, mask, chain_M, residue_idx, chain_encoding_all, randn = prepare_structure_for_mpnn(
            structure, "A", ca_only=ca_only
        )

        # Amino acids that are not allowed to be sampled at a given position
        omit_AA_mask = torch.zeros((1, S.shape[1], 21), dtype=torch.float32)
        omit_AAs_np = np.zeros(21)
        omit_AAs_np[-1] = 1.0  # disallow the `mask` token

        # Amino acids that are preferred to be sampled
        bias_AAs_np = np.zeros(21)
        bias_by_res = torch.zeros((1, S.shape[1], 21), dtype=torch.float32)

        # PSSM (position-specific scoring matrix) bias
        pssm_log_odds_mask = torch.ones((1, S.shape[1], 21), dtype=torch.float32)
        pssm_bias = torch.zeros((1, S.shape[1], 21), dtype=torch.float32)
        pssm_coef = torch.zeros((1, S.shape[1]), dtype=torch.float32)

        # Evaluation
        with torch.inference_mode():
            for _ in range(num_seq_per_target):
                randn = torch.randn_like(randn)
                samples.append(
                    model.sample(
                        X,
                        randn,
                        S,
                        chain_M,
                        chain_encoding_all,
                        residue_idx,
                        mask=mask,
                        temperature=sampling_temp,
                        chain_M_pos=torch.ones_like(chain_M),
                        omit_AA_mask=omit_AA_mask,
                        omit_AAs_np=omit_AAs_np,
                        bias_AAs_np=bias_AAs_np,
                        bias_by_res=bias_by_res,
                        pssm_bias_flag=False,
                        pssm_log_odds_flag=False,
                        pssm_log_odds_mask=pssm_log_odds_mask,
                        pssm_bias=pssm_bias,
                    )
                )
    return samples
