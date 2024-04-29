import random
from typing import Literal, Optional

import numpy as np
import torch

from .constants import PROTEIN_MPNN_MODELS
from .utils.log_utils import get_logger

logger = get_logger(__name__)

__all__ = ["load_protein_mpnn_model", "nll_score", "set_seed"]


def set_seed(seed: int):
    """Set random seeds"""
    if seed is None:
        seed = int(np.random.randint(0, high=int(1e5), size=1, dtype=int)[0])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    logger.debug(f"Set seed to %d" % seed)


def nll_score(tokenised_seq: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """Compute negative log likelihood probabilities for a given sequence."""
    if mask is None:
        mask = torch.ones_like(tokenised_seq)
    criterion = torch.nn.NLLLoss(reduction="none")
    loss = criterion(log_probs.contiguous().view(-1, log_probs.size(-1)), tokenised_seq.contiguous().view(-1)).view(
        tokenised_seq.size()
    )
    score = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
    return score


def load_protein_mpnn_model(
    model_type: Literal["vanilla", "ca", "soluble"] = "vanilla",
    model_name: str = "v_48_020",
    backbone_noise: float = 0.0,
    device: Optional[str] = None,
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
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

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
