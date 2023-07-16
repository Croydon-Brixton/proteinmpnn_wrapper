import logging
import os
from pathlib import Path

SRC_PATH = Path(os.path.abspath(__file__)).parent
PROTEIN_MPNN_DIR = SRC_PATH / "ProteinMPNN"

assert PROTEIN_MPNN_DIR.exists(), f"ProteinMPNN directory not found at {PROTEIN_MPNN_DIR}"

# Collect all valid ProteinMPNN models
PROTEIN_MPNN_MODELS = {}
for model_path in PROTEIN_MPNN_DIR.glob("*/*.pt"):
    model_type = model_path.parent.name.replace("_model_weights", "")
    if model_type not in PROTEIN_MPNN_MODELS:
        PROTEIN_MPNN_MODELS[model_type] = {model_path.stem: str(model_path)}
    else:
        PROTEIN_MPNN_MODELS[model_type][model_path.stem] = str(model_path)

_protein_mpnn_alphabet = "ACDEFGHIKLMNPQRSTVWYX"
PROTEIN_MPNN_ALPHABET = {aa: i for i, aa in enumerate(_protein_mpnn_alphabet)}


# Amino acid related constants
AA_3_TO_1 = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}

AA_1_TO_3 = {v: k for k, v in AA_3_TO_1.items()}

# Logging constants
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s:\n\t%(message)s [in %(funcName)s at %(filename)s:%(lineno)d]"
)
DEFAULT_LOG_LEVEL = logging.INFO  # verbose logging per default
