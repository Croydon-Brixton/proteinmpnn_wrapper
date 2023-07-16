import os
from pathlib import Path
import logging

SRC_PATH = Path(os.path.abspath(__file__)).parent
PROTEIN_MPNN_DIR = SRC_PATH / "ProteinMPNN"

assert PROTEIN_MPNN_DIR.exists(), f"ProteinMPNN directory not found at {PROTEIN_MPNN_DIR}"

# Collect all valid ProteinMPNN models
PROTEIN_MPNN_MODELS = {}
for model_path in PROTEIN_MPNN_DIR.glob("*/*.pt"):
    model_type = model_path.parent.name.replace("_model_weights", "")
    if model_type not in PROTEIN_MPNN_MODELS:
        PROTEIN_MPNN_MODELS[model_type] = [str(model_path)]
    else:
        PROTEIN_MPNN_MODELS[model_type].append(str(model_path))

# Logging constants
DEFAULT_LOG_FORMATTER = logging.Formatter(
    "%(asctime)s %(levelname)s:\n\t%(message)s [in %(funcName)s at %(filename)s:%(lineno)d]"
)
DEFAULT_LOG_LEVEL = logging.INFO  # verbose logging per default