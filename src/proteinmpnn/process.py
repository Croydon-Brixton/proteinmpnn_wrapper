import subprocess
import os
import numpy as np
import glob

from typing import Optional
from .utils import get_logger

logger = get_logger(__name__)

from .constants import PROTEIN_MPNN_DIR, PROTEIN_MPNN_MODELS


def run_protein_mpnn(
        input_dir: str,
        ca_only: bool, 
        output_dir: Optional[str] = None,
        num_seq_per_target: int = 5,
        motif_mask: Optional[np.ndarray] = None,
        max_retries: int = 5,
        seed: int = 38,
        sampling_temp: float = 0.1,
        batch_size: int = 1,
        use_soluble_model: bool = False,
        device: Optional[str] = None):
    """
    Runs the ProteinMPNN as a subprocess on a directory of designed backbones.

    Args:
        input_dir (str): directory where designed protein backbones are stored in `.pdb` format.
        ca_only (bool): whether to only use CA atoms for the ProteinMPNN process.
        output_dir (Optional[str]): directory where ProteinMPNN output will be stored. If 
            None, defaults to input_dir. Defaults to None.
        num_seq_per_target (int): number of sequences to generate per target. Defaults to 5.    
        motif_mask (Optional[np.ndarray]): Optional mask of which residues are the motif. These will 
            be masked out during inverse folding. Defaults to None.
        max_retries (int): maximum number of retries for ProteinMPNN. Defaults to 5.
        seed (int): random seed for ProteinMPNN. Defaults to 38.
        sampling_temp (float): sampling temperature for ProteinMPNN. Defaults to 0.1.
        batch_size (int): batch size for ProteinMPNN. Defaults to 1.
        use_soluble_model (bool): whether to use the soluble model for ProteinMPNN, i.e. no 
            membrane proteins. Defaults to False.
        device (Optional[str]): device to run ProteinMPNN on. If None, defaults to CPU. Defaults to None.

    Returns:
        Writes ProteinMPNN outputs to decoy_pdb_dir/seqs
    """
    input_dir = os.path.abspath(input_dir)
    if output_dir is None:
        output_dir = input_dir
    output_dir = os.path.abspath(output_dir)

    assert os.path.exists(input_dir), f"Input directory not found at {input_dir}"
    assert os.path.isdir(input_dir), f"Input directory is not a directory at {input_dir}"
    # Ensure input_dir has .pdb files
    n_pdbs = len(glob.glob(os.path.join(input_dir, '*.pdb')))
    assert n_pdbs > 0, f"No .pdb files found in {input_dir}"
    logger.info(f"Found {n_pdbs} .pdb files in {input_dir}")

    os.makedirs(output_dir, exist_ok=True)

    # TODO: Integrate motif mask

    # Step 1: Parse backbones into ProteinMPNN compatible format
    logger.info(f"Parsing backbones in {input_dir} into ProteinMPNN compatible format...")
    parsed_pdbs_path = os.path.join(input_dir, "parsed_pdbs.jsonl")
    parser_args = [
        'python',
        f'{PROTEIN_MPNN_DIR}/helper_scripts/parse_multiple_chains.py',
        f'--input_path={input_dir}',
        f'--output_path={parsed_pdbs_path}',
    ]
    if ca_only:
        parser_args.append('--ca_only')

    process = subprocess.Popen(parser_args)
    ret_code = process.wait()
    assert ret_code == 0, f"ProteinMPNN parsing failed with return code {ret_code}."
    logger.info(f"Successfully parsed backbones in {input_dir} into ProteinMPNN compatible format.")

    # Step 2: Run ProteinMPNN
    #  ... set up args
    pmpnn_args = [
        'python',
        f'{PROTEIN_MPNN_DIR}/protein_mpnn_run.py',
        '--out_folder',
        str(output_dir),
        '--jsonl_path',
        str(parsed_pdbs_path),
        '--num_seq_per_target',
        str(num_seq_per_target),
        '--sampling_temp',
        str(sampling_temp),
        '--seed',
        str(seed),
        '--batch_size',
        str(batch_size),
        '--use_soluble_model',
        str(use_soluble_model),
    ]
    if device is not None:
        pmpnn_args.append('--device')
        pmpnn_args.append(str(device))
    
    #  ... run ProteinMPNN for `max_retries`
    success = False
    retries = 0
    ret_code = -1
    logger.info(f"Running ProteinMPNN on {n_pdbs} ({num_seq_per_target} times each) in {input_dir}...")
    while not success:
        try:
            process = subprocess.Popen(
                pmpnn_args,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT
            )
            ret_code = process.wait()
            success = (ret_code == 0)
        except Exception as e:
            retries += 1
            logger.warning(f'ProteinMPNN with return code {ret_code}. Attempt {retries}/{max_retries}')
            try:
                torch.cuda.empty_cache()
            except:
                pass
            if retries >= max_retries:
                raise e
    logger.info(f"Successfully ran ProteinMPNN.")
            
    # Step 3: Gather ProteinMPNN outputs
    #  ... get paths to ProteinMPNN outputs
    seqs_dir = os.path.join(output_dir, "seqs")
    assert os.path.exists(seqs_dir), f"ProteinMPNN seqs directory not found at {seqs_dir}"
    return seqs_dir