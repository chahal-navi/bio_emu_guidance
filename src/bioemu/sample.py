# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script for sampling from a trained model."""

import logging
import time
import typing
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import hydra
import numpy as np
import torch
import yaml
from omegaconf import DictConfig
from torch_geometric.data.batch import Batch
from tqdm import tqdm

from bioemu.chemgraph import ChemGraph
from bioemu.convert_chemgraph import save_pdb_and_xtc
from bioemu.get_embeds import get_colabfold_embeds
from bioemu.model_utils import load_model, load_sdes, maybe_download_checkpoint
from bioemu.sde_lib import SDE
from bioemu.seq_io import check_protein_valid, parse_sequence, write_fasta
from bioemu.utils import (
    count_samples_in_output_dir,
    format_npz_samples_filename,
    print_traceback_on_exception,
)

# --- IMPORTING YOUR CUSTOM SOLVER ---
from bioemu.denoiser import membrane_guided_dpm_solver
# ------------------------------------

logger = logging.getLogger(__name__)

DEFAULT_DENOISER_CONFIG_DIR = Path(__file__).parent / "config/denoiser/"
SupportedDenoisersLiteral = Literal["dpm", "heun"]
SUPPORTED_DENOISERS = list(typing.get_args(SupportedDenoisersLiteral))

# Mapping used in training of BioEmu-1.2 model.
_NODE_LABEL_MAPPING: dict[str, int] = {
    "A": 1, "R": 15, "N": 12, "D": 3, "C": 2, "Q": 14, "E": 4, "G": 6,
    "H": 7, "I": 8, "L": 10, "K": 9, "M": 11, "F": 5, "P": 13, "S": 16,
    "T": 17, "W": 19, "Y": 20, "V": 18, "U": 21, "O": 22, "X": 0, "B": 23, "Z": 25,
}

@print_traceback_on_exception
@torch.no_grad()
def main(
    sequence: str | Path,
    num_samples: int,
    output_dir: str | Path,
    batch_size_100: int = 10,
    model_name: Literal["bioemu-v1.0", "bioemu-v1.1", "bioemu-v1.2"] | None = "bioemu-v1.1",
    ckpt_path: str | Path | None = None,
    model_config_path: str | Path | None = None,
    denoiser_type: SupportedDenoisersLiteral | None = "dpm",
    denoiser_config: str | Path | dict | None = None,
    cache_embeds_dir: str | Path | None = None,
    cache_so3_dir: str | Path | None = None,
    msa_host_url: str | None = None,
    filter_samples: bool = True,
    base_seed: int | None = None,
    membrane_csv_path: str | Path | None = None,
) -> None:
    
    if base_seed is None:
        base_seed = time.time_ns()

    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path, model_config_path = maybe_download_checkpoint(
        model_name=model_name, ckpt_path=ckpt_path, model_config_path=model_config_path
    )
    score_model = load_model(ckpt_path, model_config_path)

    sdes = load_sdes(model_config_path=model_config_path, cache_so3_dir=cache_so3_dir)

    msa_file = sequence if str(sequence).endswith(".a3m") else None
    if msa_file is not None and msa_host_url is not None:
        logger.warning(f"msa_host_url is ignored because MSA file {msa_file} is provided.")

    sequence = parse_sequence(sequence)
    check_protein_valid(sequence)

    fasta_path = output_dir / "sequence.fasta"
    if fasta_path.is_file():
        if parse_sequence(fasta_path) != sequence:
            raise ValueError(f"{fasta_path} already exists, but contains a sequence different from {sequence}!")
    else:
        write_fasta([sequence], fasta_path)

    # Note: Hydra still loads the default config here to prevent initialization errors, 
    # but we will ignore it down in generate_batch.
    if denoiser_config is None:
        denoiser_config = DEFAULT_DENOISER_CONFIG_DIR / f"{denoiser_type}.yaml"
        with open(denoiser_config) as f:
            denoiser_config = yaml.safe_load(f)
    elif isinstance(denoiser_config, str | Path):
        denoiser_config_path = Path(denoiser_config).expanduser().resolve()
        with open(denoiser_config_path) as f:
            denoiser_config = yaml.safe_load(f)

    denoiser = hydra.utils.instantiate(denoiser_config)

    logger.info(f"Sampling {num_samples} structures for sequence of length {len(sequence)} residues...")
    
    batch_size = max(1, int(batch_size_100 * (100 / len(sequence)) ** 2))
    batch_size = min(batch_size, num_samples)
    logger.info(f"Using batch size {min(batch_size, num_samples)}")

    existing_num_samples = count_samples_in_output_dir(output_dir)
    logger.info(f"Found {existing_num_samples} previous samples in {output_dir}.")
    
    for start_idx in tqdm(range(existing_num_samples, num_samples, batch_size), desc="Sampling batches..."):
        n = min(batch_size, num_samples - start_idx)
        npz_path = output_dir / format_npz_samples_filename(start_idx, n)
        if npz_path.exists():
            raise ValueError(f"Not sure why {npz_path} already exists.")
        
        seed = base_seed + start_idx
        logger.info(f"Sampling with {seed=} ({base_seed=})")
        
        batch = generate_batch(
            score_model=score_model,
            sequence=sequence,
            sdes=sdes,
            batch_size=min(batch_size, n),
            seed=seed,
            denoiser=denoiser,
            cache_embeds_dir=cache_embeds_dir,
            msa_file=msa_file,
            msa_host_url=msa_host_url,
            membrane_csv_path=membrane_csv_path,
        )

        batch = {k: v.cpu().numpy() for k, v in batch.items()}
        np.savez(npz_path, **batch, sequence=sequence)

    logger.info("Converting samples to .pdb and .xtc...")
    samples_files = sorted(list(output_dir.glob("batch_*.npz")))
    sequences = [np.load(f)["sequence"].item() for f in samples_files]
    positions = torch.tensor(np.concatenate([np.load(f)["pos"] for f in samples_files]))
    node_orientations = torch.tensor(np.concatenate([np.load(f)["node_orientations"] for f in samples_files]))
    
    save_pdb_and_xtc(
        pos_nm=positions,
        node_orientations=node_orientations,
        topology_path=output_dir / "topology.pdb",
        xtc_path=output_dir / "samples.xtc",
        sequence=sequence,
        filter_samples=filter_samples,
    )

    logger.info(f"Completed. Your samples are in {output_dir}.")

def get_context_chemgraph(
    sequence: str, cache_embeds_dir: str | Path | None = None,
    msa_file: str | Path | None = None, msa_host_url: str | None = None,
) -> ChemGraph:
    n = len(sequence)
    single_embeds_file, pair_embeds_file = get_colabfold_embeds(
        seq=sequence, cache_embeds_dir=cache_embeds_dir, msa_file=msa_file, msa_host_url=msa_host_url,
    )
    single_embeds = torch.from_numpy(np.load(single_embeds_file))
    pair_embeds = torch.from_numpy(np.load(pair_embeds_file))
    pair_embeds = pair_embeds.view(n**2, pair_embeds.shape[2])

    edge_index = torch.cat([
        torch.arange(n).repeat_interleave(n).view(1, n**2),
        torch.arange(n).repeat(n).view(1, n**2),
    ], dim=0)
    pos = torch.full((n, 3), float("nan"))
    node_orientations = torch.full((n, 3, 3), float("nan"))
    node_labels = torch.LongTensor([_NODE_LABEL_MAPPING[aa] for aa in sequence])

    return ChemGraph(
        edge_index=edge_index, pos=pos, node_orientations=node_orientations,
        single_embeds=single_embeds, pair_embeds=pair_embeds,
        sequence=sequence, node_labels=node_labels,
    )

def generate_batch(
    score_model: torch.nn.Module,
    sequence: str,
    sdes: dict[str, SDE],
    batch_size: int,
    seed: int,
    denoiser: Callable,
    cache_embeds_dir: str | Path | None,
    msa_file: str | Path | None = None,
    msa_host_url: str | None = None,
    membrane_csv_path: str | Path | None = None,
) -> dict[str, torch.Tensor]:

    torch.manual_seed(seed)

    context_chemgraph = get_context_chemgraph(
        sequence=sequence, cache_embeds_dir=cache_embeds_dir,
        msa_file=msa_file, msa_host_url=msa_host_url,
    )

    context_batch = Batch.from_data_list([context_chemgraph] * batch_size)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # CUSTOM MEMBRANE GUIDANCE OVERRIDE
    # We ignore the Hydra `denoiser` argument and call our custom 
    # solver directly. Hardcoding default DPM timesteps here for testing.
    # -------------------------------------------------------------
    logger.info("Executing custom membrane_guided_dpm_solver...")
    result = membrane_guided_dpm_solver(
        sdes=sdes,
        batch=context_batch,
        N=100,                # Number of timesteps
        score_model=score_model,
        max_t=0.999,          # Max noise level
        eps_t=1e-3,           # Min noise level
        device=device,
        membrane_csv_path=membrane_csv_path,
    )
    # -------------------------------------------------------------

    if isinstance(result, tuple):
        sampled_chemgraph_batch, _ = result
    else:
        sampled_chemgraph_batch = result
        
    sampled_chemgraphs = sampled_chemgraph_batch.to_data_list()
    pos = torch.stack([x.pos for x in sampled_chemgraphs]).to("cpu") 
    node_orientations = torch.stack([x.node_orientations for x in sampled_chemgraphs]).to("cpu")

    return {"pos": pos, "node_orientations": node_orientations}

if __name__ == "__main__":
    import fire
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
