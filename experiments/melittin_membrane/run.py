"""Sample melittin with BioEmu under each guidance method.

    python experiments/melittin_membrane/run.py --num-samples 500
    python experiments/melittin_membrane/run.py --methods baseline dps --dps-scales 0.5 1.69 5

Each run writes to <out>/<run_name>/ (BioEmu's batch_*.npz, topology.pdb, samples.xtc,
unfiltered) plus run_info.json with the config and wall time. All runs share the same
base_seed, so the initial noise is identical across methods (common random numbers).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import torch
import yaml

from bioemu.sample import main as bioemu_sample

MELITTIN = "GIGAVLKVLTTGLPALISWIKRKRQQ"
HERE = Path(__file__).resolve().parent


def load_config(name: str) -> dict:
    with open(HERE / "configs" / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_runs(methods: list[str], dps_scales: list[float]) -> list[tuple[str, dict]]:
    """One run per config, except configs exposing `guidance_scale`, which get a lambda sweep."""
    runs = []
    for name in methods:
        cfg = load_config(name)
        if "guidance_scale" in cfg and dps_scales:
            for lam in dps_scales:
                swept = load_config(name)
                swept["guidance_scale"] = lam
                runs.append((f"{name}_lam{lam:g}", swept))
        else:
            runs.append((name, cfg))
    return runs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sequence", default=MELITTIN)
    p.add_argument("--methods", nargs="+", default=["baseline", "dps", "dps_phys", "smc", "fkc"])
    p.add_argument("--dps-scales", nargs="+", type=float, default=[1.69, 5.0])
    p.add_argument("--num-samples", type=int, default=500)
    p.add_argument("--batch-size-100", type=int, default=10)
    p.add_argument("--model-name", default="bioemu-v1.1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(HERE / "results"))
    p.add_argument("--cache-embeds-dir", default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO)
    out = Path(args.out)
    for run_name, cfg in build_runs(args.methods, args.dps_scales):
        run_dir = out / run_name
        print(f"=== {run_name} -> {run_dir}", flush=True)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        bioemu_sample(
            sequence=args.sequence,
            num_samples=args.num_samples,
            output_dir=run_dir,
            batch_size_100=args.batch_size_100,
            model_name=args.model_name,
            denoiser_config=cfg,
            cache_embeds_dir=args.cache_embeds_dir,
            filter_samples=False,  # keep everything; analyze.py reports the physical fraction
            base_seed=args.seed,
        )
        info = {
            "run": run_name,
            "config": cfg,
            "sequence": args.sequence,
            "num_samples": args.num_samples,
            "model_name": args.model_name,
            "seed": args.seed,
            "wall_time_s": time.time() - t0,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "peak_gpu_mem_gb": (
                torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None
            ),
        }
        (run_dir / "run_info.json").write_text(json.dumps(info, indent=2))
        print(f"    done in {info['wall_time_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
