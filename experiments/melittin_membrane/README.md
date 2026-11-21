# Membrane-aware guidance of BioEmu: melittin

Compares four samplers on melittin (`GIGAVLKVLTTGLPALISWIKRKRQQ`), all with the same
Ez membrane potential (Senes et al. 2007; `bioemu.steering.custom_potentials.EzMembranePotential`)
evaluated at each structure's optimal membrane placement:

| run | sampler | config |
|---|---|---|
| `baseline` | DPM-Solver-2, no guidance | `configs/baseline.yaml` |
| `dps_lam*` | DPS gradient guidance on x̂₀ (Path B), λ sweep | `configs/dps.yaml` |
| `smc` | BioEmu SMC, resampling only (Path A) | `configs/smc.yaml` |
| `fkc` | BioEmu FKC: gradient guidance + importance weights (Path A) | `configs/fkc.yaml` |

λ = 1/kT = 1.69 mol/kcal is a Boltzmann tilt `p_BioEmu(x)·exp(−E_Ez(x)/kT)`.

```bash
bash experiments/melittin_membrane/setup_runpod.sh          # once, on the GPU machine
python experiments/melittin_membrane/run.py --num-samples 500
python experiments/melittin_membrane/analyze.py              # -> results/analysis/
```

Notes
- `frame: optimize` finds the membrane normal/midplane minimizing Ez per structure, so guidance
  acts on conformation, not rigid-body pose. `frame: kabsch` + `csv_path` reproduces Pratham's
  reference-aligned version (`bioemu/membrane_csv.py` builds the CSV from an OPM file).
- Guidance is applied at both DPM-Solver-2 evaluations (t and the midpoint); the position update
  uses only the midpoint score, so guiding only at t has almost no effect
  (see `tests/steering/test_membrane_guidance.py`).
